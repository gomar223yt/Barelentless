/*
 * This file is part of Baritone.
 *
 * Baritone is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Baritone is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Baritone.  If not, see <https://www.gnu.org/licenses/>.
 */

package baritone.ml;

import baritone.api.ml.ICostAdjuster;
import baritone.api.ml.memory.EpisodicMemory;
import baritone.utils.BlockStateInterface;
import it.unimi.dsi.fastutil.longs.Long2DoubleOpenHashMap;
import net.minecraft.world.level.block.state.BlockState;

/**
 * Applies what the episodic memory knows to the pathfinder's cost estimates.
 * <p>
 * The pathfinder asks about hundreds of thousands of candidate movements per calculation, so a memory lookup per
 * candidate is out of the question. What makes this affordable is that the situations repeat enormously: within one
 * path calculation, "traverse north from stone onto stone with air above" comes up thousands of times and is the same
 * question every time. So the descriptor of a situation - movement kind plus the identity of the few blocks that
 * decide the outcome - is hashed, and the answer is cached against that hash for the rest of the calculation. The
 * expensive lookup happens once per distinct situation, not once per candidate.
 * <p>
 * The adjustment itself is bounded on purpose. A learned multiplier that could reach zero would let one bad memory
 * convince A* that a route through lava is free; the clamp means the memory can bias the search substantially while
 * never being able to override the hand-written cost functions completely.
 *
 * @author Barelentless
 */
public final class LearnedCostAdjuster implements ICostAdjuster {

    /**
     * Key dimension for the memory. Small on purpose: this describes a movement's immediate surroundings, not the
     * full player state, and a compact key is what keeps recall fast.
     */
    public static final int KEY_FEATURES = 24;

    private static final double MIN_MULTIPLIER = 0.6;
    private static final double MAX_MULTIPLIER = 2.5;

    private final EpisodicMemory memory;
    private final BlockStateInterface bsi;
    private final double influence;

    /**
     * Per-calculation cache of multiplier by situation hash. Unsynchronized: one adjuster belongs to one calculation
     * on one thread.
     */
    private final Long2DoubleOpenHashMap cache = new Long2DoubleOpenHashMap();

    private long adjusted;
    private long queries;

    public LearnedCostAdjuster(EpisodicMemory memory, BlockStateInterface bsi, double influence) {
        this.memory = memory;
        this.bsi = bsi;
        this.influence = influence;
        this.cache.defaultReturnValue(Double.NaN);
    }

    @Override
    public double adjust(int moveOrdinal, int srcX, int srcY, int srcZ, int destX, int destY, int destZ, double cost) {
        this.queries++;
        if (!this.bsi.worldContainsLoadedChunk(destX, destZ)) {
            return cost;
        }
        BlockState under = this.bsi.get0(srcX, srcY - 1, srcZ);
        BlockState into = this.bsi.get0(destX, destY, destZ);
        BlockState landing = this.bsi.get0(destX, destY - 1, destZ);
        BlockState above = this.bsi.get0(destX, destY + 1, destZ);

        long hash = situationHash(moveOrdinal, destY - srcY, under, into, landing, above);
        double multiplier = this.cache.get(hash);
        if (Double.isNaN(multiplier)) {
            multiplier = lookup(moveOrdinal, destY - srcY, under, into, landing, above);
            this.cache.put(hash, multiplier);
        }
        if (multiplier == 1.0) {
            return cost;
        }
        this.adjusted++;
        return cost * multiplier;
    }

    private double lookup(int moveOrdinal, int deltaY, BlockState under, BlockState into,
                          BlockState landing, BlockState above) {
        EpisodicMemory.Estimate estimate =
                this.memory.estimate(key(deltaY, under, into, landing, above), moveOrdinal, 8);
        if (estimate.isEmpty()) {
            return 1.0;
        }
        // the memory stores the ratio of real to estimated cost, so the multiplier is that ratio, faded in by how
        // much the memory trusts itself and by the configured influence
        double blend = estimate.confidence * this.influence;
        double multiplier = 1.0 + (estimate.outcome - 1.0) * blend;
        // a movement that has been observed to fail is penalised beyond its cost ratio: failures do not show up as
        // "slow", they show up as a path that has to be recalculated
        if (estimate.successRate < 0.9) {
            multiplier *= 1.0 + (0.9 - estimate.successRate) * blend;
        }
        return Math.max(MIN_MULTIPLIER, Math.min(MAX_MULTIPLIER, multiplier));
    }

    /**
     * Builds the memory key for a situation. Block identity is hashed into a small number of dimensions rather than
     * one-hot encoded, which keeps the key short while letting two different blocks that always appear in the same
     * role still be told apart by the rest of the vector.
     */
    public static float[] key(int deltaY, BlockState under, BlockState into, BlockState landing, BlockState above) {
        float[] key = new float[KEY_FEATURES];
        int index = 0;
        key[index++] = deltaY;
        key[index++] = under.isAir() ? 1f : 0f;
        key[index++] = into.isAir() ? 1f : 0f;
        key[index++] = landing.isAir() ? 1f : 0f;
        key[index++] = above.isAir() ? 1f : 0f;
        key[index++] = into.getFluidState().isEmpty() ? 0f : 1f;
        key[index++] = landing.getFluidState().isEmpty() ? 0f : 1f;
        key[index++] = under.getBlock().getFriction() - 0.6f;
        key[index++] = landing.getBlock().getFriction() - 0.6f;
        index = hashInto(under, key, index);
        index = hashInto(into, key, index);
        index = hashInto(landing, key, index);
        hashInto(above, key, index);
        return key;
    }

    /**
     * Spreads a block's identity across three dimensions of the key. Deterministic across sessions because it is
     * derived from the block's own hash, which matters: a memory file has to still mean the same thing tomorrow.
     */
    private static int hashInto(BlockState state, float[] key, int index) {
        int hash = state.getBlock().hashCode();
        for (int i = 0; i < 3 && index < key.length; i++) {
            hash = hash * 0x9E3779B1 + 0x85EBCA6B;
            key[index++] = ((hash >>> 8) % 2000) / 1000f - 1f;
        }
        return index;
    }

    private static long situationHash(int moveOrdinal, int deltaY, BlockState under, BlockState into,
                                      BlockState landing, BlockState above) {
        long hash = moveOrdinal * 31L + deltaY;
        hash = hash * 0x9E3779B97F4A7C15L + under.getBlock().hashCode();
        hash = hash * 0x9E3779B97F4A7C15L + into.getBlock().hashCode();
        hash = hash * 0x9E3779B97F4A7C15L + landing.getBlock().hashCode();
        hash = hash * 0x9E3779B97F4A7C15L + above.getBlock().hashCode();
        return hash;
    }

    @Override
    public long[] statistics() {
        return new long[]{this.adjusted, this.queries};
    }

    public int cachedSituations() {
        return this.cache.size();
    }
}
