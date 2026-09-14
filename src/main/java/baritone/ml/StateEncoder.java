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

import baritone.api.pathing.movement.IMovement;
import baritone.api.utils.BetterBlockPos;
import baritone.api.utils.IPlayerContext;
import baritone.api.utils.Rotation;
import baritone.pathing.movement.MovementHelper;
import baritone.utils.BlockStateInterface;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.Vec3;

/**
 * Turns the game state into the fixed length vector the models consume.
 * <p>
 * What goes in here decides what the models are able to learn, so the choices matter more than the network
 * architecture does. Three principles shape it:
 * <p>
 * Everything is expressed in the player's own frame, not the world's. Velocity, the direction to the destination and
 * the surrounding terrain are all rotated by the player's yaw, so "a ledge ahead and to the right while moving
 * forward at full speed" is the same vector whether it happens facing north or facing south. Without this the model
 * would have to learn every situation four times over and still fail on diagonals.
 * <p>
 * Terrain is described by what it does, not by what it is. Each sampled block contributes whether it can be walked
 * through, whether it can be stood on, whether it is liquid and how slippery it is - so ice generalizes to packed ice
 * for free, and a block added by a mod that behaves like stone is treated like stone on first sight.
 * <p>
 * Nothing unbounded goes in raw. Coordinates, tick counters and distances are either dropped, wrapped into a local
 * offset, or squashed, because a feature that grows without limit will eventually dominate every gradient.
 *
 * @author Barelentless
 */
public final class StateEncoder {

    /**
     * Horizontal radius of the sampled terrain column grid.
     */
    public static final int GRID_RADIUS = 1;

    /**
     * Vertical samples per column: one below the feet, feet level, head level, and one above.
     */
    public static final int COLUMN_HEIGHT = 4;

    private static final int GRID_WIDTH = GRID_RADIUS * 2 + 1;
    private static final int BLOCK_FEATURES = 4;

    /**
     * Number of terrain features: a {@code 3x3} grid of {@code 4}-tall columns, four descriptors each.
     */
    public static final int TERRAIN_FEATURES = GRID_WIDTH * GRID_WIDTH * COLUMN_HEIGHT * BLOCK_FEATURES;

    /**
     * Number of features describing the player itself.
     */
    public static final int SELF_FEATURES = 20;

    /**
     * Number of features describing the current movement and its target.
     */
    public static final int TARGET_FEATURES = 12;

    /**
     * Total length of an encoded state.
     */
    public static final int FEATURES = SELF_FEATURES + TARGET_FEATURES + TERRAIN_FEATURES;

    private StateEncoder() {}

    /**
     * Encodes the live player state. Must be called on the client thread.
     *
     * @param ctx      The player context
     * @param bsi      A block accessor valid for this thread
     * @param movement The movement being executed, or {@code null}
     * @param elapsed  Ticks spent on that movement so far
     * @return A newly allocated feature vector of length {@link #FEATURES}
     */
    public static float[] encode(IPlayerContext ctx, BlockStateInterface bsi, IMovement movement, int elapsed) {
        float[] features = new float[FEATURES];
        if (ctx.player() == null) {
            return features;
        }
        int index = 0;
        index = encodeSelf(ctx, features, index);
        index = encodeTarget(ctx, movement, elapsed, features, index);
        encodeTerrain(bsi, ctx.playerFeet(), ctx.player().getYRot(), features, index);
        return features;
    }

    private static int encodeSelf(IPlayerContext ctx, float[] features, int index) {
        Vec3 velocity = ctx.playerMotion();
        float yaw = ctx.player().getYRot();
        double[] local = rotateToLocal(velocity.x, velocity.z, yaw);

        features[index++] = (float) local[0] * 10f;                 // forward speed, scaled into a sane range
        features[index++] = (float) local[1] * 10f;                 // lateral speed
        features[index++] = (float) velocity.y * 10f;
        features[index++] = (float) Math.sqrt(velocity.x * velocity.x + velocity.z * velocity.z) * 10f;
        features[index++] = ctx.player().onGround() ? 1f : 0f;
        features[index++] = ctx.player().isInWater() ? 1f : 0f;
        features[index++] = ctx.player().isInLava() ? 1f : 0f;
        features[index++] = ctx.player().isFallFlying() ? 1f : 0f;
        features[index++] = ctx.player().isSprinting() ? 1f : 0f;
        features[index++] = ctx.player().isShiftKeyDown() ? 1f : 0f;
        features[index++] = ctx.player().onClimbable() ? 1f : 0f;
        features[index++] = Math.min(1f, ctx.player().fallDistance / 8f);
        features[index++] = ctx.player().getHealth() / 20f;
        features[index++] = ctx.player().getFoodData().getFoodLevel() / 20f;
        // position within the block, which is what actually decides whether the next step lands on the edge
        features[index++] = (float) fraction(ctx.player().position().x);
        features[index++] = (float) fraction(ctx.player().position().z);
        features[index++] = (float) fraction(ctx.player().position().y);
        // pitch as a pair, and yaw only as a rate: absolute yaw is meaningless once everything else is local
        features[index++] = (float) Math.sin(Math.toRadians(ctx.player().getXRot()));
        features[index++] = (float) Math.cos(Math.toRadians(ctx.player().getXRot()));
        features[index++] = Rotation.normalizeYaw(ctx.player().getYRot() - ctx.player().yRotO) / 45f;
        return index;
    }

    private static int encodeTarget(IPlayerContext ctx, IMovement movement, int elapsed, float[] features, int index) {
        if (movement == null) {
            index += TARGET_FEATURES;
            return index;
        }
        BetterBlockPos destination = movement.getDest();
        Vec3 position = ctx.player().position();
        double dx = destination.x + 0.5 - position.x;
        double dy = destination.y - position.y;
        double dz = destination.z + 0.5 - position.z;
        double[] local = rotateToLocal(dx, dz, ctx.player().getYRot());
        double horizontal = Math.sqrt(dx * dx + dz * dz);

        features[index++] = (float) local[0];
        features[index++] = (float) local[1];
        features[index++] = (float) dy;
        features[index++] = (float) Math.min(horizontal, 8);
        features[index++] = (float) Math.atan2(local[1], local[0]) / (float) Math.PI;
        double cost = movement.getCost();
        features[index++] = (float) Math.min(cost / 20.0, 5.0);
        features[index++] = Math.min(elapsed / 20f, 5f);
        // how far over its own estimate this movement has run: the single most informative signal that something
        // has gone wrong, and the one the current code only uses as a hard timeout
        features[index++] = cost > 0 ? (float) Math.min(elapsed / cost, 5.0) : 0f;
        BetterBlockPos source = movement.getSrc();
        features[index++] = destination.y - source.y;
        features[index++] = Math.abs(destination.x - source.x) + Math.abs(destination.z - source.z);
        features[index++] = movement.calculateCurrentCost() >= baritone.api.pathing.movement.ActionCosts.COST_INF ? 1f : 0f;
        features[index++] = 1f; // presence flag: distinguishes "no movement" from "a movement that encodes to zeros"
        return index;
    }

    /**
     * Samples the terrain around the player into the feature vector, rotated so that the grid's forward axis is the
     * player's facing.
     */
    private static void encodeTerrain(BlockStateInterface bsi, BetterBlockPos feet, float yaw,
                                      float[] features, int index) {
        if (bsi == null) {
            return;
        }
        // snap the sampling grid to the nearest quarter turn; the terrain lattice is axis aligned, so a continuous
        // rotation would alias badly, while a quarter turn is exact
        int quarter = Math.floorMod(Math.round(yaw / 90f), 4);
        for (int forward = -GRID_RADIUS; forward <= GRID_RADIUS; forward++) {
            for (int right = -GRID_RADIUS; right <= GRID_RADIUS; right++) {
                int[] offset = rotateGrid(forward, right, quarter);
                for (int level = -1; level < COLUMN_HEIGHT - 1; level++) {
                    int x = feet.x + offset[0];
                    int y = feet.y + level;
                    int z = feet.z + offset[1];
                    index = encodeBlock(bsi, x, y, z, features, index);
                }
            }
        }
    }

    private static int encodeBlock(BlockStateInterface bsi, int x, int y, int z, float[] features, int index) {
        if (!bsi.worldContainsLoadedChunk(x, z)) {
            // unloaded is its own state, not "air": pretending unknown terrain is walkable is how bots walk into
            // holes at chunk borders
            features[index++] = -1f;
            features[index++] = -1f;
            features[index++] = -1f;
            features[index++] = -1f;
            return index;
        }
        BlockState state = bsi.get0(x, y, z);
        features[index++] = MovementHelper.canWalkThrough(bsi, x, y, z, state) ? 1f : 0f;
        features[index++] = MovementHelper.canWalkOn(bsi, x, y, z, state) ? 1f : 0f;
        features[index++] = state.getFluidState().isEmpty() ? 0f : 1f;
        features[index++] = state.getBlock().getFriction() - 0.6f; // centred on ordinary ground friction
        return index;
    }

    /**
     * Rotates a world-space horizontal vector into the player's frame: {@code [forward, left]}.
     */
    public static double[] rotateToLocal(double x, double z, float yaw) {
        double radians = Math.toRadians(yaw);
        double sin = Math.sin(radians);
        double cos = Math.cos(radians);
        // Minecraft's yaw zero faces +Z, and increases clockwise, hence the signs
        double forward = -x * sin + z * cos;
        double left = -x * cos - z * sin;
        return new double[]{forward, left};
    }

    private static int[] rotateGrid(int forward, int right, int quarter) {
        switch (quarter) {
            case 0:
                return new int[]{-right, forward};
            case 1:
                return new int[]{-forward, -right};
            case 2:
                return new int[]{right, -forward};
            default:
                return new int[]{forward, right};
        }
    }

    private static double fraction(double value) {
        return value - Math.floor(value);
    }
}
