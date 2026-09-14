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

package baritone.api.ml;

/**
 * Adjusts the cost the pathfinder assigns to a candidate movement, using whatever the bot has learned about how that
 * kind of movement really performs in that kind of place.
 * <p>
 * One instance is created per path calculation and used by that calculation's thread only, so implementations may
 * keep unsynchronized per-calculation state - a cache is not just allowed here, it is expected: A* evaluates hundreds
 * of thousands of candidates and anything called that often has to be a handful of nanoseconds in the common case.
 *
 * @author Barelentless
 */
public interface ICostAdjuster {

    /**
     * Read access to the world for the calculation an adjuster belongs to.
     * <p>
     * Path calculation happens on its own thread against a snapshot of the world, so an adjuster must not reach for
     * the live world itself. This is that snapshot, and it is safe to call from the calculating thread only.
     */
    interface Blocks {

        /**
         * @return The state at that position. Only meaningful when {@link #isLoaded} says so.
         */
        net.minecraft.world.level.block.state.BlockState get(int x, int y, int z);

        boolean isLoaded(int x, int z);
    }

    /**
     * Creates one adjuster per path calculation.
     * <p>
     * Per calculation rather than one shared instance, because the interesting implementations cache: A* asks about
     * hundreds of thousands of candidates and the same situation comes up thousands of times within one calculation.
     * A fresh instance per calculation is thread-confined, so that cache needs no synchronization and no
     * invalidation.
     */
    @FunctionalInterface
    interface Factory {

        /**
         * @param baritone The instance calculating
         * @param blocks   The world snapshot for this calculation
         * @return An adjuster, or {@code null} to sit this calculation out
         */
        ICostAdjuster create(baritone.api.IBaritone baritone, Blocks blocks);
    }

    /**
     * @param moveOrdinal The ordinal of the movement kind being considered
     * @param srcX        Source block x
     * @param srcY        Source block y
     * @param srcZ        Source block z
     * @param destX       Destination block x
     * @param destY       Destination block y
     * @param destZ       Destination block z
     * @param cost        The cost computed by the ordinary cost functions, in ticks
     * @return The adjusted cost, in ticks. Must be finite and strictly positive.
     */
    double adjust(int moveOrdinal, int srcX, int srcY, int srcZ, int destX, int destY, int destZ, double cost);

    /**
     * How many candidates this adjuster actually changed, and how many it was asked about, for diagnostics.
     * Index zero is adjustments made, index one is total queries.
     */
    default long[] statistics() {
        return new long[]{0, 0};
    }
}
