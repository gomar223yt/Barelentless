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
