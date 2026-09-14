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

package baritone.api.pathing.calc;

import baritone.api.ml.ICostAdjuster;

import java.util.List;

/**
 * Where route preference is decided: what the pathfinder is told a candidate movement costs.
 * <p>
 * This is the hook for changing <em>which way the bot goes</em>, as opposed to how it walks once it has decided.
 * Avoid a player's farmland, prefer lit corridors at night, treat a neighbour's roof as expensive, route around the
 * area your own mod is busy building in - all of that is a few lines here, and none of it is expressible any other
 * way short of editing the movement classes.
 * <p>
 * Every registered adjuster is consulted in registration order, each seeing the cost as the previous one left it.
 * The final value is checked before use: a cost that is not finite, not positive, or has become effectively
 * infinite is discarded and the original kept, so a broken adjuster degrades into having no opinion rather than into
 * paths through lava.
 * <p>
 * The one thing to keep in mind is that this is the hottest code in the mod. A* asks about hundreds of thousands of
 * candidates per calculation; an adjuster that takes a microsecond adds a fifth of a second to every path. Cache
 * per situation rather than computing per candidate - the same question repeats thousands of times within one
 * calculation, which is exactly why a factory hands you a fresh instance to cache in.
 *
 * @author Barelentless
 */
public interface ICostRegistry {

    /**
     * A registration handle. Close it to stop affecting path costs.
     */
    interface Registration extends AutoCloseable {

        String name();

        boolean isActive();

        @Override
        void close();
    }

    /**
     * Adds a factory. It will be asked for an adjuster at the start of every path calculation from now on.
     *
     * @param name    Shown by {@code control costs}
     * @param factory Creates one adjuster per calculation
     * @return A handle that removes it again
     */
    Registration register(String name, ICostAdjuster.Factory factory);

    List<Registration> registrations();

    /**
     * How many candidate movements the registered adjusters changed, and how many they were asked about, over the
     * most recent path calculation. Index zero is adjustments, index one is queries.
     */
    long[] lastCalculationStatistics();
}
