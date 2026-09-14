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

package baritone.pathing.calc;

import baritone.Baritone;
import baritone.api.ml.ICostAdjuster;
import baritone.api.pathing.calc.ICostRegistry;
import baritone.api.pathing.movement.ActionCosts;
import baritone.utils.BlockStateInterface;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicLongArray;

/**
 * Holds the registered cost adjusters and composes them into the single one a calculation uses.
 *
 * @author Barelentless
 */
public final class CostRegistry implements ICostRegistry {

    private static final class Entry implements Registration {

        private final String name;
        private final ICostAdjuster.Factory factory;
        private final List<Entry> owner;
        private volatile boolean active = true;

        Entry(String name, ICostAdjuster.Factory factory, List<Entry> owner) {
            this.name = name;
            this.factory = factory;
            this.owner = owner;
        }

        @Override
        public String name() {
            return this.name;
        }

        @Override
        public boolean isActive() {
            return this.active;
        }

        @Override
        public void close() {
            if (this.active) {
                this.active = false;
                this.owner.remove(this);
            }
        }
    }

    private final Baritone baritone;
    private final List<Entry> entries = new CopyOnWriteArrayList<>();
    private final AtomicLongArray statistics = new AtomicLongArray(2);

    public CostRegistry(Baritone baritone) {
        this.baritone = baritone;
    }

    @Override
    public Registration register(String name, ICostAdjuster.Factory factory) {
        Entry entry = new Entry(name, factory, this.entries);
        this.entries.add(entry);
        return entry;
    }

    @Override
    public List<Registration> registrations() {
        return new ArrayList<>(this.entries);
    }

    @Override
    public long[] lastCalculationStatistics() {
        return new long[]{this.statistics.get(0), this.statistics.get(1)};
    }

    /**
     * Builds the adjuster for one calculation, or null when nobody has anything to say about costs.
     */
    public ICostAdjuster create(BlockStateInterface blocks) {
        if (this.entries.isEmpty()) {
            return null;
        }
        ICostAdjuster.Blocks snapshot = new ICostAdjuster.Blocks() {

            @Override
            public net.minecraft.world.level.block.state.BlockState get(int x, int y, int z) {
                return blocks.get0(x, y, z);
            }

            @Override
            public boolean isLoaded(int x, int z) {
                return blocks.worldContainsLoadedChunk(x, z);
            }
        };
        List<ICostAdjuster> adjusters = new ArrayList<>(this.entries.size());
        for (Entry entry : this.entries) {
            try {
                ICostAdjuster adjuster = entry.factory.create(this.baritone, snapshot);
                if (adjuster != null) {
                    adjusters.add(adjuster);
                }
            } catch (RuntimeException e) {
                // a factory that throws is a factory that is not going to work; drop it rather than failing the
                // calculation, which would leave the bot unable to path at all
                entry.close();
            }
        }
        if (adjusters.isEmpty()) {
            return null;
        }
        return new Composite(adjusters.toArray(new ICostAdjuster[0]), this.statistics);
    }

    /**
     * Runs each adjuster in turn and validates the result.
     * <p>
     * The validation is the important part. A* trusts these numbers completely: a zero or negative cost makes a
     * route look free, a NaN poisons every comparison it touches, and an infinite one silently deletes a movement
     * from the search. So anything that is not a sane positive finite cost is discarded in favour of what the
     * ordinary cost functions said, and the adjuster that produced it is simply ignored for that candidate.
     */
    private static final class Composite implements ICostAdjuster {

        private final ICostAdjuster[] adjusters;
        private final AtomicLongArray statistics;
        private long adjusted;
        private long queries;

        Composite(ICostAdjuster[] adjusters, AtomicLongArray statistics) {
            this.adjusters = adjusters;
            this.statistics = statistics;
            statistics.set(0, 0);
            statistics.set(1, 0);
        }

        @Override
        public double adjust(int moveOrdinal, int srcX, int srcY, int srcZ,
                             int destX, int destY, int destZ, double cost) {
            this.queries++;
            double current = cost;
            for (ICostAdjuster adjuster : this.adjusters) {
                double next;
                try {
                    next = adjuster.adjust(moveOrdinal, srcX, srcY, srcZ, destX, destY, destZ, current);
                } catch (RuntimeException e) {
                    continue;
                }
                if (next > 0 && next < ActionCosts.COST_INF && !Double.isNaN(next)) {
                    current = next;
                }
            }
            if (current != cost) {
                this.adjusted++;
            }
            // publishing every candidate would be a contended atomic write in the hottest loop in the mod; once
            // every few thousand is plenty for a diagnostic readout
            if ((this.queries & 0xFFF) == 0) {
                this.statistics.set(0, this.adjusted);
                this.statistics.set(1, this.queries);
            }
            return current;
        }

        @Override
        public long[] statistics() {
            return new long[]{this.adjusted, this.queries};
        }
    }
}
