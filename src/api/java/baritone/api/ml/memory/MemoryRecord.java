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

package baritone.api.ml.memory;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * One remembered situation: what the world looked like, what was done, and how it actually turned out.
 * <p>
 * A record is not a single observation but everything ever observed about one situation. When the same situation is
 * met again the existing record is updated - visits, mean outcome, variance, last seen - instead of a new one being
 * appended. That is what makes the memory a model of the world rather than a log of it: after a hundred jumps off the
 * same kind of ledge, the record knows both the average cost and how reliable that average is, and the controller can
 * tell the difference between "this works" and "this worked once".
 *
 * @author Barelentless
 */
public final class MemoryRecord {

    /**
     * The situation embedding this record is keyed by. Unit length, so similarity is a plain dot product.
     */
    public final float[] key;

    /**
     * Discrete context id, e.g. the ordinal of the movement kind being executed. Lookups can require an exact match
     * on this so that a "traverse" situation is never answered with a "parkour" memory.
     */
    public final int context;

    /**
     * The action that was taken, in whatever encoding the owner of this memory uses.
     */
    public final float[] action;

    private long visits;
    private double outcomeMean;
    private double outcomeM2;
    private double successes;
    private long lastSeenMillis;
    private long firstSeenMillis;
    private float surprise;

    public MemoryRecord(float[] key, int context, float[] action, long nowMillis) {
        this.key = key;
        this.context = context;
        this.action = action;
        this.firstSeenMillis = nowMillis;
        this.lastSeenMillis = nowMillis;
    }

    /**
     * Folds a fresh observation of this situation into the record.
     *
     * @param outcome    The measured outcome, e.g. ticks actually taken
     * @param successful Whether the attempt achieved what it set out to do
     * @param nowMillis  Wall clock, for recency based eviction
     */
    public synchronized void observe(double outcome, boolean successful, long nowMillis) {
        this.visits++;
        double delta = outcome - this.outcomeMean;
        this.outcomeMean += delta / this.visits;
        this.outcomeM2 += delta * (outcome - this.outcomeMean);
        if (successful) {
            this.successes++;
        }
        this.lastSeenMillis = nowMillis;
        // surprise decays as the record's estimate settles; it drives both replay priority and eviction
        this.surprise = (float) Math.abs(delta) / (float) Math.max(1.0, Math.abs(this.outcomeMean));
    }

    /**
     * Blends another record's statistics into this one. Used when two records turn out to describe the same
     * situation closely enough to merge.
     */
    public synchronized void absorb(MemoryRecord other) {
        long combined = this.visits + other.visits;
        if (combined == 0) {
            return;
        }
        double delta = other.outcomeMean - this.outcomeMean;
        double mean = (this.outcomeMean * this.visits + other.outcomeMean * other.visits) / combined;
        this.outcomeM2 = this.outcomeM2 + other.outcomeM2
                + delta * delta * this.visits * other.visits / (double) combined;
        this.outcomeMean = mean;
        this.successes += other.successes;
        this.visits = combined;
        this.lastSeenMillis = Math.max(this.lastSeenMillis, other.lastSeenMillis);
        this.firstSeenMillis = Math.min(this.firstSeenMillis, other.firstSeenMillis);
        for (int i = 0; i < this.key.length; i++) {
            this.key[i] = (this.key[i] + other.key[i]) * 0.5f;
        }
        Memories.normalize(this.key);
    }

    public synchronized long getVisits() {
        return this.visits;
    }

    public synchronized double getOutcomeMean() {
        return this.outcomeMean;
    }

    /**
     * Sample standard deviation of the outcome. High deviation means the situation is genuinely unreliable, which is
     * information in itself - it is the difference between a jump that costs 12 ticks and a jump that costs 12 ticks
     * on average but occasionally kills you.
     */
    public synchronized double getOutcomeDeviation() {
        return this.visits < 2 ? 0 : Math.sqrt(this.outcomeM2 / (this.visits - 1));
    }

    public synchronized double getSuccessRate() {
        return this.visits == 0 ? 0 : this.successes / this.visits;
    }

    /**
     * How much weight this record's estimate deserves. Confidence grows with visits and shrinks with variance, so a
     * single lucky success never outvotes a dozen consistent failures.
     */
    public synchronized double getConfidence() {
        if (this.visits == 0) {
            return 0;
        }
        double deviation = getOutcomeDeviation();
        double spread = 1.0 / (1.0 + deviation / Math.max(1.0, Math.abs(this.outcomeMean)));
        return spread * (this.visits / (this.visits + 4.0));
    }

    public synchronized long getLastSeenMillis() {
        return this.lastSeenMillis;
    }

    public synchronized long getFirstSeenMillis() {
        return this.firstSeenMillis;
    }

    public synchronized float getSurprise() {
        return this.surprise;
    }

    /**
     * Eviction score - lower is more disposable. Recent, frequently visited and still surprising records survive.
     */
    public synchronized double utility(long nowMillis) {
        double ageHours = Math.max(0, nowMillis - this.lastSeenMillis) / 3_600_000.0;
        double recency = 1.0 / (1.0 + ageHours);
        return Math.log1p(this.visits) * recency * (1.0 + this.surprise);
    }

    public synchronized void write(DataOutput out) throws IOException {
        out.writeInt(this.key.length);
        for (float value : this.key) {
            out.writeFloat(value);
        }
        out.writeInt(this.context);
        out.writeInt(this.action == null ? -1 : this.action.length);
        if (this.action != null) {
            for (float value : this.action) {
                out.writeFloat(value);
            }
        }
        out.writeLong(this.visits);
        out.writeDouble(this.outcomeMean);
        out.writeDouble(this.outcomeM2);
        out.writeDouble(this.successes);
        out.writeLong(this.firstSeenMillis);
        out.writeLong(this.lastSeenMillis);
        out.writeFloat(this.surprise);
    }

    public static MemoryRecord read(DataInput in) throws IOException {
        int keyLength = in.readInt();
        float[] key = new float[keyLength];
        for (int i = 0; i < keyLength; i++) {
            key[i] = in.readFloat();
        }
        int context = in.readInt();
        int actionLength = in.readInt();
        float[] action = null;
        if (actionLength >= 0) {
            action = new float[actionLength];
            for (int i = 0; i < actionLength; i++) {
                action[i] = in.readFloat();
            }
        }
        MemoryRecord record = new MemoryRecord(key, context, action, 0);
        record.visits = in.readLong();
        record.outcomeMean = in.readDouble();
        record.outcomeM2 = in.readDouble();
        record.successes = in.readDouble();
        record.firstSeenMillis = in.readLong();
        record.lastSeenMillis = in.readLong();
        record.surprise = in.readFloat();
        return record;
    }

    @Override
    public String toString() {
        return String.format("ctx=%d visits=%d mean=%.2f sd=%.2f success=%.0f%% conf=%.2f",
                this.context, this.visits, this.outcomeMean, getOutcomeDeviation(),
                getSuccessRate() * 100, getConfidence());
    }
}
