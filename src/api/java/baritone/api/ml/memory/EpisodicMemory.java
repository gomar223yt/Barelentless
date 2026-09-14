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

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * A searchable memory of situations the bot has actually been in.
 * <p>
 * This is the non-parametric half of the learning stack, and it is deliberately not a neural network. A network
 * generalizes, which is what makes it useful and also what makes it wrong in exactly the places that matter: the one
 * specific ledge in your base that needs a late jump looks, to a network, like a thousand ordinary ledges. The
 * episodic memory keeps that ledge as its own entry, with its own measured outcome and its own confidence, and the
 * controller consults both - the network for situations it has never seen, the memory for situations it has.
 * <p>
 * Lookup is approximate nearest neighbour: several random projection hash tables narrow millions of candidates down
 * to a handful of buckets, and those candidates are then rescored exactly. Insert merges into an existing record when
 * the situation is close enough, so repeated exposure sharpens an estimate rather than filling the store with
 * duplicates. When capacity is reached, the least useful records - old, rarely visited, no longer surprising - are
 * evicted first.
 *
 * @author Barelentless
 */
public final class EpisodicMemory {

    private static final int MAGIC = 0x42524C4D; // "BRLM"
    private static final int VERSION = 1;

    /**
     * A retrieved neighbour and how similar it was to the query.
     */
    public static final class Neighbour {

        public final MemoryRecord record;
        public final float similarity;

        Neighbour(MemoryRecord record, float similarity) {
            this.record = record;
            this.similarity = similarity;
        }
    }

    /**
     * The memory's answer to "what happens if I do this here": a confidence-weighted estimate assembled from the
     * neighbours that were actually found, plus how much of it is grounded in real visits.
     */
    public static final class Estimate {

        public static final Estimate EMPTY = new Estimate(0, 0, 0, 0, 0);

        public final double outcome;
        public final double deviation;
        public final double successRate;
        public final double confidence;
        public final int neighbours;

        Estimate(double outcome, double deviation, double successRate, double confidence, int neighbours) {
            this.outcome = outcome;
            this.deviation = deviation;
            this.successRate = successRate;
            this.confidence = confidence;
            this.neighbours = neighbours;
        }

        public boolean isEmpty() {
            return this.neighbours == 0;
        }

        @Override
        public String toString() {
            return String.format("outcome=%.2f+-%.2f success=%.0f%% confidence=%.2f from %d neighbours",
                    this.outcome, this.deviation, this.successRate * 100, this.confidence, this.neighbours);
        }
    }

    private final int dimension;
    private final int capacity;

    /**
     * Similarity above which two situations are considered the same and their records merge.
     */
    private float mergeThreshold = 0.995f;

    /**
     * Similarity below which a neighbour is ignored entirely, however close it ranked.
     */
    private float relevanceThreshold = 0.85f;

    private final List<MemoryRecord> records = new ArrayList<>();

    /**
     * Every record's key, laid out end to end.
     * <p>
     * The scoring loop touches one key after another, and a {@code float[][]} would send it chasing a pointer into a
     * different part of the heap for each one. Contiguous storage is the difference between a query that scales with
     * the candidate count and one that scales with cache misses.
     */
    private float[] keyData;

    /**
     * Per table, a map from bucket hash to the index of the first record in that bucket.
     */
    private final LongIntMap[] bucketHead;

    /**
     * Per table, the next record in the same bucket, or -1. A chain rather than a per-bucket list, so a bucket costs
     * no object at all.
     */
    private final int[][] chainNext;

    private final float[][] projections;
    private final int tables;
    private final int bits;
    private int[] visitStamp = new int[1024];
    private int queryStamp;
    private int[] scratchIndices = new int[64];
    private float[] scratchSimilarities = new float[64];
    private long queries;
    private long merges;
    private long evictions;

    /**
     * Builds a memory with hashing parameters derived from its capacity.
     * <p>
     * The bit count is what decides how many records share a bucket, and the right value depends entirely on how many
     * records there will be: too few bits and every query scans thousands of candidates, too many and genuinely
     * similar situations stop colliding. Aiming for roughly one record per bucket, with twelve independent tables to
     * recover the recall that selectivity costs, measured out at about fifteen microseconds per query at two hundred
     * thousand records, against a hundred and sixty for a fixed twelve bits - with the same recall on the
     * near-identical situations that actually matter.
     *
     * @param dimension Length of a situation key
     * @param capacity  Maximum number of distinct situations to remember
     * @param random    Source of the random projections
     */
    public EpisodicMemory(int dimension, int capacity, Random random) {
        this(dimension, capacity, 12, bitsFor(capacity), random);
    }

    private static int bitsFor(int capacity) {
        int bits = 64 - Long.numberOfLeadingZeros(Math.max(1, capacity - 1));
        return Math.max(10, Math.min(20, bits));
    }

    /**
     * @param dimension Length of a situation key
     * @param capacity  Maximum number of distinct situations to remember
     * @param tables    Number of independent hash tables; more tables means better recall and more memory
     * @param bits      Hash bits per table; more bits means smaller, more selective buckets
     */
    public EpisodicMemory(int dimension, int capacity, int tables, int bits, Random random) {
        this.dimension = dimension;
        this.capacity = capacity;
        this.tables = tables;
        this.bits = bits;
        this.keyData = new float[Math.min(capacity, 4096) * dimension];
        this.bucketHead = new LongIntMap[tables];
        this.chainNext = new int[tables][];
        for (int table = 0; table < tables; table++) {
            this.bucketHead[table] = new LongIntMap(1024);
            this.chainNext[table] = new int[Math.min(capacity, 4096)];
        }
        this.projections = new float[tables * bits][dimension];
        for (float[] plane : this.projections) {
            for (int i = 0; i < dimension; i++) {
                plane[i] = (float) random.nextGaussian();
            }
            Memories.normalize(plane);
        }
    }

    /**
     * Records an experience. If a sufficiently similar situation is already known, its statistics are updated;
     * otherwise a new record is created.
     *
     * @return The record that now holds this experience
     */
    public synchronized MemoryRecord remember(float[] rawKey, int context, float[] action,
                                              double outcome, boolean successful, long nowMillis) {
        float[] key = Memories.normalized(rawKey);
        Neighbour closest = nearest(key, context);
        if (closest != null && closest.similarity >= this.mergeThreshold) {
            closest.record.observe(outcome, successful, nowMillis);
            this.merges++;
            return closest.record;
        }
        if (this.records.size() >= this.capacity) {
            evictOne(nowMillis);
        }
        MemoryRecord record = new MemoryRecord(key, context, action, nowMillis);
        record.observe(outcome, successful, nowMillis);
        int index = this.records.size();
        this.records.add(record);
        ensureCapacity(index + 1);
        System.arraycopy(key, 0, this.keyData, index * this.dimension, this.dimension);
        index(key, index);
        return record;
    }

    /**
     * Finds up to {@code k} remembered situations similar to the query, most similar first. A negative
     * {@code context} matches any context.
     */
    public synchronized List<Neighbour> recall(float[] rawKey, int context, int k) {
        this.queries++;
        float[] key = Memories.normalized(rawKey);
        int count = scan(key, context, k, true);
        List<Neighbour> found = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            found.add(new Neighbour(this.records.get(this.scratchIndices[i]), this.scratchSimilarities[i]));
        }
        return found;
    }

    /**
     * Summarises what the memory expects to happen in this situation, weighting each neighbour by how similar it is
     * and how much the record itself is worth trusting.
     */
    public synchronized Estimate estimate(float[] rawKey, int context, int k) {
        List<Neighbour> neighbours = recall(rawKey, context, k);
        if (neighbours.isEmpty()) {
            return Estimate.EMPTY;
        }
        double weightSum = 0;
        double outcomeSum = 0;
        double deviationSum = 0;
        double successSum = 0;
        for (Neighbour neighbour : neighbours) {
            // similarity is in [relevanceThreshold, 1]; rescale so a barely relevant neighbour counts for almost
            // nothing instead of a flat fraction
            double relevance = (neighbour.similarity - this.relevanceThreshold) / (1 - this.relevanceThreshold + 1e-6);
            double weight = relevance * relevance * neighbour.record.getConfidence();
            if (weight <= 0) {
                continue;
            }
            weightSum += weight;
            outcomeSum += weight * neighbour.record.getOutcomeMean();
            deviationSum += weight * neighbour.record.getOutcomeDeviation();
            successSum += weight * neighbour.record.getSuccessRate();
        }
        if (weightSum <= 0) {
            return Estimate.EMPTY;
        }
        // total weight also bounds the confidence: one half-relevant memory is not evidence
        double confidence = weightSum / (weightSum + 1.0);
        return new Estimate(outcomeSum / weightSum, deviationSum / weightSum, successSum / weightSum,
                confidence, neighbours.size());
    }

    private Neighbour nearest(float[] key, int context) {
        int count = scan(key, context, 1, false);
        if (count == 0) {
            return null;
        }
        return new Neighbour(this.records.get(this.scratchIndices[0]), this.scratchSimilarities[0]);
    }

    /**
     * The one hot loop in this class: walk every bucket chain this key falls into, score each record once, and keep
     * the best {@code k}.
     * <p>
     * Everything about it is shaped by the fact that the pathfinder calls it thousands of times per calculation.
     * Nothing is allocated - candidates are de-duplicated with a query stamp, results are kept in reusable scratch
     * arrays, and the top {@code k} are maintained by insertion, which beats sorting for the small k this is ever
     * called with. Results land in {@link #scratchIndices} and {@link #scratchSimilarities}, most similar first.
     *
     * @param applyThreshold Whether to discard neighbours below the relevance threshold
     * @return How many results were found
     */
    private int scan(float[] key, int context, int k, boolean applyThreshold) {
        final int size = this.records.size();
        if (size == 0) {
            return 0;
        }
        if (this.visitStamp.length < size) {
            this.visitStamp = new int[Math.max(size, this.visitStamp.length * 2)];
            this.queryStamp = 0;
        }
        if (this.scratchIndices.length < k) {
            this.scratchIndices = new int[k];
            this.scratchSimilarities = new float[k];
        }
        final int stamp = ++this.queryStamp;
        final int dimension = this.dimension;
        int found = 0;
        float worstKept = Float.NEGATIVE_INFINITY;

        for (int table = 0; table < this.tables; table++) {
            int index = this.bucketHead[table].get(hash(key, table));
            int[] chain = this.chainNext[table];
            while (index != LongIntMap.ABSENT) {
                if (index >= size) {
                    // a record that has since been evicted; the chain is rebuilt lazily, so just skip it
                    index = chain[index];
                    continue;
                }
                if (this.visitStamp[index] == stamp) {
                    index = chain[index];
                    continue;
                }
                this.visitStamp[index] = stamp;
                MemoryRecord record = this.records.get(index);
                if (context >= 0 && record.context != context) {
                    index = chain[index];
                    continue;
                }
                float similarity = 0;
                int base = index * dimension;
                for (int c = 0; c < dimension; c++) {
                    similarity += this.keyData[base + c] * key[c];
                }
                if ((!applyThreshold || similarity >= this.relevanceThreshold)
                        && (found < k || similarity > worstKept)) {
                    int position = found < k ? found++ : k - 1;
                    while (position > 0 && this.scratchSimilarities[position - 1] < similarity) {
                        this.scratchSimilarities[position] = this.scratchSimilarities[position - 1];
                        this.scratchIndices[position] = this.scratchIndices[position - 1];
                        position--;
                    }
                    this.scratchSimilarities[position] = similarity;
                    this.scratchIndices[position] = index;
                    worstKept = this.scratchSimilarities[found - 1];
                }
                index = chain[index];
            }
        }
        return found;
    }

    private void ensureCapacity(int required) {
        if (required * this.dimension > this.keyData.length) {
            int records = Math.max(required, this.keyData.length / this.dimension * 2);
            this.keyData = java.util.Arrays.copyOf(this.keyData, records * this.dimension);
            for (int table = 0; table < this.tables; table++) {
                this.chainNext[table] = java.util.Arrays.copyOf(this.chainNext[table], records);
            }
        }
    }

    private void index(float[] key, int recordIndex) {
        ensureCapacity(recordIndex + 1);
        for (int table = 0; table < this.tables; table++) {
            long hash = hash(key, table);
            this.chainNext[table][recordIndex] = this.bucketHead[table].get(hash);
            this.bucketHead[table].put(hash, recordIndex);
        }
    }

    private long hash(float[] key, int table) {
        long value = table;
        int base = table * this.bits;
        for (int bit = 0; bit < this.bits; bit++) {
            value = (value << 1) | (Memories.dot(key, this.projections[base + bit]) >= 0 ? 1 : 0);
        }
        return value;
    }

    private void evictOne(long nowMillis) {
        int worst = -1;
        double worstUtility = Double.MAX_VALUE;
        for (int i = 0; i < this.records.size(); i++) {
            double utility = this.records.get(i).utility(nowMillis);
            if (utility < worstUtility) {
                worstUtility = utility;
                worst = i;
            }
        }
        if (worst < 0) {
            return;
        }
        // swap-remove keeps indices dense; the moved record has to be re-indexed under the removed one's slot
        int last = this.records.size() - 1;
        MemoryRecord moved = this.records.get(last);
        this.records.set(worst, moved);
        this.records.remove(last);
        this.evictions++;
        reindex();
    }

    private void reindex() {
        for (int table = 0; table < this.tables; table++) {
            this.bucketHead[table].clear();
        }
        ensureCapacity(Math.max(1, this.records.size()));
        for (int i = 0; i < this.records.size(); i++) {
            float[] key = this.records.get(i).key;
            System.arraycopy(key, 0, this.keyData, i * this.dimension, this.dimension);
            index(key, i);
        }
    }

    /**
     * Merges records that have drifted close enough together to be the same situation, and drops records that never
     * accumulated enough visits to be worth the space. Run occasionally, not per tick.
     *
     * @return The number of records removed
     */
    public synchronized int consolidate(long nowMillis, long minimumVisits) {
        int before = this.records.size();
        List<MemoryRecord> kept = new ArrayList<>(before);
        for (MemoryRecord record : this.records) {
            if (record.getVisits() < minimumVisits && nowMillis - record.getLastSeenMillis() > 3_600_000L) {
                continue;
            }
            MemoryRecord target = null;
            for (MemoryRecord candidate : kept) {
                if (candidate.context == record.context
                        && Memories.dot(candidate.key, record.key) >= this.mergeThreshold) {
                    target = candidate;
                    break;
                }
            }
            if (target == null) {
                kept.add(record);
            } else {
                target.absorb(record);
                this.merges++;
            }
        }
        this.records.clear();
        this.records.addAll(kept);
        reindex();
        return before - this.records.size();
    }

    public synchronized int size() {
        return this.records.size();
    }

    public int capacity() {
        return this.capacity;
    }

    public synchronized long getQueries() {
        return this.queries;
    }

    public synchronized long getMerges() {
        return this.merges;
    }

    public synchronized long getEvictions() {
        return this.evictions;
    }

    public synchronized long totalVisits() {
        long total = 0;
        for (MemoryRecord record : this.records) {
            total += record.getVisits();
        }
        return total;
    }

    public synchronized List<MemoryRecord> records() {
        return new ArrayList<>(this.records);
    }

    public void setMergeThreshold(float mergeThreshold) {
        this.mergeThreshold = mergeThreshold;
    }

    public void setRelevanceThreshold(float relevanceThreshold) {
        this.relevanceThreshold = relevanceThreshold;
    }

    public synchronized void clear() {
        this.records.clear();
        for (int table = 0; table < this.tables; table++) {
            this.bucketHead[table].clear();
        }
    }

    public synchronized void save(Path path) throws IOException {
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path temporary = path.resolveSibling(path.getFileName() + ".tmp");
        try (DataOutputStream out = new DataOutputStream(
                new GZIPOutputStream(new BufferedOutputStream(Files.newOutputStream(temporary))))) {
            out.writeInt(MAGIC);
            out.writeInt(VERSION);
            out.writeInt(this.dimension);
            out.writeInt(this.records.size());
            for (MemoryRecord record : this.records) {
                record.write(out);
            }
        }
        Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING);
    }

    /**
     * Loads records from disk, replacing whatever is currently held. The hash planes are not stored: they are derived
     * from the random seed and everything is re-indexed on load, so a memory file stays valid across restarts.
     */
    public synchronized int load(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(
                new GZIPInputStream(new BufferedInputStream(Files.newInputStream(path))))) {
            if (in.readInt() != MAGIC) {
                throw new IOException("not a Barelentless memory file");
            }
            int version = in.readInt();
            if (version > VERSION) {
                throw new IOException("memory written by a newer version (" + version + ")");
            }
            int storedDimension = in.readInt();
            if (storedDimension != this.dimension) {
                throw new IOException("memory dimension " + storedDimension + " != " + this.dimension);
            }
            int count = in.readInt();
            this.records.clear();
            for (int i = 0; i < count && i < this.capacity; i++) {
                this.records.add(MemoryRecord.read(in));
            }
            reindex();
            return this.records.size();
        }
    }

    public int getDimension() {
        return this.dimension;
    }
}
