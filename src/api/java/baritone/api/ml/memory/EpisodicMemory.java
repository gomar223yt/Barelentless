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
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
    private final Map<Long, List<Integer>> buckets = new HashMap<>();
    private final float[][] projections;
    private final int tables;
    private final int bits;
    private long queries;
    private long merges;
    private long evictions;

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
        List<Integer> candidates = candidates(key);
        List<Neighbour> found = new ArrayList<>(Math.min(k, candidates.size()));
        for (int index : candidates) {
            MemoryRecord record = this.records.get(index);
            if (context >= 0 && record.context != context) {
                continue;
            }
            float similarity = Memories.dot(key, record.key);
            if (similarity >= this.relevanceThreshold) {
                found.add(new Neighbour(record, similarity));
            }
        }
        found.sort(Comparator.comparingDouble((Neighbour n) -> -n.similarity));
        return found.size() > k ? new ArrayList<>(found.subList(0, k)) : found;
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
        List<Neighbour> found = new ArrayList<>(1);
        float best = -1;
        MemoryRecord bestRecord = null;
        for (int index : candidates(key)) {
            MemoryRecord record = this.records.get(index);
            if (context >= 0 && record.context != context) {
                continue;
            }
            float similarity = Memories.dot(key, record.key);
            if (similarity > best) {
                best = similarity;
                bestRecord = record;
            }
        }
        if (bestRecord == null) {
            return null;
        }
        found.add(new Neighbour(bestRecord, best));
        return found.get(0);
    }

    private List<Integer> candidates(float[] key) {
        List<Integer> candidates = new ArrayList<>();
        boolean[] seen = new boolean[this.records.size()];
        for (int table = 0; table < this.tables; table++) {
            List<Integer> bucket = this.buckets.get(hash(key, table));
            if (bucket == null) {
                continue;
            }
            for (int index : bucket) {
                if (index < seen.length && !seen[index]) {
                    seen[index] = true;
                    candidates.add(index);
                }
            }
        }
        return candidates;
    }

    private void index(float[] key, int recordIndex) {
        for (int table = 0; table < this.tables; table++) {
            this.buckets.computeIfAbsent(hash(key, table), ignored -> new ArrayList<>()).add(recordIndex);
        }
    }

    private long hash(float[] key, int table) {
        long value = table;
        for (int bit = 0; bit < this.bits; bit++) {
            float[] plane = this.projections[table * this.bits + bit];
            value = (value << 1) | (Memories.dot(key, plane) >= 0 ? 1 : 0);
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
        this.buckets.clear();
        for (int i = 0; i < this.records.size(); i++) {
            index(this.records.get(i).key, i);
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
        this.buckets.clear();
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
