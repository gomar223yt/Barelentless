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

package baritone.api.ml.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Predicate;

/**
 * A thread-safe, fixed capacity ring of past experience with prioritized sampling.
 * <p>
 * Both halves matter. The ring is what lets the game thread hand off samples without ever blocking on training: it
 * writes, the trainer thread reads, and when the ring is full the oldest sample is dropped rather than the newest
 * refused. The priorities are what stop the model wasting its capacity on the thousands of near-identical
 * "walking in a straight line on grass" ticks: a sample whose prediction was badly wrong is replayed far more often
 * than one the model already handles, and importance weights correct for the bias that introduces.
 *
 * @param <T> The experience record type
 * @author Barelentless
 */
public final class ReplayBuffer<T> {

    /**
     * One sampled item together with the bookkeeping needed to update its priority afterwards.
     */
    public static final class Sample<T> {

        public final T value;
        public final int index;
        public final float weight;

        Sample(T value, int index, float weight) {
            this.value = value;
            this.index = index;
            this.weight = weight;
        }
    }

    private final Object[] items;
    private final float[] priorities;
    private final int capacity;
    private final Random random;

    /**
     * How strongly priorities bias sampling. Zero is uniform sampling; one is fully proportional.
     */
    private float alpha = 0.6f;

    /**
     * How strongly importance weights correct for that bias. Annealed towards one over training.
     */
    private float beta = 0.4f;

    private int head;
    private int size;
    private long added;
    private float maxPriority = 1f;

    public ReplayBuffer(int capacity, Random random) {
        this.capacity = capacity;
        this.items = new Object[capacity];
        this.priorities = new float[capacity];
        this.random = random;
    }

    public synchronized void add(T item) {
        add(item, this.maxPriority);
    }

    /**
     * Adds with an explicit priority. New samples default to the highest priority seen so far, which guarantees every
     * sample is trained on at least once before its priority is refined.
     */
    public synchronized void add(T item, float priority) {
        this.items[this.head] = item;
        this.priorities[this.head] = Math.max(1e-4f, priority);
        this.head = (this.head + 1) % this.capacity;
        this.size = Math.min(this.size + 1, this.capacity);
        this.added++;
        this.maxPriority = Math.max(this.maxPriority, priority);
    }

    /**
     * Draws a batch. Returns fewer than requested only when the buffer holds fewer items.
     */
    @SuppressWarnings("unchecked")
    public synchronized List<Sample<T>> sample(int count) {
        List<Sample<T>> batch = new ArrayList<>(Math.min(count, this.size));
        if (this.size == 0) {
            return batch;
        }
        double total = 0;
        double[] cumulative = new double[this.size];
        for (int i = 0; i < this.size; i++) {
            total += Math.pow(this.priorities[i], this.alpha);
            cumulative[i] = total;
        }
        double minProbability = Double.MAX_VALUE;
        for (int i = 0; i < this.size; i++) {
            minProbability = Math.min(minProbability, Math.pow(this.priorities[i], this.alpha) / total);
        }
        float maxWeight = (float) Math.pow(minProbability * this.size, -this.beta);
        for (int n = 0; n < count; n++) {
            double target = this.random.nextDouble() * total;
            int index = lowerBound(cumulative, target);
            double probability = Math.pow(this.priorities[index], this.alpha) / total;
            float weight = (float) Math.pow(probability * this.size, -this.beta) / maxWeight;
            batch.add(new Sample<>((T) this.items[index], index, weight));
        }
        return batch;
    }

    private static int lowerBound(double[] cumulative, double target) {
        int low = 0;
        int high = cumulative.length - 1;
        while (low < high) {
            int middle = (low + high) >>> 1;
            if (cumulative[middle] < target) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        return low;
    }

    /**
     * Refreshes the priority of a previously sampled item, normally to the magnitude of its prediction error.
     */
    public synchronized void updatePriority(int index, float priority) {
        if (index >= 0 && index < this.capacity) {
            this.priorities[index] = Math.max(1e-4f, priority);
            this.maxPriority = Math.max(this.maxPriority, this.priorities[index]);
        }
    }

    /**
     * Draws a batch ignoring priorities. Evaluation runs use this so that reported loss reflects the data as it
     * actually is, not as the sampler prefers it.
     */
    @SuppressWarnings("unchecked")
    public synchronized List<T> sampleUniform(int count) {
        List<T> batch = new ArrayList<>(Math.min(count, this.size));
        for (int i = 0; i < count && this.size > 0; i++) {
            batch.add((T) this.items[this.random.nextInt(this.size)]);
        }
        return batch;
    }

    @SuppressWarnings("unchecked")
    public synchronized List<T> toList() {
        List<T> all = new ArrayList<>(this.size);
        for (int i = 0; i < this.size; i++) {
            all.add((T) this.items[i]);
        }
        return all;
    }

    @SuppressWarnings("unchecked")
    public synchronized int removeIf(Predicate<T> predicate) {
        int removed = 0;
        for (int i = 0; i < this.size; i++) {
            if (this.items[i] != null && predicate.test((T) this.items[i])) {
                this.priorities[i] = 1e-4f;
                removed++;
            }
        }
        return removed;
    }

    public synchronized int size() {
        return this.size;
    }

    public int capacity() {
        return this.capacity;
    }

    public synchronized long totalAdded() {
        return this.added;
    }

    public synchronized boolean isEmpty() {
        return this.size == 0;
    }

    public synchronized void clear() {
        java.util.Arrays.fill(this.items, null);
        java.util.Arrays.fill(this.priorities, 0f);
        this.head = 0;
        this.size = 0;
        this.maxPriority = 1f;
    }

    public synchronized void setAlpha(float alpha) {
        this.alpha = alpha;
    }

    public synchronized void setBeta(float beta) {
        this.beta = Math.min(1f, Math.max(0f, beta));
    }

    public synchronized float getBeta() {
        return this.beta;
    }

    /**
     * Moves beta towards one, the standard annealing for prioritized replay: early training tolerates the sampling
     * bias in exchange for speed, later training corrects for it.
     */
    public synchronized void annealBeta(float increment) {
        setBeta(this.beta + increment);
    }
}
