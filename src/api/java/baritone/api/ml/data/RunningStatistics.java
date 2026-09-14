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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

/**
 * Online per-feature mean and variance (Welford's algorithm), used to standardize observations.
 * <p>
 * Raw gameplay features live on wildly different scales - a velocity component is around 0.2, a block distance is
 * around 100, a tick counter climbs without bound. Feeding those to a network directly means the large-scale features
 * dominate every gradient. Standardization is computed online because there is no offline dataset to precompute it
 * from: statistics update as the player plays, and are saved alongside the weights (a model loaded without its
 * statistics would see a completely different input distribution).
 *
 * @author Barelentless
 */
public final class RunningStatistics {

    private final int features;
    private final double[] mean;
    private final double[] m2;
    private long count;
    private boolean frozen;

    public RunningStatistics(int features) {
        this.features = features;
        this.mean = new double[features];
        this.m2 = new double[features];
    }

    public synchronized void observe(float[] sample) {
        if (this.frozen) {
            return;
        }
        this.count++;
        for (int i = 0; i < this.features; i++) {
            double value = sample[i];
            if (Double.isNaN(value) || Double.isInfinite(value)) {
                continue;
            }
            double delta = value - this.mean[i];
            this.mean[i] += delta / this.count;
            this.m2[i] += delta * (value - this.mean[i]);
        }
    }

    public synchronized void observe(float[][] samples) {
        for (float[] sample : samples) {
            observe(sample);
        }
    }

    /**
     * Standardizes in place: {@code (x - mean) / std}, clipped to +-5 standard deviations so that one freak sample
     * can't dominate a batch.
     */
    public synchronized void normalize(float[] sample) {
        for (int i = 0; i < this.features; i++) {
            float standardDeviation = deviation(i);
            float value = (float) ((sample[i] - this.mean[i]) / standardDeviation);
            sample[i] = Math.max(-5f, Math.min(5f, value));
        }
    }

    public synchronized float[] normalized(float[] sample) {
        float[] copy = sample.clone();
        normalize(copy);
        return copy;
    }

    private float deviation(int index) {
        if (this.count < 2) {
            return 1f;
        }
        double variance = this.m2[index] / (this.count - 1);
        return (float) Math.max(1e-4, Math.sqrt(variance));
    }

    public synchronized float meanOf(int index) {
        return (float) this.mean[index];
    }

    public synchronized float deviationOf(int index) {
        return deviation(index);
    }

    public synchronized long getCount() {
        return this.count;
    }

    public int getFeatures() {
        return this.features;
    }

    /**
     * Stops the statistics from moving. Used when replaying a fixed dataset, so that repeated passes don't slowly
     * shift the normalization out from under the weights being fit to it.
     */
    public synchronized void freeze(boolean frozen) {
        this.frozen = frozen;
    }

    public synchronized void reset() {
        Arrays.fill(this.mean, 0);
        Arrays.fill(this.m2, 0);
        this.count = 0;
    }

    public synchronized void write(DataOutput out) throws IOException {
        out.writeInt(this.features);
        out.writeLong(this.count);
        for (int i = 0; i < this.features; i++) {
            out.writeDouble(this.mean[i]);
            out.writeDouble(this.m2[i]);
        }
    }

    public static RunningStatistics read(DataInput in) throws IOException {
        int features = in.readInt();
        RunningStatistics statistics = new RunningStatistics(features);
        statistics.count = in.readLong();
        for (int i = 0; i < features; i++) {
            statistics.mean[i] = in.readDouble();
            statistics.m2[i] = in.readDouble();
        }
        return statistics;
    }
}
