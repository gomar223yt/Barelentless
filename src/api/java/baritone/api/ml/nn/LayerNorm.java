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

package baritone.api.ml.nn;

import baritone.api.ml.Tensor;

import java.util.Arrays;

/**
 * Per-row layer normalization with a learnable gain and shift.
 * <p>
 * Batch normalization is useless here: inference happens one state at a time inside a game tick, so there is no batch
 * to take statistics over. Layer norm normalizes across features of a single sample, which keeps training and
 * inference numerically identical - important when a model trained on a background thread is read live by the
 * movement code.
 *
 * @author Barelentless
 */
public final class LayerNorm extends Module {

    private static final float EPSILON = 1e-5f;

    private final Tensor gain;
    private final Tensor shift;

    public LayerNorm(int features) {
        this.gain = register("gain", Tensor.param(1, features));
        this.shift = register("shift", Tensor.param(1, features));
        Arrays.fill(this.gain.data, 1f);
    }

    public Tensor forward(Tensor input) {
        final int rows = input.rows;
        final int cols = input.cols;
        final float[] mean = new float[rows];
        final float[] invStd = new float[rows];
        final float[] normalized = new float[rows * cols];
        return Tensor.operation(rows, cols, new Tensor[]{input, this.gain, this.shift}, out -> {
            for (int r = 0; r < rows; r++) {
                int base = r * cols;
                float sum = 0;
                for (int c = 0; c < cols; c++) {
                    sum += input.data[base + c];
                }
                float m = sum / cols;
                float variance = 0;
                for (int c = 0; c < cols; c++) {
                    float d = input.data[base + c] - m;
                    variance += d * d;
                }
                variance /= cols;
                float inv = (float) (1.0 / Math.sqrt(variance + EPSILON));
                mean[r] = m;
                invStd[r] = inv;
                for (int c = 0; c < cols; c++) {
                    float n = (input.data[base + c] - m) * inv;
                    normalized[base + c] = n;
                    out.data[base + c] = n * this.gain.data[c] + this.shift.data[c];
                }
            }
            return g -> {
                for (int r = 0; r < rows; r++) {
                    int base = r * cols;
                    // gradient wrt the normalized values, then the standard layer-norm backward contraction
                    float[] dn = new float[cols];
                    float sumDn = 0;
                    float sumDnN = 0;
                    for (int c = 0; c < cols; c++) {
                        float gc = g[base + c];
                        Tensor.addGrad(this.gain, c, gc * normalized[base + c]);
                        Tensor.addGrad(this.shift, c, gc);
                        dn[c] = gc * this.gain.data[c];
                        sumDn += dn[c];
                        sumDnN += dn[c] * normalized[base + c];
                    }
                    float inv = invStd[r];
                    for (int c = 0; c < cols; c++) {
                        float value = (dn[c] - sumDn / cols - normalized[base + c] * sumDnN / cols) * inv;
                        Tensor.addGrad(input, base + c, value);
                    }
                }
            };
        });
    }

    public int features() {
        return this.gain.cols;
    }

    /**
     * Resets the gain to one and the shift to zero, the canonical starting point.
     */
    public void resetParameters() {
        Arrays.fill(this.gain.data, 1f);
        Arrays.fill(this.shift.data, 0f);
    }
}
