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
 * Differentiable objectives. Every method returns a scalar tensor that {@link Tensor#backward()} can be called on.
 *
 * @author Barelentless
 */
public final class Losses {

    private Losses() {}

    /**
     * Mean squared error.
     */
    public static Tensor mse(Tensor prediction, Tensor target) {
        return prediction.sub(target).square().mean();
    }

    /**
     * Huber (smooth L1) loss. The default for anything regressed from gameplay data: a single mistimed jump produces
     * an enormous residual, and squared error would let that one sample dominate the update.
     */
    public static Tensor huber(Tensor prediction, Tensor target, float delta) {
        Tensor difference = prediction.sub(target);
        final float[] values = difference.data;
        return Tensor.operation(1, 1, new Tensor[]{difference}, out -> {
            double sum = 0;
            for (float v : values) {
                float a = Math.abs(v);
                sum += a <= delta ? 0.5 * v * v : delta * (a - 0.5 * delta);
            }
            out.data[0] = (float) (sum / values.length);
            return g -> {
                float scale = g[0] / values.length;
                for (int i = 0; i < values.length; i++) {
                    float v = values[i];
                    float derivative = Math.abs(v) <= delta ? v : Math.signum(v) * delta;
                    Tensor.addGrad(difference, i, scale * derivative);
                }
            };
        });
    }

    /**
     * Categorical cross entropy from logits, averaged over rows. {@code targets[i]} is the correct class of row i.
     */
    public static Tensor crossEntropy(Tensor logits, int[] targets) {
        Tensor logProbabilities = logits.logSoftmax();
        final int rows = logits.rows;
        final int cols = logits.cols;
        return Tensor.operation(1, 1, new Tensor[]{logProbabilities}, out -> {
            double sum = 0;
            for (int r = 0; r < rows; r++) {
                sum -= logProbabilities.data[r * cols + targets[r]];
            }
            out.data[0] = (float) (sum / rows);
            return g -> {
                float scale = -g[0] / rows;
                for (int r = 0; r < rows; r++) {
                    Tensor.addGrad(logProbabilities, r * cols + targets[r], scale);
                }
            };
        });
    }

    /**
     * Binary cross entropy from logits, elementwise, averaged.
     */
    public static Tensor binaryCrossEntropy(Tensor logits, float[] targets) {
        final int n = logits.size();
        return Tensor.operation(1, 1, new Tensor[]{logits}, out -> {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                float x = logits.data[i];
                float y = targets[i];
                // log(1 + exp(-|x|)) + max(x, 0) - x * y, the stable form
                sum += Math.max(x, 0) - x * y + Math.log1p(Math.exp(-Math.abs(x)));
            }
            out.data[0] = (float) (sum / n);
            return g -> {
                float scale = g[0] / n;
                for (int i = 0; i < n; i++) {
                    float sigmoid = (float) (1.0 / (1.0 + Math.exp(-logits.data[i])));
                    Tensor.addGrad(logits, i, scale * (sigmoid - targets[i]));
                }
            };
        });
    }

    /**
     * Negative log likelihood of {@code target} under a diagonal Gaussian with the given mean and log standard
     * deviation. This is the objective behind the aim model: it learns not only where to point but how precisely,
     * so the controller can report an honest confidence instead of a fixed tolerance.
     */
    public static Tensor gaussianNll(Tensor mean, Tensor logStandardDeviation, Tensor target) {
        Tensor difference = mean.sub(target);
        Tensor variance = logStandardDeviation.scale(2f).exp();
        Tensor quadratic = difference.square().div(variance).scale(0.5f);
        return quadratic.add(logStandardDeviation).plus(0.5f * (float) Math.log(2 * Math.PI)).mean();
    }

    /**
     * Log probability of {@code value} under a diagonal Gaussian, summed over dimensions. Used by the policy
     * gradient trainers, which need per-sample log probabilities rather than a mean loss.
     */
    public static Tensor gaussianLogProbability(Tensor mean, Tensor logStandardDeviation, Tensor value) {
        Tensor difference = value.sub(mean);
        Tensor variance = logStandardDeviation.scale(2f).exp();
        Tensor term = difference.square().div(variance).scale(0.5f)
                .add(logStandardDeviation)
                .plus(0.5f * (float) Math.log(2 * Math.PI));
        return term.sumRows().neg();
    }

    /**
     * Entropy of a diagonal Gaussian, per sample. Policy gradient methods add this to the objective to stop the
     * policy collapsing onto one deterministic answer before it has explored.
     */
    public static Tensor gaussianEntropy(Tensor logStandardDeviation) {
        return logStandardDeviation.plus(0.5f * (float) (Math.log(2 * Math.PI) + 1)).sumRows();
    }

    /**
     * Entropy of a categorical distribution given its logits, per row.
     */
    public static Tensor categoricalEntropy(Tensor logits) {
        Tensor logProbabilities = logits.logSoftmax();
        Tensor probabilities = logits.softmax();
        return probabilities.mul(logProbabilities).sumRows().neg();
    }

    /**
     * Log probability of a set of independent binary actions under their logits, summed per row.
     * <p>
     * This is how key presses are scored: forward, jump, sprint and sneak are each their own Bernoulli, so the policy
     * can hold "definitely forward, maybe jump" rather than being forced to pick one key combination out of a
     * flattened list of all of them.
     */
    public static Tensor bernoulliLogProbability(Tensor logits, boolean[][] actions) {
        final int rows = logits.rows;
        final int cols = logits.cols;
        return Tensor.operation(rows, 1, new Tensor[]{logits}, out -> {
            for (int r = 0; r < rows; r++) {
                double sum = 0;
                for (int c = 0; c < cols; c++) {
                    float x = logits.data[r * cols + c];
                    // log sigmoid(x) for a true action, log sigmoid(-x) for a false one, stable form
                    float signed = actions[r][c] ? x : -x;
                    sum += signed >= 0
                            ? -Math.log1p(Math.exp(-signed))
                            : signed - Math.log1p(Math.exp(signed));
                }
                out.data[r] = (float) sum;
            }
            return g -> {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        float sigmoid = (float) (1.0 / (1.0 + Math.exp(-logits.data[r * cols + c])));
                        float target = actions[r][c] ? 1f : 0f;
                        Tensor.addGrad(logits, r * cols + c, g[r] * (target - sigmoid));
                    }
                }
            };
        });
    }

    /**
     * Entropy of independent Bernoulli actions, summed per row.
     */
    public static Tensor bernoulliEntropy(Tensor logits) {
        final int rows = logits.rows;
        final int cols = logits.cols;
        return Tensor.operation(rows, 1, new Tensor[]{logits}, out -> {
            for (int r = 0; r < rows; r++) {
                double sum = 0;
                for (int c = 0; c < cols; c++) {
                    double p = 1.0 / (1.0 + Math.exp(-logits.data[r * cols + c]));
                    double q = 1 - p;
                    sum -= p * Math.log(Math.max(1e-9, p)) + q * Math.log(Math.max(1e-9, q));
                }
                out.data[r] = (float) sum;
            }
            return g -> {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        double p = 1.0 / (1.0 + Math.exp(-logits.data[r * cols + c]));
                        // d/dz of the binary entropy is -sigmoid'(z) * logit(z)
                        double derivative = -p * (1 - p) * Math.log(Math.max(1e-9, p) / Math.max(1e-9, 1 - p));
                        Tensor.addGrad(logits, r * cols + c, (float) (g[r] * derivative));
                    }
                }
            };
        });
    }

    /**
     * Kullback-Leibler divergence between two categorical distributions given as logits, per row. Used to keep an
     * updated policy from moving too far from the one that generated the data.
     */
    public static Tensor categoricalKl(Tensor logitsP, Tensor logitsQ) {
        Tensor logP = logitsP.logSoftmax();
        Tensor logQ = logitsQ.logSoftmax();
        return logitsP.softmax().mul(logP.sub(logQ)).sumRows();
    }
}
