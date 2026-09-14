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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Random;

/**
 * A dense, row-major, two dimensional float matrix with reverse mode automatic differentiation.
 * <p>
 * Everything in the learning stack is expressed in terms of this type. A rank one vector is simply a tensor with a
 * single row, and a sequence of {@code n} observations of {@code d} features is a tensor of shape {@code [n, d]}. Two
 * dimensions are enough to express every model that runs inside the game loop (linear layers, attention, recurrences)
 * while keeping the kernels simple enough to stay allocation-light and predictable, which matters a lot more here than
 * generality does: inference happens on the client thread, between two ticks.
 * <p>
 * Autodiff works by taping. Each operation produces a new tensor that remembers its parents and a closure that knows
 * how to push a gradient from the output back into those parents. {@link #backward()} walks that tape in reverse
 * topological order exactly once per node, so a diamond shaped graph accumulates correctly instead of recomputing.
 * Tensors created by {@link #param} (or any operation touching one) participate in gradients; constants do not, and
 * their branches of the graph are skipped entirely.
 *
 * @author Barelentless
 */
public final class Tensor {

    /**
     * Back-propagation step for a single operation. Receives the gradient of the loss with respect to this tensor's
     * output and is responsible for accumulating the corresponding gradients into the parents.
     */
    @FunctionalInterface
    public interface Backward {

        void apply(float[] gradOutput);
    }

    public final int rows;
    public final int cols;
    public final float[] data;

    /**
     * Gradient accumulator, lazily allocated. Only non-null once this tensor takes part in a differentiable graph.
     */
    private float[] grad;

    private boolean requiresGrad;
    private Backward backward;
    private Tensor[] parents;

    /**
     * Set while a tensor is a leaf that an optimizer owns. Distinguishes "trainable weight" from "intermediate result".
     */
    private boolean parameter;

    private String name = "";

    public Tensor(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("bad shape " + rows + "x" + cols);
        }
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows * cols];
    }

    public Tensor(int rows, int cols, float[] data) {
        if (data.length != rows * cols) {
            throw new IllegalArgumentException("data length " + data.length + " != " + rows + "x" + cols);
        }
        this.rows = rows;
        this.cols = cols;
        this.data = data;
    }

    // ------------------------------------------------------------------------------------------------ construction

    public static Tensor zeros(int rows, int cols) {
        return new Tensor(rows, cols);
    }

    public static Tensor filled(int rows, int cols, float value) {
        Tensor t = new Tensor(rows, cols);
        Arrays.fill(t.data, value);
        return t;
    }

    public static Tensor scalar(float value) {
        return new Tensor(1, 1, new float[]{value});
    }

    public static Tensor vector(float... values) {
        return new Tensor(1, values.length, values.clone());
    }

    public static Tensor of(int rows, int cols, float... values) {
        return new Tensor(rows, cols, values.clone());
    }

    /**
     * Creates a trainable leaf. Only parameters and tensors derived from them carry gradients.
     */
    public static Tensor param(int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        t.requiresGrad = true;
        t.parameter = true;
        return t;
    }

    public static Tensor param(Tensor from) {
        Tensor t = new Tensor(from.rows, from.cols, from.data.clone());
        t.requiresGrad = true;
        t.parameter = true;
        return t;
    }

    /**
     * Kaiming/He normal initialization, the sane default for ReLU-family activations.
     */
    public static Tensor kaiming(int rows, int cols, Random random) {
        Tensor t = param(rows, cols);
        double std = Math.sqrt(2.0 / rows);
        for (int i = 0; i < t.data.length; i++) {
            t.data[i] = (float) (random.nextGaussian() * std);
        }
        return t;
    }

    /**
     * Xavier/Glorot uniform initialization, used for tanh/sigmoid gates and attention projections.
     */
    public static Tensor xavier(int rows, int cols, Random random) {
        Tensor t = param(rows, cols);
        double limit = Math.sqrt(6.0 / (rows + cols));
        for (int i = 0; i < t.data.length; i++) {
            t.data[i] = (float) ((random.nextDouble() * 2 - 1) * limit);
        }
        return t;
    }

    // ---------------------------------------------------------------------------------------------------- metadata

    public int size() {
        return this.data.length;
    }

    public boolean requiresGrad() {
        return this.requiresGrad;
    }

    public boolean isParameter() {
        return this.parameter;
    }

    public String getName() {
        return this.name;
    }

    public Tensor named(String name) {
        this.name = name;
        return this;
    }

    public float get(int row, int col) {
        return this.data[row * this.cols + col];
    }

    public void set(int row, int col, float value) {
        this.data[row * this.cols + col] = value;
    }

    public float item() {
        if (this.data.length != 1) {
            throw new IllegalStateException("item() on " + this.rows + "x" + this.cols);
        }
        return this.data[0];
    }

    public float[] gradient() {
        return this.grad;
    }

    public float[] gradientOrAllocate() {
        if (this.grad == null) {
            this.grad = new float[this.data.length];
        }
        return this.grad;
    }

    public void zeroGrad() {
        if (this.grad != null) {
            Arrays.fill(this.grad, 0f);
        }
    }

    /**
     * Drops the recorded graph but keeps the values, turning this tensor back into a leaf. Used on recurrent hidden
     * states so that a long lived rollout doesn't retain every tick it has ever seen.
     */
    public Tensor detach() {
        Tensor t = new Tensor(this.rows, this.cols, this.data.clone());
        return t;
    }

    public Tensor copy() {
        Tensor t = new Tensor(this.rows, this.cols, this.data.clone());
        t.requiresGrad = this.requiresGrad;
        t.parameter = this.parameter;
        t.name = this.name;
        return t;
    }

    public float[] row(int row) {
        return Arrays.copyOfRange(this.data, row * this.cols, (row + 1) * this.cols);
    }

    public void setRow(int row, float[] values) {
        System.arraycopy(values, 0, this.data, row * this.cols, this.cols);
    }

    // ------------------------------------------------------------------------------------------------------- graph

    private Tensor result(int rows, int cols, Tensor[] parents, Backward backward) {
        Tensor out = new Tensor(rows, cols);
        boolean any = false;
        for (Tensor parent : parents) {
            if (parent != null && parent.requiresGrad) {
                any = true;
                break;
            }
        }
        if (any) {
            out.requiresGrad = true;
            out.parents = parents;
            out.backward = backward;
        }
        return out;
    }

    private static void accumulate(Tensor target, int index, float value) {
        if (target != null && target.requiresGrad) {
            target.gradientOrAllocate()[index] += value;
        }
    }

    /**
     * Seeds this scalar with a gradient of one and propagates it through the whole recorded graph.
     */
    public void backward() {
        if (this.data.length != 1) {
            throw new IllegalStateException("backward() requires a scalar, got " + this.rows + "x" + this.cols);
        }
        this.gradientOrAllocate()[0] = 1f;
        this.backwardFrom();
    }

    /**
     * Propagates whatever gradient has already been placed on this tensor. Useful when a loss is assembled from
     * several independent heads that each seed their own upstream gradient.
     */
    public void backwardFrom() {
        List<Tensor> order = new ArrayList<>();
        IdentityHashMap<Tensor, Boolean> seen = new IdentityHashMap<>();
        topo(this, seen, order);
        for (int i = order.size() - 1; i >= 0; i--) {
            Tensor t = order.get(i);
            if (t.backward != null && t.grad != null) {
                t.backward.apply(t.grad);
            }
        }
    }

    private static void topo(Tensor node, IdentityHashMap<Tensor, Boolean> seen, List<Tensor> order) {
        // iterative to keep deep unrolled recurrences off the JVM stack
        ArrayList<Tensor> stack = new ArrayList<>();
        ArrayList<Integer> cursor = new ArrayList<>();
        stack.add(node);
        cursor.add(0);
        seen.put(node, Boolean.TRUE);
        while (!stack.isEmpty()) {
            int top = stack.size() - 1;
            Tensor current = stack.get(top);
            int index = cursor.get(top);
            Tensor[] parents = current.parents;
            if (parents != null && index < parents.length) {
                cursor.set(top, index + 1);
                Tensor parent = parents[index];
                if (parent != null && parent.requiresGrad && !seen.containsKey(parent)) {
                    seen.put(parent, Boolean.TRUE);
                    stack.add(parent);
                    cursor.add(0);
                }
            } else {
                stack.remove(top);
                cursor.remove(top);
                order.add(current);
            }
        }
    }

    /**
     * Escape hatch for defining a differentiable operation outside this class.
     * <p>
     * The body receives the freshly allocated output tensor, fills in its values, and returns the closure that pushes
     * gradients back into {@code parents}. Layers whose kernel is more efficient written by hand than composed out of
     * primitives (layer norm, attention, gathers) are built this way, and they participate in the tape exactly like
     * the built-in operations do.
     *
     * @param rows    Row count of the output
     * @param cols    Column count of the output
     * @param parents The tensors this operation reads from
     * @param body    Computes the output and returns its backward pass
     * @return The output tensor, wired into the graph
     */
    public static Tensor operation(int rows, int cols, Tensor[] parents, java.util.function.Function<Tensor, Backward> body) {
        Tensor out = new Tensor(rows, cols);
        boolean any = false;
        for (Tensor parent : parents) {
            if (parent != null && parent.requiresGrad) {
                any = true;
                break;
            }
        }
        Backward backward = body.apply(out);
        if (any) {
            out.requiresGrad = true;
            out.parents = parents;
            out.backward = backward;
        }
        return out;
    }

    /**
     * Accumulates a gradient contribution into a tensor, ignoring tensors that don't take part in gradients.
     * Intended for use from {@link #operation} bodies.
     */
    public static void addGrad(Tensor target, int index, float value) {
        accumulate(target, index, value);
    }

    // ---------------------------------------------------------------------------------------------------- elementwise

    public Tensor add(Tensor other) {
        if (other.rows == 1 && this.rows != 1) {
            return this.addRowVector(other);
        }
        requireSameShape(other, "add");
        Tensor out = result(this.rows, this.cols, new Tensor[]{this, other}, g -> {
            for (int i = 0; i < g.length; i++) {
                accumulate(this, i, g[i]);
                accumulate(other, i, g[i]);
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = this.data[i] + other.data[i];
        }
        return out;
    }

    /**
     * Broadcasts a {@code [1, cols]} bias across every row.
     */
    public Tensor addRowVector(Tensor bias) {
        if (bias.rows != 1 || bias.cols != this.cols) {
            throw new IllegalArgumentException("bias must be 1x" + this.cols + ", got " + bias.rows + "x" + bias.cols);
        }
        Tensor out = result(this.rows, this.cols, new Tensor[]{this, bias}, g -> {
            for (int r = 0; r < this.rows; r++) {
                int base = r * this.cols;
                for (int c = 0; c < this.cols; c++) {
                    accumulate(this, base + c, g[base + c]);
                    accumulate(bias, c, g[base + c]);
                }
            }
        });
        for (int r = 0; r < this.rows; r++) {
            int base = r * this.cols;
            for (int c = 0; c < this.cols; c++) {
                out.data[base + c] = this.data[base + c] + bias.data[c];
            }
        }
        return out;
    }

    public Tensor sub(Tensor other) {
        requireSameShape(other, "sub");
        Tensor out = result(this.rows, this.cols, new Tensor[]{this, other}, g -> {
            for (int i = 0; i < g.length; i++) {
                accumulate(this, i, g[i]);
                accumulate(other, i, -g[i]);
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = this.data[i] - other.data[i];
        }
        return out;
    }

    public Tensor mul(Tensor other) {
        requireSameShape(other, "mul");
        Tensor out = result(this.rows, this.cols, new Tensor[]{this, other}, g -> {
            for (int i = 0; i < g.length; i++) {
                accumulate(this, i, g[i] * other.data[i]);
                accumulate(other, i, g[i] * this.data[i]);
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = this.data[i] * other.data[i];
        }
        return out;
    }

    public Tensor div(Tensor other) {
        requireSameShape(other, "div");
        Tensor out = result(this.rows, this.cols, new Tensor[]{this, other}, g -> {
            for (int i = 0; i < g.length; i++) {
                float d = other.data[i];
                accumulate(this, i, g[i] / d);
                accumulate(other, i, -g[i] * this.data[i] / (d * d));
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = this.data[i] / other.data[i];
        }
        return out;
    }

    public Tensor scale(float factor) {
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, g -> {
            for (int i = 0; i < g.length; i++) {
                accumulate(this, i, g[i] * factor);
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = this.data[i] * factor;
        }
        return out;
    }

    public Tensor plus(float offset) {
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, g -> {
            for (int i = 0; i < g.length; i++) {
                accumulate(this, i, g[i]);
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = this.data[i] + offset;
        }
        return out;
    }

    public Tensor neg() {
        return this.scale(-1f);
    }

    public Tensor square() {
        return this.mul(this);
    }

    public Tensor sqrt() {
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, g -> {
            for (int i = 0; i < g.length; i++) {
                accumulate(this, i, g[i] * 0.5f / (float) Math.max(1e-12, Math.sqrt(this.data[i])));
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = (float) Math.sqrt(this.data[i]);
        }
        return out;
    }

    public Tensor exp() {
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, null);
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = (float) Math.exp(this.data[i]);
        }
        out.bind(new Tensor[]{this}, g -> {
            for (int i = 0; i < g.length; i++) {
                accumulate(this, i, g[i] * out.data[i]);
            }
        });
        return out;
    }

    public Tensor log() {
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, g -> {
            for (int i = 0; i < g.length; i++) {
                accumulate(this, i, g[i] / Math.max(1e-12f, this.data[i]));
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = (float) Math.log(Math.max(1e-12f, this.data[i]));
        }
        return out;
    }

    public Tensor abs() {
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, g -> {
            for (int i = 0; i < g.length; i++) {
                accumulate(this, i, this.data[i] >= 0 ? g[i] : -g[i]);
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = Math.abs(this.data[i]);
        }
        return out;
    }

    /**
     * Rebinds the backward closure of a tensor that had to be allocated before its closure could capture it.
     */
    private void bind(Tensor[] parents, Backward backward) {
        if (this.requiresGrad) {
            this.parents = parents;
            this.backward = backward;
        }
    }

    /**
     * Clamps every element into {@code [low, high]}. Gradient flows only through elements that were inside the range,
     * which is exactly the behaviour the clipped policy objective relies on.
     */
    public Tensor clamp(float low, float high) {
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, g -> {
            for (int i = 0; i < g.length; i++) {
                float v = this.data[i];
                if (v >= low && v <= high) {
                    accumulate(this, i, g[i]);
                }
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = Math.min(high, Math.max(low, this.data[i]));
        }
        return out;
    }

    /**
     * Elementwise minimum. The gradient goes to whichever side was selected.
     */
    public Tensor min(Tensor other) {
        requireSameShape(other, "min");
        Tensor out = result(this.rows, this.cols, new Tensor[]{this, other}, g -> {
            for (int i = 0; i < g.length; i++) {
                if (this.data[i] <= other.data[i]) {
                    accumulate(this, i, g[i]);
                } else {
                    accumulate(other, i, g[i]);
                }
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = Math.min(this.data[i], other.data[i]);
        }
        return out;
    }

    // ---------------------------------------------------------------------------------------------------- activations

    public Tensor relu() {
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, g -> {
            for (int i = 0; i < g.length; i++) {
                if (this.data[i] > 0) {
                    accumulate(this, i, g[i]);
                }
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = Math.max(0f, this.data[i]);
        }
        return out;
    }

    public Tensor leakyRelu(float slope) {
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, g -> {
            for (int i = 0; i < g.length; i++) {
                accumulate(this, i, this.data[i] > 0 ? g[i] : g[i] * slope);
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = this.data[i] > 0 ? this.data[i] : this.data[i] * slope;
        }
        return out;
    }

    /**
     * The tanh approximation of GELU, which is what most transformer implementations actually ship.
     */
    public Tensor gelu() {
        final float c = (float) Math.sqrt(2.0 / Math.PI);
        float[] inner = new float[this.data.length];
        float[] tanhInner = new float[this.data.length];
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, g -> {
            for (int i = 0; i < g.length; i++) {
                float x = this.data[i];
                float t = tanhInner[i];
                float dInner = c * (1f + 3f * 0.044715f * x * x);
                float derivative = 0.5f * (1f + t) + 0.5f * x * (1f - t * t) * dInner;
                accumulate(this, i, g[i] * derivative);
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            float x = this.data[i];
            inner[i] = c * (x + 0.044715f * x * x * x);
            tanhInner[i] = (float) Math.tanh(inner[i]);
            out.data[i] = 0.5f * x * (1f + tanhInner[i]);
        }
        return out;
    }

    public Tensor tanh() {
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, null);
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = (float) Math.tanh(this.data[i]);
        }
        out.bind(new Tensor[]{this}, g -> {
            for (int i = 0; i < g.length; i++) {
                float t = out.data[i];
                accumulate(this, i, g[i] * (1f - t * t));
            }
        });
        return out;
    }

    public Tensor sigmoid() {
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, null);
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = (float) (1.0 / (1.0 + Math.exp(-this.data[i])));
        }
        out.bind(new Tensor[]{this}, g -> {
            for (int i = 0; i < g.length; i++) {
                float s = out.data[i];
                accumulate(this, i, g[i] * s * (1f - s));
            }
        });
        return out;
    }

    public Tensor softplus() {
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, g -> {
            for (int i = 0; i < g.length; i++) {
                accumulate(this, i, (float) (g[i] / (1.0 + Math.exp(-this.data[i]))));
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            // numerically stable log1p(exp(x))
            float x = this.data[i];
            out.data[i] = x > 20 ? x : (float) Math.log1p(Math.exp(x));
        }
        return out;
    }

    /**
     * Row-wise softmax, computed in a numerically stable way (max subtracted before exponentiating).
     */
    public Tensor softmax() {
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, null);
        for (int r = 0; r < this.rows; r++) {
            int base = r * this.cols;
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < this.cols; c++) {
                max = Math.max(max, this.data[base + c]);
            }
            float sum = 0;
            for (int c = 0; c < this.cols; c++) {
                float e = (float) Math.exp(this.data[base + c] - max);
                out.data[base + c] = e;
                sum += e;
            }
            for (int c = 0; c < this.cols; c++) {
                out.data[base + c] /= sum;
            }
        }
        out.bind(new Tensor[]{this}, g -> {
            for (int r = 0; r < this.rows; r++) {
                int base = r * this.cols;
                float dot = 0;
                for (int c = 0; c < this.cols; c++) {
                    dot += g[base + c] * out.data[base + c];
                }
                for (int c = 0; c < this.cols; c++) {
                    accumulate(this, base + c, out.data[base + c] * (g[base + c] - dot));
                }
            }
        });
        return out;
    }

    /**
     * Row-wise log-softmax. Preferred over {@code softmax().log()} for losses: it avoids the exp/log round trip and
     * stays finite for saturated logits.
     */
    public Tensor logSoftmax() {
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, null);
        float[] soft = new float[this.data.length];
        for (int r = 0; r < this.rows; r++) {
            int base = r * this.cols;
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < this.cols; c++) {
                max = Math.max(max, this.data[base + c]);
            }
            double sum = 0;
            for (int c = 0; c < this.cols; c++) {
                sum += Math.exp(this.data[base + c] - max);
            }
            float logSum = (float) (max + Math.log(sum));
            for (int c = 0; c < this.cols; c++) {
                out.data[base + c] = this.data[base + c] - logSum;
                soft[base + c] = (float) Math.exp(out.data[base + c]);
            }
        }
        out.bind(new Tensor[]{this}, g -> {
            for (int r = 0; r < this.rows; r++) {
                int base = r * this.cols;
                float sum = 0;
                for (int c = 0; c < this.cols; c++) {
                    sum += g[base + c];
                }
                for (int c = 0; c < this.cols; c++) {
                    accumulate(this, base + c, g[base + c] - soft[base + c] * sum);
                }
            }
        });
        return out;
    }

    // -------------------------------------------------------------------------------------------------- linear algebra

    /**
     * Standard matrix product: {@code [n, k] x [k, m] -> [n, m]}.
     */
    public Tensor matmul(Tensor other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("matmul shape mismatch " + this.rows + "x" + this.cols
                    + " times " + other.rows + "x" + other.cols);
        }
        final int n = this.rows;
        final int k = this.cols;
        final int m = other.cols;
        Tensor out = result(n, m, new Tensor[]{this, other}, g -> {
            if (this.requiresGrad) {
                float[] ga = this.gradientOrAllocate();
                for (int i = 0; i < n; i++) {
                    for (int p = 0; p < k; p++) {
                        float sum = 0;
                        int bRow = p * m;
                        int gRow = i * m;
                        for (int j = 0; j < m; j++) {
                            sum += g[gRow + j] * other.data[bRow + j];
                        }
                        ga[i * k + p] += sum;
                    }
                }
            }
            if (other.requiresGrad) {
                float[] gb = other.gradientOrAllocate();
                for (int p = 0; p < k; p++) {
                    for (int j = 0; j < m; j++) {
                        float sum = 0;
                        for (int i = 0; i < n; i++) {
                            sum += this.data[i * k + p] * g[i * m + j];
                        }
                        gb[p * m + j] += sum;
                    }
                }
            }
        });
        for (int i = 0; i < n; i++) {
            int aRow = i * k;
            int oRow = i * m;
            for (int p = 0; p < k; p++) {
                float a = this.data[aRow + p];
                if (a == 0f) {
                    continue;
                }
                int bRow = p * m;
                for (int j = 0; j < m; j++) {
                    out.data[oRow + j] += a * other.data[bRow + j];
                }
            }
        }
        return out;
    }

    public Tensor transpose() {
        Tensor out = result(this.cols, this.rows, new Tensor[]{this}, g -> {
            for (int r = 0; r < this.rows; r++) {
                for (int c = 0; c < this.cols; c++) {
                    accumulate(this, r * this.cols + c, g[c * this.rows + r]);
                }
            }
        });
        for (int r = 0; r < this.rows; r++) {
            for (int c = 0; c < this.cols; c++) {
                out.data[c * this.rows + r] = this.data[r * this.cols + c];
            }
        }
        return out;
    }

    // -------------------------------------------------------------------------------------------------- reductions

    public Tensor sum() {
        Tensor out = result(1, 1, new Tensor[]{this}, g -> {
            for (int i = 0; i < this.data.length; i++) {
                accumulate(this, i, g[0]);
            }
        });
        float sum = 0;
        for (float v : this.data) {
            sum += v;
        }
        out.data[0] = sum;
        return out;
    }

    public Tensor mean() {
        final float inv = 1f / this.data.length;
        Tensor out = result(1, 1, new Tensor[]{this}, g -> {
            for (int i = 0; i < this.data.length; i++) {
                accumulate(this, i, g[0] * inv);
            }
        });
        float sum = 0;
        for (float v : this.data) {
            sum += v;
        }
        out.data[0] = sum * inv;
        return out;
    }

    /**
     * Sums along the row axis, producing a {@code [rows, 1]} column.
     */
    public Tensor sumRows() {
        Tensor out = result(this.rows, 1, new Tensor[]{this}, g -> {
            for (int r = 0; r < this.rows; r++) {
                for (int c = 0; c < this.cols; c++) {
                    accumulate(this, r * this.cols + c, g[r]);
                }
            }
        });
        for (int r = 0; r < this.rows; r++) {
            float sum = 0;
            int base = r * this.cols;
            for (int c = 0; c < this.cols; c++) {
                sum += this.data[base + c];
            }
            out.data[r] = sum;
        }
        return out;
    }

    /**
     * Averages across rows, producing a {@code [1, cols]} row. This is the pooling step that turns a sequence of
     * encoded ticks into a single summary vector.
     */
    public Tensor meanRows() {
        final float inv = 1f / this.rows;
        Tensor out = result(1, this.cols, new Tensor[]{this}, g -> {
            for (int r = 0; r < this.rows; r++) {
                for (int c = 0; c < this.cols; c++) {
                    accumulate(this, r * this.cols + c, g[c] * inv);
                }
            }
        });
        for (int r = 0; r < this.rows; r++) {
            int base = r * this.cols;
            for (int c = 0; c < this.cols; c++) {
                out.data[c] += this.data[base + c] * inv;
            }
        }
        return out;
    }

    // ----------------------------------------------------------------------------------------------------- slicing

    public Tensor sliceRows(int from, int to) {
        final int n = to - from;
        Tensor out = result(n, this.cols, new Tensor[]{this}, g -> {
            for (int i = 0; i < n * this.cols; i++) {
                accumulate(this, from * this.cols + i, g[i]);
            }
        });
        System.arraycopy(this.data, from * this.cols, out.data, 0, n * this.cols);
        return out;
    }

    public Tensor sliceCols(int from, int to) {
        final int width = to - from;
        Tensor out = result(this.rows, width, new Tensor[]{this}, g -> {
            for (int r = 0; r < this.rows; r++) {
                for (int c = 0; c < width; c++) {
                    accumulate(this, r * this.cols + from + c, g[r * width + c]);
                }
            }
        });
        for (int r = 0; r < this.rows; r++) {
            System.arraycopy(this.data, r * this.cols + from, out.data, r * width, width);
        }
        return out;
    }

    public static Tensor concatCols(Tensor... parts) {
        int rows = parts[0].rows;
        int cols = 0;
        for (Tensor part : parts) {
            if (part.rows != rows) {
                throw new IllegalArgumentException("concatCols row mismatch");
            }
            cols += part.cols;
        }
        final int total = cols;
        Tensor probe = new Tensor(rows, cols);
        boolean any = false;
        for (Tensor part : parts) {
            any |= part.requiresGrad;
        }
        Tensor out = probe;
        if (any) {
            out.requiresGrad = true;
            out.bindPublic(parts, g -> {
                int offset = 0;
                for (Tensor part : parts) {
                    for (int r = 0; r < rows; r++) {
                        for (int c = 0; c < part.cols; c++) {
                            accumulate(part, r * part.cols + c, g[r * total + offset + c]);
                        }
                    }
                    offset += part.cols;
                }
            });
        }
        int offset = 0;
        for (Tensor part : parts) {
            for (int r = 0; r < rows; r++) {
                System.arraycopy(part.data, r * part.cols, out.data, r * total + offset, part.cols);
            }
            offset += part.cols;
        }
        return out;
    }

    public static Tensor concatRows(Tensor... parts) {
        int cols = parts[0].cols;
        int rows = 0;
        for (Tensor part : parts) {
            if (part.cols != cols) {
                throw new IllegalArgumentException("concatRows col mismatch");
            }
            rows += part.rows;
        }
        Tensor out = new Tensor(rows, cols);
        boolean any = false;
        for (Tensor part : parts) {
            any |= part.requiresGrad;
        }
        if (any) {
            out.requiresGrad = true;
            out.bindPublic(parts, g -> {
                int offset = 0;
                for (Tensor part : parts) {
                    for (int i = 0; i < part.data.length; i++) {
                        accumulate(part, i, g[offset + i]);
                    }
                    offset += part.data.length;
                }
            });
        }
        int offset = 0;
        for (Tensor part : parts) {
            System.arraycopy(part.data, 0, out.data, offset, part.data.length);
            offset += part.data.length;
        }
        return out;
    }

    private void bindPublic(Tensor[] parents, Backward backward) {
        this.parents = parents;
        this.backward = backward;
    }

    /**
     * Additive mask used by attention: adds {@code -inf}-like penalties to disallowed positions. The mask is a
     * constant, so no gradient flows into it.
     */
    public Tensor maskAdd(float[] mask) {
        Tensor out = result(this.rows, this.cols, new Tensor[]{this}, g -> {
            for (int i = 0; i < g.length; i++) {
                accumulate(this, i, g[i]);
            }
        });
        for (int i = 0; i < this.data.length; i++) {
            out.data[i] = this.data[i] + mask[i];
        }
        return out;
    }

    // ------------------------------------------------------------------------------------------------------ helpers

    private void requireSameShape(Tensor other, String op) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException(op + " shape mismatch " + this.rows + "x" + this.cols
                    + " vs " + other.rows + "x" + other.cols);
        }
    }

    public boolean isFinite() {
        for (float v : this.data) {
            if (Float.isNaN(v) || Float.isInfinite(v)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor[").append(this.rows).append('x').append(this.cols).append(']');
        if (this.data.length <= 32) {
            sb.append(' ').append(Arrays.toString(this.data));
        }
        return sb.toString();
    }
}
