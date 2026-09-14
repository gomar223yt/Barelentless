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

import java.util.Random;

/**
 * A fully connected layer, {@code y = x W + b}.
 *
 * @author Barelentless
 */
public final class Linear extends Module {

    private final Tensor weight;
    private final Tensor bias;

    public Linear(int inputs, int outputs, boolean useBias, boolean kaiming, Random random) {
        this.weight = register("weight", kaiming
                ? Tensor.kaiming(inputs, outputs, random)
                : Tensor.xavier(inputs, outputs, random));
        this.bias = useBias ? register("bias", Tensor.param(1, outputs)) : null;
    }

    public Linear(int inputs, int outputs, Random random) {
        this(inputs, outputs, true, true, random);
    }

    public Tensor forward(Tensor input) {
        Tensor out = input.matmul(this.weight);
        return this.bias == null ? out : out.addRowVector(this.bias);
    }

    public Tensor weight() {
        return this.weight;
    }

    public Tensor bias() {
        return this.bias;
    }

    public int inputs() {
        return this.weight.rows;
    }

    public int outputs() {
        return this.weight.cols;
    }

    /**
     * Scales the weights of this layer down after construction. Used for output heads, where a near-zero
     * initialization means the model starts out as a no-op and only earns its influence through training - which is
     * exactly what we want when a head is wired into live movement code.
     */
    public Linear scaleInit(float factor) {
        for (int i = 0; i < this.weight.size(); i++) {
            this.weight.data[i] *= factor;
        }
        return this;
    }
}
