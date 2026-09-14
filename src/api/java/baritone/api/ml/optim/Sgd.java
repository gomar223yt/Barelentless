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

package baritone.api.ml.optim;

import baritone.api.ml.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Stochastic gradient descent with optional Nesterov momentum. Cheaper per step than {@link Adam} and, for the small
 * linear heads that get updated every single tick, entirely sufficient.
 *
 * @author Barelentless
 */
public final class Sgd extends Optimizer {

    private final float momentum;
    private final boolean nesterov;
    private final Map<Tensor, float[]> velocity = new IdentityHashMap<>();

    public Sgd(List<Tensor> parameters, float learningRate) {
        this(parameters, learningRate, 0f, false);
    }

    public Sgd(List<Tensor> parameters, float learningRate, float momentum, boolean nesterov) {
        super(parameters, learningRate);
        this.momentum = momentum;
        this.nesterov = nesterov;
        if (momentum != 0f) {
            for (Tensor parameter : parameters) {
                this.velocity.put(parameter, new float[parameter.size()]);
            }
        }
    }

    @Override
    protected void update(Tensor parameter, float gradientScale) {
        float[] gradient = parameter.gradient();
        if (this.momentum == 0f) {
            for (int i = 0; i < gradient.length; i++) {
                parameter.data[i] -= this.learningRate * gradient[i] * gradientScale;
            }
            return;
        }
        float[] v = this.velocity.get(parameter);
        for (int i = 0; i < gradient.length; i++) {
            float g = gradient[i] * gradientScale;
            v[i] = this.momentum * v[i] + g;
            float step = this.nesterov ? g + this.momentum * v[i] : v[i];
            parameter.data[i] -= this.learningRate * step;
        }
    }
}
