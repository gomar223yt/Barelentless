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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A plain feed-forward stack with optional layer norm between hidden layers. The workhorse for the heads that sit on
 * top of the encoder.
 *
 * @author Barelentless
 */
public final class Mlp extends Module {

    private final List<Linear> layers = new ArrayList<>();
    private final List<LayerNorm> norms = new ArrayList<>();
    private final Activation activation;
    private final Activation outputActivation;

    public Mlp(int inputs, int[] hidden, int outputs, Activation activation, Activation outputActivation,
               boolean normalize, Random random) {
        this.activation = activation;
        this.outputActivation = outputActivation;
        int previous = inputs;
        for (int i = 0; i < hidden.length; i++) {
            Linear layer = child("fc" + i, new Linear(previous, hidden[i], true, activation.prefersKaiming(), random));
            this.layers.add(layer);
            if (normalize) {
                this.norms.add(child("norm" + i, new LayerNorm(hidden[i])));
            }
            previous = hidden[i];
        }
        this.layers.add(child("out", new Linear(previous, outputs, true, false, random)));
    }

    public Tensor forward(Tensor input) {
        Tensor current = input;
        for (int i = 0; i < this.layers.size() - 1; i++) {
            current = this.layers.get(i).forward(current);
            if (!this.norms.isEmpty()) {
                current = this.norms.get(i).forward(current);
            }
            current = this.activation.apply(current);
        }
        return this.outputActivation.apply(this.layers.get(this.layers.size() - 1).forward(current));
    }

    /**
     * Shrinks the output layer so the whole network starts out close to zero. See {@link Linear#scaleInit}.
     */
    public Mlp scaleOutput(float factor) {
        this.layers.get(this.layers.size() - 1).scaleInit(factor);
        return this;
    }

    public Linear outputLayer() {
        return this.layers.get(this.layers.size() - 1);
    }
}
