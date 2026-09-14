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
import java.util.function.UnaryOperator;

/**
 * Chains differentiable stages into a single module. Stages may be modules or plain functions (activations); either
 * way they are applied in order and any parameters they own are registered for training.
 *
 * @author Barelentless
 */
public final class Sequential extends Module {

    private final List<UnaryOperator<Tensor>> stages = new ArrayList<>();

    public Sequential add(String name, Linear layer) {
        child(name, layer);
        this.stages.add(layer::forward);
        return this;
    }

    public Sequential add(String name, LayerNorm layer) {
        child(name, layer);
        this.stages.add(layer::forward);
        return this;
    }

    public Sequential add(String name, Dropout layer) {
        child(name, layer);
        this.stages.add(layer::forward);
        return this;
    }

    public Sequential add(String name, TransformerBlock layer) {
        child(name, layer);
        this.stages.add(layer::forward);
        return this;
    }

    public Sequential add(Activation activation) {
        this.stages.add(activation);
        return this;
    }

    public Tensor forward(Tensor input) {
        Tensor current = input;
        for (UnaryOperator<Tensor> stage : this.stages) {
            current = stage.apply(current);
        }
        return current;
    }

    public int stageCount() {
        return this.stages.size();
    }
}
