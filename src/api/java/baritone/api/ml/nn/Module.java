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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Base class for everything that owns trainable state.
 * <p>
 * A module registers its parameters and child modules by name, which gives three things for free: a flat parameter
 * list for the optimizer, a stable qualified name per tensor for checkpointing (so a saved model still loads after
 * layers are reordered in code), and a single switch for train/eval behaviour.
 *
 * @author Barelentless
 */
public abstract class Module {

    private final Map<String, Tensor> parameters = new LinkedHashMap<>();
    private final Map<String, Module> children = new LinkedHashMap<>();
    private boolean training = true;

    /**
     * Registers a trainable tensor under the given local name and returns it for direct field assignment.
     */
    protected <T extends Tensor> T register(String name, T tensor) {
        if (this.parameters.put(name, tensor) != null) {
            throw new IllegalStateException("duplicate parameter " + name);
        }
        tensor.named(name);
        return tensor;
    }

    protected <T extends Module> T child(String name, T module) {
        if (this.children.put(name, module) != null) {
            throw new IllegalStateException("duplicate child " + name);
        }
        return module;
    }

    public Map<String, Module> children() {
        return this.children;
    }

    public Map<String, Tensor> localParameters() {
        return this.parameters;
    }

    /**
     * Every trainable tensor in this module and, recursively, its children.
     */
    public List<Tensor> parameters() {
        List<Tensor> all = new ArrayList<>();
        collect("", (name, tensor) -> all.add(tensor));
        return all;
    }

    /**
     * Every trainable tensor keyed by its fully qualified path, e.g. {@code encoder.block0.attn.wq}.
     */
    public Map<String, Tensor> namedParameters() {
        Map<String, Tensor> all = new LinkedHashMap<>();
        collect("", all::put);
        return all;
    }

    private void collect(String prefix, ParameterVisitor visitor) {
        for (Map.Entry<String, Tensor> entry : this.parameters.entrySet()) {
            visitor.visit(prefix + entry.getKey(), entry.getValue());
        }
        for (Map.Entry<String, Module> entry : this.children.entrySet()) {
            entry.getValue().collect(prefix + entry.getKey() + ".", visitor);
        }
    }

    public int parameterCount() {
        int total = 0;
        for (Tensor tensor : parameters()) {
            total += tensor.size();
        }
        return total;
    }

    public void zeroGrad() {
        for (Tensor tensor : parameters()) {
            tensor.zeroGrad();
        }
    }

    public boolean isTraining() {
        return this.training;
    }

    /**
     * Switches this module and its children between training and inference behaviour. Dropout and any other
     * stochastic layer keys off this.
     */
    public void setTraining(boolean training) {
        this.training = training;
        for (Module child : this.children.values()) {
            child.setTraining(training);
        }
    }

    public void train() {
        setTraining(true);
    }

    public void eval() {
        setTraining(false);
    }

    /**
     * Copies parameter values (not gradients, not graph) from another module with an identical parameter layout.
     * Used for target networks and for hot-swapping a freshly trained model into the live one.
     */
    public void loadStateFrom(Module other) {
        Map<String, Tensor> mine = namedParameters();
        Map<String, Tensor> theirs = other.namedParameters();
        for (Map.Entry<String, Tensor> entry : mine.entrySet()) {
            Tensor source = theirs.get(entry.getKey());
            if (source == null) {
                throw new IllegalArgumentException("missing parameter " + entry.getKey());
            }
            Tensor target = entry.getValue();
            if (source.size() != target.size()) {
                throw new IllegalArgumentException("size mismatch for " + entry.getKey());
            }
            System.arraycopy(source.data, 0, target.data, 0, source.size());
        }
    }

    /**
     * Polyak averaging: {@code this = tau * other + (1 - tau) * this}. A cheap way to keep a stable target copy of a
     * value network without the jitter of a hard copy.
     */
    public void softUpdateFrom(Module other, float tau) {
        Map<String, Tensor> mine = namedParameters();
        Map<String, Tensor> theirs = other.namedParameters();
        for (Map.Entry<String, Tensor> entry : mine.entrySet()) {
            Tensor source = theirs.get(entry.getKey());
            Tensor target = entry.getValue();
            for (int i = 0; i < target.size(); i++) {
                target.data[i] = tau * source.data[i] + (1 - tau) * target.data[i];
            }
        }
    }

    public boolean isFinite() {
        for (Tensor tensor : parameters()) {
            if (!tensor.isFinite()) {
                return false;
            }
        }
        return true;
    }

    @FunctionalInterface
    private interface ParameterVisitor {

        void visit(String name, Tensor tensor);
    }
}
