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
 * A gated recurrent unit, composed out of primitive differentiable operations.
 * <p>
 * The recurrent path is what gives the controller a memory that survives across ticks without re-encoding a whole
 * window every tick: the hidden state is carried forward, so a movement that has been going wrong for eight ticks
 * looks different to the model than the same instantaneous state reached cleanly.
 *
 * @author Barelentless
 */
public final class GruCell extends Module {

    private final Linear inputToGates;
    private final Linear hiddenToGates;
    private final Linear inputToCandidate;
    private final Linear hiddenToCandidate;
    private final int hidden;

    public GruCell(int inputs, int hidden, Random random) {
        this.hidden = hidden;
        this.inputToGates = child("xh", new Linear(inputs, hidden * 2, true, false, random));
        this.hiddenToGates = child("hh", new Linear(hidden, hidden * 2, false, false, random));
        this.inputToCandidate = child("xc", new Linear(inputs, hidden, true, false, random));
        this.hiddenToCandidate = child("hc", new Linear(hidden, hidden, false, false, random));
    }

    /**
     * Advances the state by one step.
     *
     * @param input Shape {@code [batch, inputs]}
     * @param state Shape {@code [batch, hidden]}
     * @return The new state, shape {@code [batch, hidden]}
     */
    public Tensor forward(Tensor input, Tensor state) {
        Tensor gates = this.inputToGates.forward(input).add(this.hiddenToGates.forward(state)).sigmoid();
        Tensor reset = gates.sliceCols(0, this.hidden);
        Tensor update = gates.sliceCols(this.hidden, this.hidden * 2);
        Tensor candidate = this.inputToCandidate.forward(input)
                .add(this.hiddenToCandidate.forward(state.mul(reset)))
                .tanh();
        // h' = (1 - z) * h + z * candidate
        Tensor keep = update.neg().plus(1f);
        return state.mul(keep).add(candidate.mul(update));
    }

    public Tensor initialState(int batch) {
        return Tensor.zeros(batch, this.hidden);
    }

    public int hiddenSize() {
        return this.hidden;
    }
}
