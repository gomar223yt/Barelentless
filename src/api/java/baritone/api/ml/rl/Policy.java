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

package baritone.api.ml.rl;

import baritone.api.ml.Tensor;
import baritone.api.ml.nn.Module;

/**
 * Anything that can turn a batch of observations into action distributions and a value estimate.
 * <p>
 * Kept as an interface rather than a concrete class so that an addon can plug in its own architecture - a smaller
 * network for a weak machine, a hand-written controller wrapped as a policy, a model trained elsewhere - and still
 * use the trainers, the rollout plumbing and the in-game diagnostics unchanged.
 *
 * @author Barelentless
 */
public interface Policy {

    /**
     * @param observations Shape {@code [batch, features]}
     * @return The distributions and value for each row
     */
    PolicyOutput forward(Tensor observations);

    /**
     * The trainable module behind this policy, for the optimizer and for checkpointing.
     */
    Module module();

    /**
     * Number of continuous action dimensions this policy emits.
     */
    int continuousActions();

    /**
     * Number of independent binary actions this policy emits; zero if it controls nothing discrete.
     */
    default int discreteActions() {
        return 0;
    }
}
