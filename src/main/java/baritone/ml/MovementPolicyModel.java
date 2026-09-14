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

package baritone.ml;

import baritone.api.ml.Tensor;
import baritone.api.ml.nn.Activation;
import baritone.api.ml.nn.Linear;
import baritone.api.ml.nn.Mlp;
import baritone.api.ml.nn.Module;
import baritone.api.ml.rl.Policy;
import baritone.api.ml.rl.PolicyOutput;

import java.util.Random;

/**
 * The policy that adjusts how a movement is actually walked.
 * <p>
 * It does not replace the movement code, and could not: the movement classes know things the model cannot infer,
 * like which block has to be placed and where the path goes next. What it controls is the part the movement classes
 * were never able to express - how fast to approach, whether to commit to the jump this tick or the next, whether
 * sprinting into this particular gap is worth it. It emits small bounded adjustments to the command that pathing
 * already produced, so the worst a bad policy can do is make the bot slightly clumsy rather than send it somewhere
 * else entirely.
 * <p>
 * The continuous half adjusts the movement vector; the discrete half is a set of independent Bernoullis over jump,
 * sprint and sneak. They come out of the same network on purpose: the decision to jump and the decision to be at
 * full speed are the same decision, and splitting them into two models would throw that away.
 *
 * @author Barelentless
 */
public final class MovementPolicyModel extends Module implements Policy {

    /**
     * Adjustments to the movement vector: forward scale and strafe offset.
     */
    public static final int CONTINUOUS_ACTIONS = 2;

    /**
     * Jump, sprint, sneak.
     */
    public static final int DISCRETE_ACTIONS = 3;

    private static final int OUTPUTS = CONTINUOUS_ACTIONS * 2 + DISCRETE_ACTIONS + 1;

    private final Mlp trunk;
    private final Linear head;

    public MovementPolicyModel(int hidden, Random random) {
        this.trunk = child("trunk", new Mlp(StateEncoder.FEATURES, new int[]{hidden, hidden},
                hidden, Activation.GELU, Activation.GELU, true, random));
        // near-zero output layer: an untrained policy proposes no adjustment at all, so enabling it changes nothing
        // until it has learned something
        this.head = child("head", new Linear(hidden, OUTPUTS, true, false, random).scaleInit(0.01f));
    }

    @Override
    public PolicyOutput forward(Tensor observations) {
        Tensor output = this.head.forward(this.trunk.forward(observations));
        Tensor mean = output.sliceCols(0, CONTINUOUS_ACTIONS).tanh();
        Tensor logDeviation = output.sliceCols(CONTINUOUS_ACTIONS, CONTINUOUS_ACTIONS * 2)
                // floored well above zero: a policy that collapses to a deterministic answer stops exploring, and
                // this one only ever gets the data its own behaviour produces
                .clamp(-2.5f, 0.5f);
        Tensor discrete = output.sliceCols(CONTINUOUS_ACTIONS * 2, CONTINUOUS_ACTIONS * 2 + DISCRETE_ACTIONS);
        Tensor value = output.sliceCols(CONTINUOUS_ACTIONS * 2 + DISCRETE_ACTIONS, OUTPUTS);
        return new PolicyOutput(mean, logDeviation, discrete, value);
    }

    @Override
    public Module module() {
        return this;
    }

    @Override
    public int continuousActions() {
        return CONTINUOUS_ACTIONS;
    }

    @Override
    public int discreteActions() {
        return DISCRETE_ACTIONS;
    }
}
