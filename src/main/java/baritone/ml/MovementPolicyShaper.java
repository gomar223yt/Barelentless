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

import baritone.Baritone;
import baritone.api.control.ControlContext;
import baritone.api.control.IInputShaper;
import baritone.api.control.MovementCommand;
import baritone.api.ml.Tensor;
import baritone.api.ml.rl.PolicyOutput;
import baritone.api.ml.rl.Trajectory;
import baritone.api.utils.input.Input;

import java.util.Random;

/**
 * Applies the movement policy to the command pathing produced, and records what it did for training.
 * <p>
 * Every adjustment is bounded and multiplicative rather than absolute: the policy scales the movement vector and
 * nudges the strafe, it does not choose a direction. Combined with an output layer initialized at zero, that means
 * the behaviour degrades gracefully in both directions - an untrained policy is invisible, and a badly trained one is
 * clumsy rather than dangerous.
 * <p>
 * Three situations bypass it entirely: low health, standing in lava, and a movement that needs to place a block.
 * Those are the cases where being slightly wrong is not slightly worse, and none of them are frequent enough for
 * excluding them to cost the policy much data.
 *
 * @author Barelentless
 */
public final class MovementPolicyShaper implements IInputShaper {

    private final MlManager manager;
    private final Random random = new Random();

    private Trajectory.Step pendingStep;
    private float lastValue;
    private boolean lastActed;

    public MovementPolicyShaper(MlManager manager) {
        this.manager = manager;
    }

    @Override
    public String name() {
        return "movementPolicy";
    }

    @Override
    public boolean isEnabled() {
        return Baritone.settings().mlEnabled.value
                && Baritone.settings().mlMovement.value
                && this.manager.getMovementPolicy() != null;
    }

    @Override
    public void shape(ControlContext context, MovementCommand command) {
        this.lastActed = false;
        MovementPolicyModel policy = this.manager.getMovementPolicy();
        if (policy == null || context.movement() == null || command.isIdle()) {
            return;
        }
        if (context.player().player() == null
                || context.player().player().getHealth() <= 6f
                || context.player().player().isInLava()) {
            return;
        }

        if (!this.manager.canInfer()) {
            return;
        }
        float[] observation = this.manager.currentStateFeatures().clone();
        PolicyOutput output;
        try {
            long start = System.nanoTime();
            output = policy.forward(new Tensor(1, StateEncoder.FEATURES, observation.clone()));
            this.manager.recordInference(System.nanoTime() - start);
        } catch (RuntimeException e) {
            this.manager.onInferenceFailure("movement", e);
            return;
        }

        boolean exploring = Baritone.settings().mlMovementExplore.value;
        float[] action = new float[MovementPolicyModel.CONTINUOUS_ACTIONS];
        double logProbability = 0;
        for (int i = 0; i < action.length; i++) {
            float mean = output.mean.data[i];
            float deviation = (float) Math.exp(output.logStandardDeviation.data[i]);
            action[i] = exploring ? (float) (mean + this.random.nextGaussian() * deviation) : mean;
            logProbability += -0.5 * Math.pow((action[i] - mean) / deviation, 2)
                    - Math.log(deviation) - 0.5 * Math.log(2 * Math.PI);
        }
        boolean[] discrete = new boolean[MovementPolicyModel.DISCRETE_ACTIONS];
        for (int i = 0; i < discrete.length; i++) {
            double probability = 1.0 / (1.0 + Math.exp(-output.discreteLogits.data[i]));
            discrete[i] = exploring ? this.random.nextDouble() < probability : probability > 0.5;
            logProbability += Math.log(Math.max(1e-6, discrete[i] ? probability : 1 - probability));
        }

        apply(command, action, discrete, (float) Math.max(0, Math.min(1, Baritone.settings().mlMovementStrength.value)));

        this.lastValue = output.value.data[0];
        this.pendingStep = new Trajectory.Step(observation, action, discrete, (float) logProbability, this.lastValue);
        this.lastActed = true;
        this.manager.recordPolicyStep(this.pendingStep);
    }

    /**
     * Turns the policy's outputs into an actual change to the command.
     */
    private void apply(MovementCommand command, float[] action, boolean[] discrete, float strength) {
        // forward: scale between roughly a third and full speed. The policy can slow the bot down a lot and speed it
        // up not at all, because full speed is already what pathing asked for.
        float scale = 1f + action[0] * 0.66f * strength;
        float strafe = command.getStrafeImpulse() + action[1] * 0.5f * strength;
        command.setAnalog(command.getForwardImpulse() * scale, strafe);

        // the discrete actions may only ever withhold jump, not invent one: a jump the path did not plan for is how
        // a bot ends up in a hole, while declining one it did plan for merely wastes a tick
        if (command.isPressed(Input.JUMP) && !discrete[0] && strength > 0.5f) {
            command.set(Input.JUMP, false);
        }
        command.setSprintAllowed(command.isSprintAllowed() && discrete[1]);
        if (discrete[2]) {
            command.set(Input.SNEAK, true);
        }
    }

    /**
     * The value the policy assigned to the most recent state, for bootstrapping a trajectory that was cut short.
     */
    public float getLastValue() {
        return this.lastValue;
    }

    public boolean actedLastTick() {
        return this.lastActed;
    }
}
