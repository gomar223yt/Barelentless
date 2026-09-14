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

import java.util.ArrayList;
import java.util.List;

/**
 * A recorded sequence of decisions and what followed them.
 * <p>
 * One trajectory is one attempt at something with a clear end - a single movement executed from its start block to
 * its destination, or a stretch of elytra flight between two corrections. Bounding trajectories this way is what
 * makes credit assignment meaningful: the reward for arriving cleanly is attributed to the ticks that actually got
 * there, rather than smeared across an unbounded stream.
 *
 * @author Barelentless
 */
public final class Trajectory {

    /**
     * One decision point.
     */
    public static final class Step {

        public final float[] observation;
        public final float[] continuousAction;
        public final boolean[] discreteAction;
        public final float logProbability;
        public final float value;
        public float reward;
        public boolean terminal;

        /**
         * Filled in by {@link #finish}: the generalized advantage estimate and the value target for this step.
         */
        public float advantage;
        public float valueTarget;

        public Step(float[] observation, float[] continuousAction, boolean[] discreteAction,
                    float logProbability, float value) {
            this.observation = observation;
            this.continuousAction = continuousAction;
            this.discreteAction = discreteAction;
            this.logProbability = logProbability;
            this.value = value;
        }
    }

    private final List<Step> steps = new ArrayList<>();
    private final int context;
    private boolean finished;
    private float totalReward;

    public Trajectory(int context) {
        this.context = context;
    }

    public void add(Step step) {
        if (this.finished) {
            throw new IllegalStateException("trajectory already finished");
        }
        this.steps.add(step);
    }

    /**
     * Adds a reward to the most recent step. Rewards usually become known one tick after the action that earned them,
     * so this is the normal way they arrive.
     */
    public void reward(float amount) {
        if (!this.steps.isEmpty()) {
            this.steps.get(this.steps.size() - 1).reward += amount;
            this.totalReward += amount;
        }
    }

    /**
     * Closes the trajectory and computes advantages with generalized advantage estimation.
     *
     * @param bootstrapValue Value estimate of the state after the last step, or zero if the episode truly ended
     * @param gamma          Discount factor
     * @param lambda         GAE trace decay; lower trades variance for bias
     */
    public void finish(float bootstrapValue, float gamma, float lambda) {
        this.finished = true;
        float nextValue = bootstrapValue;
        float accumulator = 0;
        for (int i = this.steps.size() - 1; i >= 0; i--) {
            Step step = this.steps.get(i);
            float mask = step.terminal ? 0f : 1f;
            float delta = step.reward + gamma * nextValue * mask - step.value;
            accumulator = delta + gamma * lambda * mask * accumulator;
            step.advantage = accumulator;
            step.valueTarget = accumulator + step.value;
            nextValue = step.value;
        }
    }

    public List<Step> steps() {
        return this.steps;
    }

    public int size() {
        return this.steps.size();
    }

    public boolean isFinished() {
        return this.finished;
    }

    public int getContext() {
        return this.context;
    }

    public float getTotalReward() {
        return this.totalReward;
    }
}
