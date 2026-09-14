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

/**
 * What a policy produces for a batch of observations: a continuous action distribution, an optional set of discrete
 * action logits, and a value estimate.
 * <p>
 * Movement control needs both halves at once. Aiming is continuous (a yaw and pitch delta); keys are discrete
 * (forward, jump, sprint, sneak - each on or off). Splitting them into two models would throw away the fact that the
 * decision to jump and the decision to look down are the same decision.
 *
 * @author Barelentless
 */
public final class PolicyOutput {

    /**
     * Mean of the continuous action distribution, shape {@code [batch, continuousActions]}.
     */
    public final Tensor mean;

    /**
     * Log standard deviation of the continuous action distribution, same shape as {@link #mean}.
     */
    public final Tensor logStandardDeviation;

    /**
     * Logits for each independent binary action, shape {@code [batch, discreteActions]}. May be {@code null}.
     */
    public final Tensor discreteLogits;

    /**
     * State value estimate, shape {@code [batch, 1]}. May be {@code null} for models that do not learn a critic.
     */
    public final Tensor value;

    public PolicyOutput(Tensor mean, Tensor logStandardDeviation, Tensor discreteLogits, Tensor value) {
        this.mean = mean;
        this.logStandardDeviation = logStandardDeviation;
        this.discreteLogits = discreteLogits;
        this.value = value;
    }

    public boolean hasDiscrete() {
        return this.discreteLogits != null;
    }

    public boolean hasValue() {
        return this.value != null;
    }
}
