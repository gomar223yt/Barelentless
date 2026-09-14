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

/**
 * Learning rate schedules.
 * <p>
 * Training here never really "finishes" - the player keeps playing and the data keeps coming - so the schedules that
 * matter are the ones that stay bounded forever rather than decaying to zero at some fixed horizon. Cosine restarts
 * and the plateau-driven schedule both keep the model able to react to a new world months into a save.
 *
 * @author Barelentless
 */
public interface LearningRateSchedule {

    float rateAt(long step);

    static LearningRateSchedule constant(float rate) {
        return step -> rate;
    }

    /**
     * Linear warmup followed by cosine decay from {@code peak} down to {@code floor} over {@code period} steps,
     * restarting afterwards.
     */
    static LearningRateSchedule cosineWithWarmup(float peak, float floor, long warmup, long period) {
        return step -> {
            if (step < warmup) {
                return peak * ((float) step / Math.max(1, warmup));
            }
            long into = (step - warmup) % Math.max(1, period);
            float progress = (float) into / Math.max(1, period);
            return floor + 0.5f * (peak - floor) * (1f + (float) Math.cos(Math.PI * progress));
        };
    }

    /**
     * Exponential decay with a floor, {@code rate = max(floor, peak * gamma^(step / interval))}.
     */
    static LearningRateSchedule exponential(float peak, float floor, float gamma, long interval) {
        return step -> Math.max(floor, peak * (float) Math.pow(gamma, (double) step / Math.max(1, interval)));
    }

    /**
     * Applies this schedule to an optimizer for its current step count.
     */
    default void apply(Optimizer optimizer) {
        optimizer.setLearningRate(rateAt(optimizer.getSteps()));
    }
}
