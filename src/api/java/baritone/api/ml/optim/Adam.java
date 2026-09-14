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

import baritone.api.ml.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Adam, with decoupled weight decay (AdamW) when a decay is configured.
 * <p>
 * Decoupled rather than L2-in-the-gradient because the models here are trained continuously and the effective decay
 * would otherwise drift with the adaptive scale, quietly shrinking whichever weights happen to see small gradients.
 *
 * @author Barelentless
 */
public final class Adam extends Optimizer {

    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private float weightDecay;

    private final Map<Tensor, float[]> firstMoment = new IdentityHashMap<>();
    private final Map<Tensor, float[]> secondMoment = new IdentityHashMap<>();

    public Adam(List<Tensor> parameters, float learningRate) {
        this(parameters, learningRate, 0.9f, 0.999f, 1e-8f, 0f);
    }

    public Adam(List<Tensor> parameters, float learningRate, float beta1, float beta2, float epsilon, float weightDecay) {
        super(parameters, learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        for (Tensor parameter : parameters) {
            this.firstMoment.put(parameter, new float[parameter.size()]);
            this.secondMoment.put(parameter, new float[parameter.size()]);
        }
    }

    @Override
    protected void update(Tensor parameter, float gradientScale) {
        float[] gradient = parameter.gradient();
        float[] m = this.firstMoment.get(parameter);
        float[] v = this.secondMoment.get(parameter);
        float biasCorrection1 = (float) (1 - Math.pow(this.beta1, this.steps));
        float biasCorrection2 = (float) (1 - Math.pow(this.beta2, this.steps));
        for (int i = 0; i < gradient.length; i++) {
            float g = gradient[i] * gradientScale;
            m[i] = this.beta1 * m[i] + (1 - this.beta1) * g;
            v[i] = this.beta2 * v[i] + (1 - this.beta2) * g * g;
            float mHat = m[i] / biasCorrection1;
            float vHat = v[i] / biasCorrection2;
            float delta = this.learningRate * mHat / ((float) Math.sqrt(vHat) + this.epsilon);
            if (this.weightDecay != 0f) {
                delta += this.learningRate * this.weightDecay * parameter.data[i];
            }
            parameter.data[i] -= delta;
        }
    }

    public void setWeightDecay(float weightDecay) {
        this.weightDecay = weightDecay;
    }

    /**
     * Clears the moment estimates. Needed after a model is replaced wholesale (a checkpoint load, a reset), since
     * moments from the old weights are meaningless for the new ones.
     */
    public void resetState() {
        for (Tensor parameter : this.parameters) {
            java.util.Arrays.fill(this.firstMoment.get(parameter), 0f);
            java.util.Arrays.fill(this.secondMoment.get(parameter), 0f);
        }
        this.steps = 0;
    }
}
