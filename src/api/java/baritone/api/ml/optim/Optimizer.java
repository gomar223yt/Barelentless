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

import java.util.List;

/**
 * Base class for parameter updates.
 * <p>
 * Two things every optimizer here does that a textbook one usually skips: it clips the global gradient norm (a single
 * bad episode - a lava death, a chunk unload mid-movement - must not blow up weights that took hours to train), and
 * it refuses to apply a non-finite update at all, leaving the model at its last good state rather than poisoning it.
 *
 * @author Barelentless
 */
public abstract class Optimizer {

    protected final List<Tensor> parameters;
    protected float learningRate;
    protected float maxGradientNorm = 1f;
    protected long steps;
    private float lastGradientNorm;
    private long rejectedSteps;

    protected Optimizer(List<Tensor> parameters, float learningRate) {
        this.parameters = parameters;
        this.learningRate = learningRate;
    }

    /**
     * Applies one update. Returns false if the step was rejected because the gradients were not finite.
     */
    public final boolean step() {
        double sumSquares = 0;
        boolean finite = true;
        for (Tensor parameter : this.parameters) {
            float[] gradient = parameter.gradient();
            if (gradient == null) {
                continue;
            }
            for (float g : gradient) {
                if (Float.isNaN(g) || Float.isInfinite(g)) {
                    finite = false;
                    break;
                }
                sumSquares += (double) g * g;
            }
            if (!finite) {
                break;
            }
        }
        if (!finite) {
            this.rejectedSteps++;
            zeroGrad();
            return false;
        }
        this.lastGradientNorm = (float) Math.sqrt(sumSquares);
        float scale = 1f;
        if (this.maxGradientNorm > 0 && this.lastGradientNorm > this.maxGradientNorm) {
            scale = this.maxGradientNorm / (this.lastGradientNorm + 1e-6f);
        }
        this.steps++;
        for (Tensor parameter : this.parameters) {
            if (parameter.gradient() != null) {
                update(parameter, scale);
            }
        }
        return true;
    }

    /**
     * Applies this optimizer's rule to a single parameter. {@code gradientScale} is the clipping factor.
     */
    protected abstract void update(Tensor parameter, float gradientScale);

    public void zeroGrad() {
        for (Tensor parameter : this.parameters) {
            parameter.zeroGrad();
        }
    }

    public float getLearningRate() {
        return this.learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public void setMaxGradientNorm(float maxGradientNorm) {
        this.maxGradientNorm = maxGradientNorm;
    }

    public float getLastGradientNorm() {
        return this.lastGradientNorm;
    }

    public long getSteps() {
        return this.steps;
    }

    public long getRejectedSteps() {
        return this.rejectedSteps;
    }

    public List<Tensor> getParameters() {
        return this.parameters;
    }
}
