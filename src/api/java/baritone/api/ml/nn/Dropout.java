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
 * Inverted dropout. Active only while {@link Module#isTraining() training}; a no-op at inference, so the model the
 * game loop sees is always deterministic.
 *
 * @author Barelentless
 */
public final class Dropout extends Module {

    private final float rate;
    private final Random random;

    public Dropout(float rate, Random random) {
        if (rate < 0 || rate >= 1) {
            throw new IllegalArgumentException("dropout rate must be in [0, 1), got " + rate);
        }
        this.rate = rate;
        this.random = random;
    }

    public Tensor forward(Tensor input) {
        if (!isTraining() || this.rate == 0f) {
            return input;
        }
        final float keep = 1f - this.rate;
        final float scale = 1f / keep;
        final boolean[] mask = new boolean[input.size()];
        for (int i = 0; i < mask.length; i++) {
            mask[i] = this.random.nextFloat() < keep;
        }
        return Tensor.operation(input.rows, input.cols, new Tensor[]{input}, out -> {
            for (int i = 0; i < mask.length; i++) {
                out.data[i] = mask[i] ? input.data[i] * scale : 0f;
            }
            return g -> {
                for (int i = 0; i < mask.length; i++) {
                    if (mask[i]) {
                        Tensor.addGrad(input, i, g[i] * scale);
                    }
                }
            };
        });
    }

    public float rate() {
        return this.rate;
    }
}
