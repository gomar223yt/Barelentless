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

import java.util.function.UnaryOperator;

/**
 * The nonlinearities available to configured models. Kept as an enum rather than a lambda so that a model's
 * architecture can be described declaratively (and therefore saved, printed and rebuilt from a checkpoint).
 *
 * @author Barelentless
 */
public enum Activation implements UnaryOperator<Tensor> {

    IDENTITY {
        @Override
        public Tensor apply(Tensor input) {
            return input;
        }
    },
    RELU {
        @Override
        public Tensor apply(Tensor input) {
            return input.relu();
        }
    },
    LEAKY_RELU {
        @Override
        public Tensor apply(Tensor input) {
            return input.leakyRelu(0.01f);
        }
    },
    GELU {
        @Override
        public Tensor apply(Tensor input) {
            return input.gelu();
        }
    },
    TANH {
        @Override
        public Tensor apply(Tensor input) {
            return input.tanh();
        }
    },
    SIGMOID {
        @Override
        public Tensor apply(Tensor input) {
            return input.sigmoid();
        }
    },
    SOFTPLUS {
        @Override
        public Tensor apply(Tensor input) {
            return input.softplus();
        }
    };

    /**
     * The initialization scheme that suits this nonlinearity. ReLU-family layers want He, saturating ones want Xavier.
     */
    public boolean prefersKaiming() {
        return this == RELU || this == LEAKY_RELU || this == GELU;
    }
}
