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

package baritone.api.ml;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The properties of the autodiff tape that the numerical gradient checks cannot see.
 * <p>
 * A finite difference tells you a derivative is right; it cannot tell you that a shared subexpression accumulated
 * both of its contributions rather than being visited twice, or that a constant branch was skipped instead of
 * silently allocating gradients for the whole graph. Those are exactly the mistakes that stay invisible until a
 * model trains for a week and comes out subtly wrong.
 *
 * @author Barelentless
 */
public class TensorTest {

    @Test
    public void diamondGraphAccumulatesBothPaths() {
        // y = x * x + x, so dy/dx = 2x + 1. The tape must visit x once and add both contributions, not overwrite.
        Tensor x = Tensor.param(1, 1);
        x.data[0] = 3f;
        Tensor y = x.mul(x).add(x).sum();
        y.backward();
        assertEquals(7f, x.gradient()[0], 1e-5f);
    }

    @Test
    public void deepChainDoesNotOverflowTheStack() {
        // the topological sort is iterative precisely so that a long unrolled recurrence cannot blow the JVM stack
        Tensor x = Tensor.param(1, 1);
        x.data[0] = 1f;
        Tensor current = x;
        for (int i = 0; i < 20_000; i++) {
            current = current.scale(1.00001f);
        }
        current.sum().backward();
        assertTrue(x.gradient()[0] > 1f);
    }

    @Test
    public void constantsReceiveNoGradient() {
        Tensor parameter = Tensor.param(1, 2);
        parameter.data[0] = 0.5f;
        parameter.data[1] = -0.25f;
        Tensor constant = Tensor.vector(2f, 3f);
        Tensor loss = parameter.mul(constant).sum();
        loss.backward();
        assertEquals(2f, parameter.gradient()[0], 1e-6f);
        assertNull("a constant must not allocate a gradient buffer", constant.gradient());
        assertFalse(constant.requiresGrad());
    }

    @Test
    public void zeroGradClearsWithoutReallocating() {
        Tensor x = Tensor.param(1, 1);
        x.data[0] = 2f;
        x.square().sum().backward();
        float[] first = x.gradient();
        assertEquals(4f, first[0], 1e-6f);
        x.zeroGrad();
        assertEquals(0f, first[0], 0f);
        x.square().sum().backward();
        assertTrue("gradient buffer should be reused", first == x.gradient());
        assertEquals(4f, first[0], 1e-6f);
    }

    @Test
    public void softmaxIsStableOnLargeLogits() {
        // the naive formulation overflows here; the shipped one subtracts the row max first
        Tensor logits = Tensor.of(1, 3, 1000f, 1001f, 999f);
        Tensor probabilities = logits.softmax();
        float sum = 0;
        for (float value : probabilities.data) {
            assertTrue("softmax produced " + value, Float.isFinite(value));
            sum += value;
        }
        assertEquals(1f, sum, 1e-5f);
        assertTrue(probabilities.data[1] > probabilities.data[0]);
    }

    @Test
    public void detachCutsTheGraph() {
        Tensor x = Tensor.param(1, 1);
        x.data[0] = 4f;
        Tensor detached = x.square().detach();
        detached.sum().backward();
        assertNull("nothing should flow back through a detached tensor", x.gradient());
        assertEquals(16f, detached.data[0], 1e-5f);
    }

    @Test
    public void matmulShapesAndValues() {
        Tensor a = Tensor.of(2, 3, 1, 2, 3, 4, 5, 6);
        Tensor b = Tensor.of(3, 2, 7, 8, 9, 10, 11, 12);
        Tensor product = a.matmul(b);
        assertEquals(2, product.rows);
        assertEquals(2, product.cols);
        assertEquals(58f, product.get(0, 0), 1e-5f);
        assertEquals(64f, product.get(0, 1), 1e-5f);
        assertEquals(139f, product.get(1, 0), 1e-5f);
        assertEquals(154f, product.get(1, 1), 1e-5f);
    }

    @Test
    public void concatAndSliceRoundTrip() {
        Tensor left = Tensor.of(2, 2, 1, 2, 3, 4);
        Tensor right = Tensor.of(2, 1, 5, 6);
        Tensor joined = Tensor.concatCols(left, right);
        assertEquals(3, joined.cols);
        assertEquals(5f, joined.get(0, 2), 1e-6f);
        assertEquals(3f, joined.sliceCols(0, 2).get(1, 0), 1e-6f);
        assertEquals(6f, joined.sliceRows(1, 2).get(0, 2), 1e-6f);
    }
}
