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

import java.util.Arrays;
import java.util.Random;

/**
 * Multi-head self attention over a sequence of encoded ticks, shape {@code [sequence, dimension]}.
 * <p>
 * Attention is the part that makes the model reason about a situation instead of a single frame. When the model
 * decides how to aim on this tick it can look back at the tick the jump started, the tick the block edge appeared,
 * the tick the last correction overshot - and weight them by relevance rather than by a fixed decay constant.
 * Masking is causal, so nothing can attend to a tick that has not happened yet, which also means the same weights
 * can be used for offline training over a whole episode and for one-tick-at-a-time inference.
 *
 * @author Barelentless
 */
public final class MultiHeadAttention extends Module {

    private final Linear queryProjection;
    private final Linear keyProjection;
    private final Linear valueProjection;
    private final Linear outputProjection;
    private final Dropout dropout;

    private final int heads;
    private final int headDimension;
    private final boolean causal;

    public MultiHeadAttention(int dimension, int heads, boolean causal, float dropoutRate, Random random) {
        if (dimension % heads != 0) {
            throw new IllegalArgumentException("dimension " + dimension + " not divisible by heads " + heads);
        }
        this.heads = heads;
        this.headDimension = dimension / heads;
        this.causal = causal;
        this.queryProjection = child("wq", new Linear(dimension, dimension, false, false, random));
        this.keyProjection = child("wk", new Linear(dimension, dimension, false, false, random));
        this.valueProjection = child("wv", new Linear(dimension, dimension, false, false, random));
        this.outputProjection = child("wo", new Linear(dimension, dimension, true, false, random));
        this.dropout = child("drop", new Dropout(dropoutRate, random));
    }

    public Tensor forward(Tensor sequence) {
        final int length = sequence.rows;
        Tensor queries = this.queryProjection.forward(sequence);
        Tensor keys = this.keyProjection.forward(sequence);
        Tensor values = this.valueProjection.forward(sequence);

        final float scale = (float) (1.0 / Math.sqrt(this.headDimension));
        final float[] mask = this.causal ? causalMask(length) : null;

        Tensor[] headOutputs = new Tensor[this.heads];
        for (int h = 0; h < this.heads; h++) {
            int from = h * this.headDimension;
            int to = from + this.headDimension;
            Tensor q = queries.sliceCols(from, to);
            Tensor k = keys.sliceCols(from, to);
            Tensor v = values.sliceCols(from, to);
            Tensor scores = q.matmul(k.transpose()).scale(scale);
            if (mask != null) {
                scores = scores.maskAdd(mask);
            }
            Tensor weights = this.dropout.forward(scores.softmax());
            headOutputs[h] = weights.matmul(v);
        }
        return this.outputProjection.forward(Tensor.concatCols(headOutputs));
    }

    /**
     * Returns the attention weights of a single head for the given sequence, without recording a graph. This exists
     * so that the in-game diagnostics can show which past ticks the controller is actually keying off - a learned
     * controller you cannot inspect is a controller you cannot trust near lava.
     */
    public float[] inspect(Tensor sequence, int head) {
        Tensor queries = this.queryProjection.forward(sequence).detach();
        Tensor keys = this.keyProjection.forward(sequence).detach();
        int from = head * this.headDimension;
        int to = from + this.headDimension;
        Tensor scores = queries.sliceCols(from, to)
                .matmul(keys.sliceCols(from, to).transpose())
                .scale((float) (1.0 / Math.sqrt(this.headDimension)));
        if (this.causal) {
            scores = scores.maskAdd(causalMask(sequence.rows));
        }
        return scores.softmax().data.clone();
    }

    private static float[] causalMask(int length) {
        float[] mask = new float[length * length];
        Arrays.fill(mask, 0f);
        for (int r = 0; r < length; r++) {
            for (int c = r + 1; c < length; c++) {
                mask[r * length + c] = -1e9f;
            }
        }
        return mask;
    }

    public int heads() {
        return this.heads;
    }
}
