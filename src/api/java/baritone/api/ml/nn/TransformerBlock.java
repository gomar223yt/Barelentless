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
 * A pre-norm transformer block: {@code x + attn(norm(x))} followed by {@code x + ffn(norm(x))}.
 * <p>
 * Pre-norm rather than post-norm because these models are trained incrementally, in small steps, on a background
 * thread while the game is running - pre-norm tolerates a badly conditioned first few hundred updates without the
 * warmup schedule that post-norm needs to avoid diverging.
 *
 * @author Barelentless
 */
public final class TransformerBlock extends Module {

    private final LayerNorm attentionNorm;
    private final MultiHeadAttention attention;
    private final LayerNorm feedForwardNorm;
    private final Linear up;
    private final Linear down;
    private final Dropout dropout;
    private final Activation activation;

    public TransformerBlock(int dimension, int heads, int feedForward, float dropoutRate, boolean causal, Random random) {
        this.attentionNorm = child("norm1", new LayerNorm(dimension));
        this.attention = child("attn", new MultiHeadAttention(dimension, heads, causal, dropoutRate, random));
        this.feedForwardNorm = child("norm2", new LayerNorm(dimension));
        this.up = child("up", new Linear(dimension, feedForward, true, true, random));
        this.down = child("down", new Linear(feedForward, dimension, true, false, random));
        this.dropout = child("drop", new Dropout(dropoutRate, random));
        this.activation = Activation.GELU;
    }

    public Tensor forward(Tensor sequence) {
        Tensor attended = sequence.add(this.attention.forward(this.attentionNorm.forward(sequence)));
        Tensor projected = this.down.forward(this.activation.apply(this.up.forward(this.feedForwardNorm.forward(attended))));
        return attended.add(this.dropout.forward(projected));
    }

    public MultiHeadAttention attention() {
        return this.attention;
    }
}
