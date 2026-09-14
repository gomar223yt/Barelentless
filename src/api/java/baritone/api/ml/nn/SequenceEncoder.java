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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Encodes a window of consecutive tick observations into one context vector.
 * <p>
 * Input is {@code [window, features]} - oldest tick first, current tick last. The encoder projects each tick into the
 * model dimension, adds a learned position embedding, runs a causal transformer stack over the window, and returns
 * the representation of the most recent tick, which by construction has attended to the whole window.
 *
 * @author Barelentless
 */
public final class SequenceEncoder extends Module {

    private final Linear inputProjection;
    private final Embedding positions;
    private final List<TransformerBlock> blocks = new ArrayList<>();
    private final LayerNorm finalNorm;
    private final int dimension;
    private final int maxWindow;

    public SequenceEncoder(int features, int dimension, int heads, int depth, int feedForward, int maxWindow,
                           float dropoutRate, Random random) {
        this.dimension = dimension;
        this.maxWindow = maxWindow;
        this.inputProjection = child("in", new Linear(features, dimension, true, true, random));
        this.positions = child("pos", new Embedding(maxWindow, dimension, random));
        for (int i = 0; i < depth; i++) {
            this.blocks.add(child("block" + i,
                    new TransformerBlock(dimension, heads, feedForward, dropoutRate, true, random)));
        }
        this.finalNorm = child("normf", new LayerNorm(dimension));
    }

    /**
     * @param window Shape {@code [length, features]}, oldest first
     * @return Shape {@code [length, dimension]}, one encoded vector per tick
     */
    public Tensor encodeAll(Tensor window) {
        if (window.rows > this.maxWindow) {
            throw new IllegalArgumentException("window of " + window.rows + " exceeds max " + this.maxWindow);
        }
        int[] ids = new int[window.rows];
        for (int i = 0; i < ids.length; i++) {
            // index from the end so that "the current tick" is always the same position id regardless of how much
            // history happens to be available - a half-filled window at the start of an episode must not shift
            // everything the model has learned about recency
            ids[i] = this.maxWindow - window.rows + i;
        }
        Tensor encoded = this.inputProjection.forward(window).add(this.positions.forward(ids));
        for (TransformerBlock block : this.blocks) {
            encoded = block.forward(encoded);
        }
        return this.finalNorm.forward(encoded);
    }

    /**
     * @return The encoding of the most recent tick, shape {@code [1, dimension]}
     */
    public Tensor encode(Tensor window) {
        Tensor all = encodeAll(window);
        return all.sliceRows(all.rows - 1, all.rows);
    }

    public int dimension() {
        return this.dimension;
    }

    public int maxWindow() {
        return this.maxWindow;
    }

    public List<TransformerBlock> blocks() {
        return this.blocks;
    }
}
