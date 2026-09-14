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
 * A learned lookup table mapping discrete ids to dense vectors.
 * <p>
 * This is how the model gets a real understanding of block types and movement kinds instead of a hand-written cost
 * number per case: every block id and every movement kind owns a vector that training is free to move around, so
 * "gravel" and "sand" can end up near each other because they behave alike, not because someone declared them
 * similar in a table.
 *
 * @author Barelentless
 */
public final class Embedding extends Module {

    private final Tensor table;

    public Embedding(int vocabulary, int dimension, Random random) {
        this.table = register("table", Tensor.param(vocabulary, dimension));
        double std = 1.0 / Math.sqrt(dimension);
        for (int i = 0; i < this.table.size(); i++) {
            this.table.data[i] = (float) (random.nextGaussian() * std);
        }
    }

    /**
     * Gathers one row per id, producing {@code [ids.length, dimension]}. Out of range ids map to the zero vector,
     * which lets callers pass an "unknown" sentinel without a branch.
     */
    public Tensor forward(int[] ids) {
        final int dim = this.table.cols;
        final int vocabulary = this.table.rows;
        return Tensor.operation(ids.length, dim, new Tensor[]{this.table}, out -> {
            for (int i = 0; i < ids.length; i++) {
                int id = ids[i];
                if (id >= 0 && id < vocabulary) {
                    System.arraycopy(this.table.data, id * dim, out.data, i * dim, dim);
                }
            }
            return g -> {
                for (int i = 0; i < ids.length; i++) {
                    int id = ids[i];
                    if (id >= 0 && id < vocabulary) {
                        for (int c = 0; c < dim; c++) {
                            Tensor.addGrad(this.table, id * dim + c, g[i * dim + c]);
                        }
                    }
                }
            };
        });
    }

    public Tensor forward(int id) {
        return forward(new int[]{id});
    }

    public int vocabulary() {
        return this.table.rows;
    }

    public int dimension() {
        return this.table.cols;
    }

    public Tensor table() {
        return this.table;
    }

    /**
     * Cosine similarity between two entries of the table. Handy for inspecting what the model has actually learned -
     * {@code ml similar minecraft:gravel} answers with real numbers rather than a guess.
     */
    public float similarity(int a, int b) {
        int dim = this.table.cols;
        double dot = 0;
        double na = 0;
        double nb = 0;
        for (int c = 0; c < dim; c++) {
            float va = this.table.data[a * dim + c];
            float vb = this.table.data[b * dim + c];
            dot += va * vb;
            na += va * va;
            nb += vb * vb;
        }
        double denominator = Math.sqrt(na) * Math.sqrt(nb);
        return denominator == 0 ? 0f : (float) (dot / denominator);
    }
}
