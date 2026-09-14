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

package baritone.api.ml.memory;

/**
 * Small vector helpers shared by the memory classes.
 *
 * @author Barelentless
 */
public final class Memories {

    private Memories() {}

    /**
     * Scales a vector to unit length in place. Keys are normalized so that similarity is a dot product and so that
     * two situations differing only in overall magnitude are recognised as the same shape.
     */
    public static void normalize(float[] vector) {
        double sum = 0;
        for (float value : vector) {
            sum += (double) value * value;
        }
        if (sum <= 1e-12) {
            return;
        }
        float inverse = (float) (1.0 / Math.sqrt(sum));
        for (int i = 0; i < vector.length; i++) {
            vector[i] *= inverse;
        }
    }

    public static float[] normalized(float[] vector) {
        float[] copy = vector.clone();
        normalize(copy);
        return copy;
    }

    /**
     * Dot product, which for unit vectors is cosine similarity.
     */
    public static float dot(float[] a, float[] b) {
        float sum = 0;
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    public static float distanceSquared(float[] a, float[] b) {
        float sum = 0;
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }
}
