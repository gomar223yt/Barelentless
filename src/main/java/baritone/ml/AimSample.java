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

package baritone.ml;

import baritone.api.ml.Tensor;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * One recorded tick of aiming: the window of state leading up to it, and the view movement that was actually made.
 * <p>
 * Samples come from two sources, and both are useful. When the player is looking around themselves, the sample is a
 * demonstration - this is what a human does with this error, at this speed, in this terrain - and training on it is
 * plain imitation. When the bot is aiming, the sample is its own behaviour, and is only worth learning from once it
 * is paired with an outcome. The {@link #demonstration} flag is what keeps the two from being confused.
 *
 * @author Barelentless
 */
public final class AimSample {

    /**
     * The window of encoded ticks, oldest first, each of length {@link AimModel#INPUT_FEATURES}.
     */
    public final float[][] window;

    /**
     * The yaw delta that was applied on this tick, in degrees.
     */
    public final float yawDelta;

    /**
     * The pitch delta that was applied on this tick, in degrees.
     */
    public final float pitchDelta;

    /**
     * True when this came from a human rather than from the bot.
     */
    public final boolean demonstration;

    public AimSample(float[][] window, float yawDelta, float pitchDelta, boolean demonstration) {
        this.window = window;
        this.yawDelta = yawDelta;
        this.pitchDelta = pitchDelta;
        this.demonstration = demonstration;
    }

    public Tensor windowTensor() {
        Tensor tensor = new Tensor(this.window.length, AimModel.INPUT_FEATURES);
        for (int r = 0; r < this.window.length; r++) {
            System.arraycopy(this.window[r], 0, tensor.data, r * AimModel.INPUT_FEATURES, AimModel.INPUT_FEATURES);
        }
        return tensor;
    }

    public Tensor targetTensor() {
        return Tensor.of(1, 2, this.yawDelta, this.pitchDelta);
    }

    /**
     * How much this sample is worth training on. Ticks where the view barely moved are the overwhelming majority and
     * teach almost nothing, so they get a low priority; large corrections and anything unusual get a high one.
     */
    public float priority() {
        float magnitude = Math.abs(this.yawDelta) + Math.abs(this.pitchDelta);
        return (this.demonstration ? 2f : 1f) * (0.1f + Math.min(magnitude, 30f));
    }

    public void write(DataOutput out) throws IOException {
        out.writeInt(this.window.length);
        out.writeInt(AimModel.INPUT_FEATURES);
        for (float[] row : this.window) {
            for (float value : row) {
                out.writeFloat(value);
            }
        }
        out.writeFloat(this.yawDelta);
        out.writeFloat(this.pitchDelta);
        out.writeBoolean(this.demonstration);
    }

    public static AimSample read(DataInput in) throws IOException {
        int rows = in.readInt();
        int cols = in.readInt();
        float[][] window = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                window[r][c] = in.readFloat();
            }
        }
        float yaw = in.readFloat();
        float pitch = in.readFloat();
        boolean demonstration = in.readBoolean();
        if (cols != AimModel.INPUT_FEATURES) {
            // the feature layout changed since this was recorded; the values no longer mean what they did, so the
            // sample is dropped rather than silently taught to the model as noise
            return null;
        }
        return new AimSample(window, yaw, pitch, demonstration);
    }
}
