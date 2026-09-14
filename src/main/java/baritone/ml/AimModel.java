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
import baritone.api.ml.nn.Linear;
import baritone.api.ml.nn.Module;
import baritone.api.ml.nn.SequenceEncoder;
import baritone.api.ml.rl.Policy;
import baritone.api.ml.rl.PolicyOutput;

import java.util.Random;

/**
 * The model that decides how to move the view.
 * <p>
 * It predicts a distribution rather than a number, and that is the point. Aim is not a function of the current frame:
 * the same error of thirty degrees means "flick" at the start of a turn and "you have overshot, settle" at the end of
 * one, and the difference is only visible in what happened over the preceding ticks. So the model reads a window of
 * ticks through a causal transformer, and emits a mean delta together with how sure it is - which lets the controller
 * hand back to the deterministic path when the model itself reports that it does not know this situation, instead of
 * confidently steering into a wall.
 * <p>
 * The window includes the aim error itself, so the model is free to learn the trivial proportional response as its
 * baseline and spend its capacity on everything that deviates from it.
 *
 * @author Barelentless
 */
public final class AimModel extends Module implements Policy {

    /**
     * Extra per-tick features appended to the encoded state: the current yaw and pitch error, their rates of change,
     * and the target's purpose.
     */
    public static final int AIM_FEATURES = 8;

    public static final int INPUT_FEATURES = StateEncoder.FEATURES + AIM_FEATURES;

    private final SequenceEncoder encoder;
    private final Linear head;
    private final int window;

    public AimModel(int window, int dimension, int heads, int depth, Random random) {
        this.window = window;
        this.encoder = child("encoder", new SequenceEncoder(
                INPUT_FEATURES, dimension, heads, depth, dimension * 2, window, 0.05f, random));
        // four outputs: yaw delta, pitch delta, and a log standard deviation for each. Initialized near zero so a
        // freshly created model steers exactly like unmodified Baritone until it has learned something.
        this.head = child("head", new Linear(dimension, 4, true, false, random).scaleInit(0.01f));
        eval();
    }

    /**
     * A single prediction, in degrees.
     */
    public static final class Aim {

        public final float yawDelta;
        public final float pitchDelta;
        public final float yawDeviation;
        public final float pitchDeviation;

        Aim(float yawDelta, float pitchDelta, float yawDeviation, float pitchDeviation) {
            this.yawDelta = yawDelta;
            this.pitchDelta = pitchDelta;
            this.yawDeviation = yawDeviation;
            this.pitchDeviation = pitchDeviation;
        }

        /**
         * How much this prediction should be trusted, in {@code [0, 1]}. Derived from the predicted spread: a model
         * that reports a five degree standard deviation on a two degree correction is guessing.
         */
        public float confidence() {
            float spread = Math.max(this.yawDeviation, this.pitchDeviation);
            return (float) (1.0 / (1.0 + spread));
        }

        @Override
        public String toString() {
            return String.format("yaw %+.2f+-%.2f pitch %+.2f+-%.2f (confidence %.2f)",
                    this.yawDelta, this.yawDeviation, this.pitchDelta, this.pitchDeviation, confidence());
        }
    }

    /**
     * Runs the model over a window of ticks, oldest first.
     *
     * @param window Shape {@code [length, INPUT_FEATURES]}, at most {@link #getWindow()} rows
     */
    public Aim predict(Tensor window) {
        Tensor output = this.head.forward(this.encoder.encode(window));
        // the deltas are bounded by a tanh so that an untrained or diverged model cannot produce a wild flick; the
        // scale is generous enough for any correction a player would make in one tick
        float yaw = (float) Math.tanh(output.data[0]) * 45f;
        float pitch = (float) Math.tanh(output.data[1]) * 30f;
        float yawDeviation = (float) Math.exp(clampLogDeviation(output.data[2]));
        float pitchDeviation = (float) Math.exp(clampLogDeviation(output.data[3]));
        return new Aim(yaw, pitch, yawDeviation, pitchDeviation);
    }

    /**
     * The differentiable form used by training: returns the mean and log deviation tensors for a batch of windows
     * that have already been encoded, so the graph stays intact.
     */
    public PolicyOutput forwardEncoded(Tensor encoded) {
        Tensor output = this.head.forward(encoded);
        Tensor mean = output.sliceCols(0, 2).tanh().mul(scaleTensor(output.rows));
        Tensor logDeviation = output.sliceCols(2, 4).clamp(-6f, 2f);
        return new PolicyOutput(mean, logDeviation, null, null);
    }

    private static Tensor scaleTensor(int rows) {
        Tensor scale = new Tensor(rows, 2);
        for (int r = 0; r < rows; r++) {
            scale.data[r * 2] = 45f;
            scale.data[r * 2 + 1] = 30f;
        }
        return scale;
    }

    @Override
    public PolicyOutput forward(Tensor observations) {
        // observations arrive flattened, one window per row, as the trainers hand them over
        if (observations.cols == INPUT_FEATURES) {
            return forwardEncoded(this.encoder.encode(observations));
        }
        throw new IllegalArgumentException("expected " + INPUT_FEATURES + " features, got " + observations.cols);
    }

    /**
     * Encodes a window into its context vector, for the trainer.
     */
    public Tensor encode(Tensor window) {
        return this.encoder.encode(window);
    }

    @Override
    public Module module() {
        return this;
    }

    @Override
    public int continuousActions() {
        return 2;
    }

    public int getWindow() {
        return this.window;
    }

    public SequenceEncoder getEncoder() {
        return this.encoder;
    }

    private static float clampLogDeviation(float value) {
        return Math.max(-6f, Math.min(2f, value));
    }
}
