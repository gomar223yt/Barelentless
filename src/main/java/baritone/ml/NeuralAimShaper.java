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

import baritone.Baritone;
import baritone.api.control.ControlContext;
import baritone.api.control.IRotationShaper;
import baritone.api.control.RotationTarget;
import baritone.api.ml.Tensor;
import baritone.api.utils.Rotation;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Aims using the learned model, to the extent the model has earned it.
 * <p>
 * Two safeguards decide how much influence the model actually gets, and both matter more than the model itself. The
 * first is its own reported confidence: the model predicts a spread alongside its mean, and a prediction it is unsure
 * about is blended only weakly into the deterministic target. The second is the target's tolerance, enforced by the
 * control manager after this runs, which means that even a completely diverged model cannot break a block
 * interaction - it can only make aim look worse.
 * <p>
 * The result is that turning this on is never a cliff. An untrained model is initialized near zero and produces
 * almost no deviation; as it trains on real play, its confidence rises and its influence rises with it.
 *
 * @author Barelentless
 */
public final class NeuralAimShaper implements IRotationShaper {

    private final MlManager manager;
    private final Deque<float[]> window = new ArrayDeque<>();

    private Rotation previousRotation;
    private float previousYawError;
    private float previousPitchError;
    private AimModel.Aim lastPrediction;
    private long lastTick = -1;

    public NeuralAimShaper(MlManager manager) {
        this.manager = manager;
    }

    @Override
    public String name() {
        return "neuralAim";
    }

    @Override
    public boolean isEnabled() {
        return Baritone.settings().mlAim.value && this.manager.getAimModel() != null;
    }

    @Override
    public boolean appliesTo(RotationTarget.Purpose purpose) {
        // precise targets are allowed through: the tolerance clamp keeps the interaction valid, and excluding them
        // would mean the model never learns how a human approaches a block they are about to break
        return true;
    }

    @Override
    public Rotation shape(ControlContext context, RotationTarget target, Rotation current) {
        AimModel model = this.manager.getAimModel();
        if (model == null || context.player().player() == null) {
            return current;
        }
        Rotation from = context.player().playerRotations();
        float yawError = Rotation.normalizeYaw(current.getYaw() - from.getYaw());
        float pitchError = current.getPitch() - from.getPitch();

        if (context.tick() - this.lastTick > 4) {
            this.window.clear();
            this.previousYawError = yawError;
            this.previousPitchError = pitchError;
        }
        this.lastTick = context.tick();

        float[] row = buildRow(context, target, yawError, pitchError);
        this.window.addLast(row);
        while (this.window.size() > model.getWindow()) {
            this.window.removeFirst();
        }
        this.manager.recordAimContext(this.window, yawError, pitchError, from, this.previousRotation);
        this.previousYawError = yawError;
        this.previousPitchError = pitchError;
        this.previousRotation = from;

        AimModel.Aim aim;
        try {
            aim = model.predict(windowTensor(model));
        } catch (RuntimeException e) {
            this.manager.onInferenceFailure("aim", e);
            return current;
        }
        this.lastPrediction = aim;
        if (!Float.isFinite(aim.yawDelta) || !Float.isFinite(aim.pitchDelta)) {
            this.manager.onInferenceFailure("aim", new IllegalStateException("non-finite prediction"));
            return current;
        }

        float strength = (float) Math.max(0, Math.min(1, Baritone.settings().mlAimStrength.value));
        float weight = strength * aim.confidence();
        if (weight <= 0.001f) {
            return current;
        }
        Rotation predicted = new Rotation(from.getYaw() + aim.yawDelta, from.getPitch() + aim.pitchDelta);
        return new Rotation(
                from.getYaw() + Rotation.normalizeYaw(predicted.getYaw() - from.getYaw()) * weight
                        + yawError * (1 - weight),
                from.getPitch() + (predicted.getPitch() - from.getPitch()) * weight + pitchError * (1 - weight)
        ).normalizeAndClamp();
    }

    private float[] buildRow(ControlContext context, RotationTarget target, float yawError, float pitchError) {
        float[] state = this.manager.currentStateFeatures();
        float[] row = new float[AimModel.INPUT_FEATURES];
        System.arraycopy(state, 0, row, 0, Math.min(state.length, StateEncoder.FEATURES));
        int index = StateEncoder.FEATURES;
        row[index++] = yawError / 45f;
        row[index++] = pitchError / 45f;
        row[index++] = (yawError - this.previousYawError) / 45f;
        row[index++] = (pitchError - this.previousPitchError) / 45f;
        row[index++] = target.getPurpose().isPrecise() ? 1f : 0f;
        row[index++] = target.getPurpose() == RotationTarget.Purpose.MOVEMENT ? 1f : 0f;
        row[index++] = Math.min(target.getTolerance(), 30f) / 30f;
        row[index] = (float) Math.min(context.speed() * 10, 5);
        return row;
    }

    private Tensor windowTensor(AimModel model) {
        int rows = Math.max(1, Math.min(this.window.size(), model.getWindow()));
        Tensor tensor = new Tensor(rows, AimModel.INPUT_FEATURES);
        int r = 0;
        for (float[] row : this.window) {
            if (r >= rows) {
                break;
            }
            System.arraycopy(row, 0, tensor.data, r * AimModel.INPUT_FEATURES, AimModel.INPUT_FEATURES);
            r++;
        }
        return tensor;
    }

    /**
     * The most recent prediction, for the status readout. Null until the model has run once.
     */
    public AimModel.Aim getLastPrediction() {
        return this.lastPrediction;
    }
}
