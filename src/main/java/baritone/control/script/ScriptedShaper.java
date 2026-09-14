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

package baritone.control.script;

import baritone.api.control.ControlContext;
import baritone.api.control.IInputShaper;
import baritone.api.control.IRotationShaper;
import baritone.api.control.MovementCommand;
import baritone.api.control.RotationTarget;
import baritone.api.control.script.ControlScript;
import baritone.api.utils.Rotation;
import baritone.api.utils.input.Input;

/**
 * Runs a control script as a stage of the movement and aim pipeline.
 * <p>
 * The script decides values; this decides what they mean. A script that writes {@code forward} is a movement shaper,
 * one that writes {@code yaw} or {@code yawDelta} is an aim shaper, and one that writes both is both - inferred from
 * what it actually assigns, so nobody has to declare it.
 * <p>
 * Every value coming out of a script is treated as hostile: non-finite numbers are dropped, impulses are clamped to
 * the legal range, and angles go through the same tolerance clamp as any other shaper. A script is arithmetic written
 * by a person in a text file at three in the morning; it will produce a NaN eventually, and when it does the right
 * outcome is that the tick carries on.
 *
 * @author Barelentless
 */
public final class ScriptedShaper implements IInputShaper, IRotationShaper {

    private final ControlScript script;
    private final ScriptBindings bindings;
    private final double[] frame;

    private boolean enabled = true;
    private long evaluations;
    private long skipped;
    private String lastProblem = "";

    public ScriptedShaper(ControlScript script, ScriptBindings bindings) {
        this.script = script;
        this.bindings = bindings;
        this.frame = new double[script.frameSize()];
    }

    @Override
    public String name() {
        return this.script.getName();
    }

    @Override
    public boolean isEnabled() {
        return this.enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public ControlScript getScript() {
        return this.script;
    }

    public boolean shapesMovement() {
        return this.script.assigns("forward") || this.script.assigns("strafe") || this.script.assigns("jump")
                || this.script.assigns("sprint") || this.script.assigns("sneak")
                || this.script.assigns("sneakScale");
    }

    public boolean shapesAim() {
        return this.script.assigns("yaw") || this.script.assigns("pitch")
                || this.script.assigns("yawDelta") || this.script.assigns("pitchDelta");
    }

    // ------------------------------------------------------------------------------------------------------ input

    @Override
    public void shape(ControlContext context, MovementCommand command) {
        if (!shapesMovement()) {
            return;
        }
        try {
            this.bindings.fill(this.frame, context, command);
            if (!this.script.run(this.frame)) {
                this.skipped++;
                return;
            }
        } catch (RuntimeException e) {
            fail(e.toString());
            return;
        } finally {
            this.bindings.release();
        }
        this.evaluations++;

        if (this.script.assigns("forward") || this.script.assigns("strafe")) {
            double forward = this.frame[this.bindings.slotForward()];
            double strafe = this.frame[this.bindings.slotStrafe()];
            if (finite(forward) && finite(strafe)) {
                command.setAnalog((float) clamp(forward), (float) clamp(strafe));
            } else {
                fail("forward or strafe was not a finite number");
            }
        }
        if (this.script.assigns("jump")) {
            command.set(Input.JUMP, truthy(this.frame[this.bindings.slotJump()]));
        }
        if (this.script.assigns("sprint")) {
            command.setSprintAllowed(truthy(this.frame[this.bindings.slotSprint()]));
        }
        if (this.script.assigns("sneak")) {
            command.set(Input.SNEAK, truthy(this.frame[this.bindings.slotSneak()]));
        }
        if (this.script.assigns("sneakScale")) {
            double scale = this.frame[this.bindings.slotSneakScale()];
            if (finite(scale)) {
                command.setSneakScale((float) scale);
            }
        }
    }

    // --------------------------------------------------------------------------------------------------- rotation

    @Override
    public boolean appliesTo(RotationTarget.Purpose purpose) {
        // scripts see every purpose, including the precise ones. The tolerance clamp downstream is what keeps a
        // block interaction valid, and a script that wants to leave precise aim alone can say so with `when !precise`
        return true;
    }

    @Override
    public Rotation shape(ControlContext context, RotationTarget target, Rotation current) {
        if (!shapesAim()) {
            return current;
        }
        try {
            this.bindings.fill(this.frame, context, context.previousCommand());
            this.bindings.fillAim(this.frame, context, target, current);
            if (!this.script.run(this.frame)) {
                this.skipped++;
                return current;
            }
        } catch (RuntimeException e) {
            fail(e.toString());
            return current;
        } finally {
            this.bindings.release();
        }
        this.evaluations++;

        Rotation from = context.player().playerRotations();
        float yaw = current.getYaw();
        float pitch = current.getPitch();
        // absolute assignments win over deltas, because a script that sets both meant the absolute one
        if (this.script.assigns("yawDelta")) {
            double delta = this.frame[this.bindings.slotYawDelta()];
            if (finite(delta)) {
                yaw = from.getYaw() + (float) delta;
            }
        }
        if (this.script.assigns("yaw")) {
            double value = this.frame[this.bindings.slotYaw()];
            if (finite(value)) {
                yaw = (float) value;
            }
        }
        if (this.script.assigns("pitchDelta")) {
            double delta = this.frame[this.bindings.slotPitchDelta()];
            if (finite(delta)) {
                pitch = from.getPitch() + (float) delta;
            }
        }
        if (this.script.assigns("pitch")) {
            double value = this.frame[this.bindings.slotPitch()];
            if (finite(value)) {
                pitch = (float) value;
            }
        }
        if (!finite(yaw) || !finite(pitch)) {
            fail("aim was not a finite number");
            return current;
        }
        return new Rotation(yaw, pitch).normalizeAndClamp();
    }

    // ---------------------------------------------------------------------------------------------------- helpers

    private void fail(String problem) {
        this.lastProblem = problem;
        this.enabled = false;
    }

    private static boolean finite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }

    private static boolean truthy(double value) {
        return value > 0.5;
    }

    private static double clamp(double value) {
        return Math.max(-1, Math.min(1, value));
    }

    public long getEvaluations() {
        return this.evaluations;
    }

    public long getSkipped() {
        return this.skipped;
    }

    public String getLastProblem() {
        return this.lastProblem;
    }

    /**
     * A one line summary for {@code control script list}.
     */
    public String describe() {
        StringBuilder sb = new StringBuilder();
        sb.append(this.script.getName()).append("  priority ").append(this.script.getPriority());
        sb.append("  [");
        if (shapesMovement()) {
            sb.append("movement");
        }
        if (shapesMovement() && shapesAim()) {
            sb.append(", ");
        }
        if (shapesAim()) {
            sb.append("aim");
        }
        sb.append(']');
        sb.append(this.enabled ? "" : "  DISABLED");
        if (!this.lastProblem.isEmpty()) {
            sb.append("  (").append(this.lastProblem).append(')');
        }
        sb.append("  ran ").append(this.evaluations);
        if (this.script.hasGuard()) {
            sb.append(", skipped ").append(this.skipped);
        }
        return sb.toString();
    }
}
