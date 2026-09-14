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

package baritone.control.shaper;

import baritone.Baritone;
import baritone.api.control.ControlContext;
import baritone.api.control.IRotationShaper;
import baritone.api.control.RotationTarget;
import baritone.api.utils.Rotation;

/**
 * Turns instant snaps into movements a hand could have made.
 * <p>
 * Baritone's aim is teleportation: the rotation for tick {@code n} is computed and applied whole, so a ninety degree
 * turn happens in fifty milliseconds with no approach and no settle. This shaper replaces that with a second order
 * response - the aim has a velocity, that velocity is accelerated towards the error and damped as the error closes,
 * and a large correction is allowed to swing slightly past before coming back. The result passes through the existing
 * mouse quantization afterwards, so what leaves the client still lands on a legal GCD multiple.
 * <p>
 * It runs only on rotations that do not need to be exact, and even then the control manager clamps its output back
 * inside the target's tolerance. It cannot break a block interaction no matter how it is configured.
 *
 * @author Barelentless
 */
public final class HumanAimShaper implements IRotationShaper {

    private float yawVelocity;
    private float pitchVelocity;
    private long lastTick = -1;

    @Override
    public String name() {
        return "humanAim";
    }

    @Override
    public boolean isEnabled() {
        return Baritone.settings().humanAim.value;
    }

    @Override
    public Rotation shape(ControlContext context, RotationTarget target, Rotation current) {
        Rotation from = context.player().playerRotations();
        if (from == null) {
            return current;
        }
        if (context.tick() - this.lastTick > 4) {
            // been idle or in another world; starting from a stale velocity would fling the view
            this.yawVelocity = 0;
            this.pitchVelocity = 0;
        }
        this.lastTick = context.tick();

        final double responsiveness = clamp(Baritone.settings().humanAimResponsiveness.value, 0.02, 1.0);
        final double maxYawSpeed = Math.max(0.5, Baritone.settings().humanAimMaxYawSpeed.value);
        final double maxPitchSpeed = Math.max(0.5, Baritone.settings().humanAimMaxPitchSpeed.value);
        final double overshoot = clamp(Baritone.settings().humanAimOvershoot.value, 0.0, 0.5);

        float yawError = Rotation.normalizeYaw(current.getYaw() - from.getYaw());
        float pitchError = current.getPitch() - from.getPitch();

        this.yawVelocity = step(this.yawVelocity, yawError, responsiveness, maxYawSpeed, overshoot);
        this.pitchVelocity = step(this.pitchVelocity, pitchError, responsiveness, maxPitchSpeed, overshoot);

        return new Rotation(from.getYaw() + this.yawVelocity, from.getPitch() + this.pitchVelocity)
                .normalizeAndClamp();
    }

    /**
     * One axis of the response. Velocity is pulled towards the fraction of the error this tick should cover, which
     * gives acceleration at the start of a turn and deceleration at its end for free; the overshoot term adds a small
     * excess that only matters while the error is still large.
     */
    private static float step(float velocity, float error, double responsiveness, double maxSpeed, double overshoot) {
        if (Math.abs(error) < 1e-4) {
            return 0f;
        }
        double desired = error * responsiveness;
        if (Math.abs(error) > 25) {
            desired += Math.signum(error) * Math.abs(error) * overshoot * responsiveness;
        }
        // ease velocity towards the desired step rather than jumping to it; this is what makes a turn start slowly
        double next = velocity + (desired - velocity) * 0.5;
        next = Math.max(-maxSpeed, Math.min(maxSpeed, next));
        // never step past the target: overshoot is deliberate above, not an artifact of a large step
        if (Math.abs(next) > Math.abs(error) * (1 + overshoot)) {
            next = Math.signum(next) * Math.abs(error) * (1 + overshoot);
        }
        return (float) next;
    }

    private static double clamp(double value, double low, double high) {
        return Math.max(low, Math.min(high, value));
    }
}
