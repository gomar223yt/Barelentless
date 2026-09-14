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

package baritone.control.profile;

import baritone.Baritone;
import baritone.api.control.ControlContext;
import baritone.api.control.IInputShaper;
import baritone.api.control.IRotationShaper;
import baritone.api.control.MovementCommand;
import baritone.api.control.RotationTarget;
import baritone.api.control.profile.AimKnob;
import baritone.api.control.profile.GaitKnob;
import baritone.api.control.profile.MotionProfile;
import baritone.api.utils.Rotation;
import baritone.api.utils.input.Input;
import baritone.pathing.movement.MovementHelper;
import baritone.utils.BlockStateInterface;
import net.minecraft.world.level.block.state.BlockState;

import java.util.Random;
import java.util.function.Supplier;

/**
 * Applies a {@link MotionProfile} to movement and aim.
 * <p>
 * The aim half is a second-order model. Aim has a velocity; that velocity accelerates towards the speed the
 * responsiveness dial asks for, is capped, and is allowed a little excess on large corrections so the view swings
 * past and settles. On top of that sit two kinds of noise, which are different things and are dialled separately:
 * <em>jitter</em> is per-tick white noise, the tremor of a hand that is not perfectly still, and <em>drift</em> is a
 * slow smooth wander that makes a view look alive rather than vibrating. Neither accumulates into a bias, so spread
 * never pulls the aim off target - it just stops it being exactly on it.
 * <p>
 * Everything here is bounded twice over. The model's own output is capped by the speed dials, and the control
 * manager then clamps whatever comes out back inside the tolerance of the rotation that was requested. Turning the
 * spread up to its maximum makes aim look drunk; it still cannot cost you a block break.
 *
 * @author Barelentless
 */
public final class ProfileShaper implements IInputShaper, IRotationShaper {

    private final Supplier<MotionProfile> profile;
    private final Random random;

    // aim state
    private double yawVelocity;
    private double pitchVelocity;
    private float previousTargetYaw;
    private int reactionCountdown;
    private long lastAimTick = -1;

    // gait state
    private double speedRamp;
    private double smoothedStrafe;
    private int hesitationCountdown;
    private long lastGaitTick = -1;

    public ProfileShaper(Supplier<MotionProfile> profile) {
        this.profile = profile;
        this.random = new Random();
    }

    @Override
    public String name() {
        return "profile";
    }

    @Override
    public boolean isEnabled() {
        MotionProfile active = this.profile.get();
        return active != null && active.isEnabled();
    }

    // ------------------------------------------------------------------------------------------------------- gait

    @Override
    public void shape(ControlContext context, MovementCommand command) {
        MotionProfile active = this.profile.get();
        if (active == null || command.isIdle()) {
            this.speedRamp = 0;
            this.smoothedStrafe = 0;
            return;
        }
        if (context.tick() - this.lastGaitTick > 4) {
            // been idle or in another world; a stale ramp would make the first step lurch
            this.speedRamp = 0;
            this.smoothedStrafe = command.getStrafeImpulse();
            this.hesitationCountdown = 0;
        }
        this.lastGaitTick = context.tick();

        double forward = command.getForwardImpulse();
        double strafe = command.getStrafeImpulse() * active.get(GaitKnob.STRAFE);

        // lean into the corner: feed part of the angle to the destination into the sideways axis, so the bot arcs
        // through a turn instead of pivoting on the spot
        double lean = active.get(GaitKnob.CORNER_LEAN);
        if (lean > 0 && context.movement() != null) {
            double heading = headingToDestination(context);
            strafe += Math.max(-1, Math.min(1, heading / 60)) * lean;
        }

        double target = active.get(GaitKnob.SPEED);
        double edge = active.get(GaitKnob.EDGE_CAUTION);
        boolean atEdge = false;
        if (edge > 0) {
            atEdge = !floorAhead(context, (int) Math.round(active.get(GaitKnob.EDGE_LOOKAHEAD)));
            if (atEdge) {
                target *= 1 - edge * 0.8;
            }
        }

        if (this.hesitationCountdown > 0) {
            this.hesitationCountdown--;
            target = 0;
        } else if (active.get(GaitKnob.HESITATION) > 0
                && this.random.nextDouble() < active.get(GaitKnob.HESITATION)) {
            this.hesitationCountdown = (int) Math.round(active.get(GaitKnob.HESITATION_TICKS));
            target = 0;
        }

        // ease towards the target speed rather than jumping to it: this is the difference between a walk that starts
        // and one that teleports into motion
        double accelerationTicks = active.get(GaitKnob.ACCELERATION_TICKS);
        double brakingTicks = active.get(GaitKnob.BRAKING_TICKS);
        double ticks = target > this.speedRamp ? accelerationTicks : brakingTicks;
        if (ticks <= 0) {
            this.speedRamp = target;
        } else {
            this.speedRamp += (target - this.speedRamp) / Math.max(1, ticks);
        }

        double smoothing = active.get(GaitKnob.STRAFE_SMOOTHING);
        this.smoothedStrafe = this.smoothedStrafe * smoothing + strafe * (1 - smoothing);

        double jitter = active.get(GaitKnob.JITTER);
        double forwardOut = forward * this.speedRamp;
        double strafeOut = this.smoothedStrafe * this.speedRamp;
        if (jitter > 0) {
            forwardOut += (this.random.nextDouble() * 2 - 1) * jitter;
            strafeOut += (this.random.nextDouble() * 2 - 1) * jitter;
        }
        command.setAnalog((float) clampUnit(forwardOut), (float) clampUnit(strafeOut));

        int policy = (int) Math.round(active.get(GaitKnob.SPRINT_POLICY));
        switch (policy) {
            case 0:
                command.setSprintAllowed(false);
                break;
            case 1:
                command.setSprintAllowed(command.isSprintAllowed() && !atEdge);
                break;
            case 3:
                command.setSprintAllowed(true);
                if (forwardOut > 0.5) {
                    command.set(Input.SPRINT, true);
                }
                break;
            default:
                break;
        }

        if (atEdge && active.get(GaitKnob.EDGE_SNEAK) > 0.5 && !command.isPressed(Input.JUMP)) {
            command.set(Input.SNEAK, true);
        }
        command.setSneakScale((float) active.get(GaitKnob.SNEAK_SCALE));
    }

    private static double headingToDestination(ControlContext context) {
        if (context.player().player() == null || context.movement() == null) {
            return 0;
        }
        double toEast = context.movement().getDest().x + 0.5 - context.player().player().position().x;
        double toSouth = context.movement().getDest().z + 0.5 - context.player().player().position().z;
        double radians = Math.toRadians(context.player().player().getYRot());
        double ahead = -toEast * Math.sin(radians) + toSouth * Math.cos(radians);
        double left = -toEast * Math.cos(radians) - toSouth * Math.sin(radians);
        return Math.toDegrees(Math.atan2(left, ahead));
    }

    private static boolean floorAhead(ControlContext context, int lookahead) {
        BlockStateInterface blocks = ((Baritone) context.baritone()).bsi;
        if (blocks == null || context.player().player() == null) {
            return true;
        }
        baritone.api.utils.BetterBlockPos feet = context.player().playerFeet();
        float yaw = context.player().player().getYRot();
        int quarter = Math.floorMod(Math.round(yaw / 90f), 4);
        for (int step = 1; step <= Math.max(1, lookahead); step++) {
            int east;
            int south;
            switch (quarter) {
                case 0:
                    east = 0;
                    south = step;
                    break;
                case 1:
                    east = -step;
                    south = 0;
                    break;
                case 2:
                    east = 0;
                    south = -step;
                    break;
                default:
                    east = step;
                    south = 0;
                    break;
            }
            int x = feet.x + east;
            int y = feet.y - 1;
            int z = feet.z + south;
            if (!blocks.worldContainsLoadedChunk(x, z)) {
                return true; // unknown terrain is not evidence of a drop
            }
            BlockState state = blocks.get0(x, y, z);
            if (MovementHelper.canWalkOn(blocks, x, y, z, state)) {
                return true;
            }
        }
        return false;
    }

    // -------------------------------------------------------------------------------------------------------- aim

    @Override
    public boolean appliesTo(RotationTarget.Purpose purpose) {
        MotionProfile active = this.profile.get();
        if (active == null) {
            return false;
        }
        return !purpose.isPrecise() || active.get(AimKnob.SHAPE_PRECISE, purpose) > 0.5;
    }

    @Override
    public Rotation shape(ControlContext context, RotationTarget target, Rotation current) {
        MotionProfile active = this.profile.get();
        if (active == null || context.player().player() == null) {
            return current;
        }
        RotationTarget.Purpose purpose = target.getPurpose();
        Rotation from = context.player().playerRotations();

        if (context.tick() - this.lastAimTick > 4) {
            this.yawVelocity = 0;
            this.pitchVelocity = 0;
            this.reactionCountdown = 0;
            this.previousTargetYaw = current.getYaw();
        }
        this.lastAimTick = context.tick();

        // a target that has just jumped a long way is something a person would take a moment to react to
        float targetJump = Math.abs(Rotation.normalizeYaw(current.getYaw() - this.previousTargetYaw));
        this.previousTargetYaw = current.getYaw();
        double reactionTicks = active.get(AimKnob.REACTION_TICKS, purpose);
        if (reactionTicks > 0 && targetJump > active.get(AimKnob.REACTION_THRESHOLD, purpose)) {
            this.reactionCountdown = (int) Math.round(reactionTicks);
        }
        if (this.reactionCountdown > 0) {
            this.reactionCountdown--;
            this.yawVelocity *= 0.5;
            this.pitchVelocity *= 0.5;
            return new Rotation(from.getYaw() + (float) this.yawVelocity,
                    from.getPitch() + (float) this.pitchVelocity).normalizeAndClamp();
        }

        float yawError = Rotation.normalizeYaw(current.getYaw() - from.getYaw());
        float pitchError = current.getPitch() - from.getPitch();

        double speedScale = context.player().player().isSprinting()
                ? active.get(AimKnob.SPRINT_PENALTY, purpose)
                : 1;
        double snap = active.get(AimKnob.SNAP_BELOW, purpose);

        double yaw;
        double pitch;
        if (Math.abs(yawError) <= snap) {
            this.yawVelocity = 0;
            yaw = yawError;
        } else {
            this.yawVelocity = step(this.yawVelocity, yawError, active, purpose,
                    active.get(AimKnob.MAX_YAW_SPEED, purpose) * speedScale);
            yaw = this.yawVelocity;
        }
        if (Math.abs(pitchError) <= snap) {
            this.pitchVelocity = 0;
            pitch = pitchError;
        } else {
            this.pitchVelocity = step(this.pitchVelocity, pitchError, active, purpose,
                    active.get(AimKnob.MAX_PITCH_SPEED, purpose) * speedScale);
            pitch = this.pitchVelocity;
        }

        if (!purpose.isPrecise() || active.get(AimKnob.NOISE_ON_PRECISE, purpose) > 0.5) {
            double jitter = active.get(AimKnob.JITTER, purpose);
            if (jitter > 0) {
                yaw += (this.random.nextDouble() * 2 - 1) * jitter;
                pitch += (this.random.nextDouble() * 2 - 1) * jitter * 0.5;
            }
            double drift = active.get(AimKnob.DRIFT, purpose);
            if (drift > 0) {
                // sampled at the current tick so it is smooth and repeatable, rather than a random walk that would
                // slowly pull the aim away from where it is supposed to be
                double phase = context.tick() * active.get(AimKnob.DRIFT_SPEED, purpose) / 20.0;
                yaw += noise(phase, active.getSeed()) * drift;
                pitch += noise(phase + 137.0, active.getSeed()) * drift * 0.5;
            }
        }

        return new Rotation(from.getYaw() + (float) yaw, from.getPitch() + (float) pitch).normalizeAndClamp();
    }

    /**
     * One axis of the turn. Velocity eases towards the speed responsiveness asks for, which produces acceleration at
     * the start of a turn and deceleration at its end for free, and the overshoot term adds a controlled excess that
     * only applies while the error is still large.
     */
    static double step(double velocity, double error, MotionProfile profile,
                               RotationTarget.Purpose purpose, double maxSpeed) {
        double responsiveness = profile.get(AimKnob.RESPONSIVENESS, purpose);
        double acceleration = profile.get(AimKnob.ACCELERATION, purpose);
        double overshoot = profile.get(AimKnob.OVERSHOOT, purpose);
        double threshold = profile.get(AimKnob.OVERSHOOT_THRESHOLD, purpose);

        double desired = error * responsiveness;
        if (overshoot > 0 && Math.abs(error) > threshold) {
            desired += Math.signum(error) * Math.abs(error) * overshoot * responsiveness;
        }
        double next = velocity + (desired - velocity) * acceleration;
        next = Math.max(-maxSpeed, Math.min(maxSpeed, next));

        // never step further past the target than the overshoot allows; anything more is an artifact of a large
        // step rather than a deliberate swing
        double limit = Math.abs(error) * (1 + overshoot);
        if (Math.abs(next) > limit) {
            next = Math.signum(next) * limit;
        }
        return next;
    }

    /**
     * Smooth value noise in {@code [-1, 1]}: hash the integer lattice and interpolate. Deterministic for a given
     * phase and seed, which is what makes a profile behave the same way twice.
     */
    static double noise(double phase, long seed) {
        double floor = Math.floor(phase);
        double t = phase - floor;
        double smooth = t * t * (3 - 2 * t);
        return hash((long) floor, seed) * (1 - smooth) + hash((long) floor + 1, seed) * smooth;
    }

    private static double hash(long value, long seed) {
        long x = (value ^ seed) * 0x9E3779B97F4A7C15L;
        x ^= x >>> 29;
        x *= 0xBF58476D1CE4E5B9L;
        x ^= x >>> 32;
        // 53 significant bits shifted out, so the divisor is 2^53 to land in [0, 1) before the rescale;
        // dividing by 2^52 would quietly produce [-1, 3] and a "random" spread biased upward
        return (x >>> 11) / (double) (1L << 53) * 2 - 1;
    }

    private static double clampUnit(double value) {
        if (Double.isNaN(value)) {
            return 0;
        }
        return Math.max(-1, Math.min(1, value));
    }
}
