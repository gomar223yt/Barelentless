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

package baritone.api.control.profile;

/**
 * Every dial that shapes how the view turns.
 * <p>
 * An enum rather than fields because every consumer needs the same three things: a stable name to save and address it
 * by, a legal range to clamp into, and a description to show. With an enum, a profile is a {@code double[]} indexed
 * by ordinal - trivially copied, saved, diffed and edited by name, with no reflection anywhere, which matters because
 * the shipped jar is shrunk by ProGuard.
 * <p>
 * The model these describe is a second-order one: aim has a velocity, that velocity accelerates towards the error and
 * decays as the error closes, and noise is layered on top. That is what a hand on a mouse does; a proportional
 * controller alone reads as a machine no matter how you tune it.
 *
 * @author Barelentless
 */
public enum AimKnob {

    /**
     * Maximum degrees of yaw change per tick. The hard ceiling on how fast the view can turn.
     */
    MAX_YAW_SPEED("maxYawSpeed", 22, 0.1, 3600, "degrees of yaw per tick, at most"),

    /**
     * Maximum degrees of pitch change per tick.
     */
    MAX_PITCH_SPEED("maxPitchSpeed", 14, 0.1, 3600, "degrees of pitch per tick, at most"),

    /**
     * Fraction of the remaining error aimed at per tick. Low is lazy, high is snappy.
     */
    RESPONSIVENESS("responsiveness", 0.35, 0.01, 1, "fraction of the remaining error covered per tick"),

    /**
     * How quickly the turn's velocity reaches the speed responsiveness asks for. One means instantly, which removes
     * the acceleration entirely and gives you a plain proportional controller.
     */
    ACCELERATION("acceleration", 0.5, 0.05, 1, "how fast the turn accelerates, 1 for no ramp"),

    /**
     * How far past the target a large correction may swing, as a fraction of the correction.
     */
    OVERSHOOT("overshoot", 0.06, 0, 0.5, "how far a big turn swings past the target, as a fraction"),

    /**
     * Error below which overshoot is not applied, in degrees. Small corrections do not overshoot.
     */
    OVERSHOOT_THRESHOLD("overshootThreshold", 25, 0, 180, "smallest turn that is allowed to overshoot, degrees"),

    /**
     * Random noise added to the aim every tick, in degrees. This is the tremor of a hand that is not perfectly
     * still - it never accumulates, so it does not pull aim off target, it just stops it being exact.
     */
    JITTER("jitter", 0, 0, 45, "random spread added every tick, degrees"),

    /**
     * Slow wandering added to the aim, in degrees. Unlike jitter this is smooth and correlated over time, which is
     * what makes a view look alive rather than vibrating.
     */
    DRIFT("drift", 0, 0, 45, "slow smooth wander, degrees"),

    /**
     * How fast the drift wanders, in cycles per second. Low is a lazy sway, high is restless.
     */
    DRIFT_SPEED("driftSpeed", 0.4, 0.01, 20, "how fast the drift wanders, cycles per second"),

    /**
     * Ticks to wait before reacting to a target that has just moved a long way. A person does not begin turning on
     * the same frame the thing they are reacting to appears.
     */
    REACTION_TICKS("reactionTicks", 0, 0, 20, "ticks of delay before reacting to a new target"),

    /**
     * How far the target must move to count as new, for the reaction delay, in degrees.
     */
    REACTION_THRESHOLD("reactionThreshold", 35, 0, 180, "how far a target must jump to trigger the reaction delay"),

    /**
     * Error below which the aim simply snaps to the target, in degrees. Keeps the last fraction of a degree from
     * taking twenty ticks of asymptotic crawling.
     */
    SNAP_BELOW("snapBelow", 0.35, 0, 45, "error below which aim snaps to target, degrees"),

    /**
     * Multiplies every speed while sprinting. People aim less precisely while moving fast.
     */
    SPRINT_PENALTY("sprintPenalty", 1, 0.1, 4, "speed multiplier while sprinting"),

    /**
     * Whether this profile shapes rotations that have to be exact, such as breaking a block. Zero leaves them
     * untouched; one shapes them too, still clamped inside their tolerance so the interaction cannot break.
     */
    SHAPE_PRECISE("shapePrecise", 0, 0, 1, "1 to shape aim that must be exact (still clamped to tolerance)"),

    /**
     * Whether jitter and drift apply to exact rotations. Off by default even when {@link #SHAPE_PRECISE} is on,
     * because noise is the part most likely to cost you a block break at the edge of tolerance.
     */
    NOISE_ON_PRECISE("noiseOnPrecise", 0, 0, 1, "1 to apply jitter and drift to exact aim too");

    private final String key;
    private final double defaultValue;
    private final double minimum;
    private final double maximum;
    private final String description;

    AimKnob(String key, double defaultValue, double minimum, double maximum, String description) {
        this.key = key;
        this.defaultValue = defaultValue;
        this.minimum = minimum;
        this.maximum = maximum;
        this.description = description;
    }

    public String key() {
        return this.key;
    }

    public double defaultValue() {
        return this.defaultValue;
    }

    public double minimum() {
        return this.minimum;
    }

    public double maximum() {
        return this.maximum;
    }

    public String description() {
        return this.description;
    }

    public double clamp(double value) {
        if (Double.isNaN(value)) {
            return this.defaultValue;
        }
        return Math.max(this.minimum, Math.min(this.maximum, value));
    }

    public static AimKnob byKey(String key) {
        for (AimKnob knob : values()) {
            if (knob.key.equalsIgnoreCase(key)) {
                return knob;
            }
        }
        return null;
    }
}
