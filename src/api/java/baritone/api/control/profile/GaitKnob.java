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
 * Every dial that shapes how the bot walks.
 * <p>
 * All of these are expressible only because movement is analog: "eighty percent speed", "ease into a start over four
 * ticks", "lean into corners" are not things eight booleans can say. With vanilla-style input the only available
 * gaits are "full speed" and "stopped".
 *
 * @author Barelentless
 */
public enum GaitKnob {

    /**
     * Multiplies the movement vector. Below one the bot walks deliberately; at one it moves as pathing intended.
     */
    SPEED("speed", 1, 0.05, 1, "multiplier on the movement vector"),

    /**
     * Ticks taken to ramp from a standstill to full speed. Zero starts instantly, which is what vanilla input does
     * and what reads most obviously as a machine.
     */
    ACCELERATION_TICKS("accelerationTicks", 0, 0, 40, "ticks to ease from stopped to full speed"),

    /**
     * Ticks taken to ease back down when the requested speed drops.
     */
    BRAKING_TICKS("brakingTicks", 0, 0, 40, "ticks to ease down when slowing"),

    /**
     * How much of the requested strafe survives, as a fraction. Below one the bot commits more to its heading and
     * turns rather than sliding sideways.
     */
    STRAFE("strafe", 1, 0, 2, "multiplier on sideways movement"),

    /**
     * Smooths the strafe axis over time, zero to one. Higher is smoother and slower to respond.
     */
    STRAFE_SMOOTHING("strafeSmoothing", 0, 0, 0.95, "how much the strafe axis is smoothed over ticks"),

    /**
     * Random noise added to the movement vector every tick, as a fraction of full speed. A person does not hold a
     * key perfectly; this is that, and it is small on purpose.
     */
    JITTER("jitter", 0, 0, 0.5, "random noise on the movement vector, fraction of full speed"),

    /**
     * How strongly the bot leans into a turn, arcing through corners rather than pivoting on the spot. One means the
     * full angle to the destination is fed into the strafe axis.
     */
    CORNER_LEAN("cornerLean", 0, 0, 1, "how much the bot arcs through corners instead of turning in place"),

    /**
     * Sprint policy: 0 never, 1 only when the ground ahead is safe, 2 whatever pathing asked for, 3 always when
     * moving forward.
     */
    SPRINT_POLICY("sprintPolicy", 2, 0, 3, "0 never, 1 when safe, 2 as pathing asked, 3 always"),

    /**
     * Speed multiplier when there is no floor one block ahead. Zero disables the check entirely.
     */
    EDGE_CAUTION("edgeCaution", 0, 0, 1, "how much to slow down when there is no floor just ahead"),

    /**
     * How many blocks ahead the edge check looks.
     */
    EDGE_LOOKAHEAD("edgeLookahead", 1, 1, 4, "how many blocks ahead to check for floor"),

    /**
     * Whether to sneak when the edge check fires and a fall looks likely. One to sneak, zero to only slow down.
     */
    EDGE_SNEAK("edgeSneak", 0, 0, 1, "1 to sneak at edges rather than only slowing"),

    /**
     * Chance per tick of a brief hesitation, as a fraction. A person walking somewhere does not move at a perfectly
     * constant rate for four minutes.
     */
    HESITATION("hesitation", 0, 0, 0.2, "chance per tick of a brief pause"),

    /**
     * How many ticks a hesitation lasts.
     */
    HESITATION_TICKS("hesitationTicks", 2, 1, 20, "ticks a hesitation lasts"),

    /**
     * How much sneaking slows movement. Vanilla is 0.3.
     */
    SNEAK_SCALE("sneakScale", 0.3, 0.05, 1, "how much sneaking slows movement; vanilla is 0.3");

    private final String key;
    private final double defaultValue;
    private final double minimum;
    private final double maximum;
    private final String description;

    GaitKnob(String key, double defaultValue, double minimum, double maximum, String description) {
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

    public static GaitKnob byKey(String key) {
        for (GaitKnob knob : values()) {
            if (knob.key.equalsIgnoreCase(key)) {
                return knob;
            }
        }
        return null;
    }
}
