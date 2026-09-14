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

import baritone.api.control.RotationTarget;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * How the bot walks and how it turns, as a set of numbers you can change at runtime.
 * <p>
 * The pipeline already lets anyone write a shaper, but writing Java is the wrong price for "turn a bit slower and add
 * some spread". A profile is the set of dials that covers what most people actually want to change, with a built-in
 * shaper that applies them - so tuning gait and aim is setting numbers, not writing code, and the numbers are
 * addressable by name from a command, a config screen or an addon.
 * <p>
 * Aim settings can be overridden per {@link RotationTarget.Purpose}. That matters more than it sounds: the right
 * amount of spread on "point roughly where I am walking" is not the right amount on "point at the block I am about
 * to break", and without per-purpose overrides you have to tune for the strictest case and lose the effect
 * everywhere else.
 *
 * @author Barelentless
 */
public final class MotionProfile {

    private String name;
    private final double[] aim = new double[AimKnob.values().length];
    private final double[] gait = new double[GaitKnob.values().length];
    private final Map<RotationTarget.Purpose, double[]> overrides = new EnumMap<>(RotationTarget.Purpose.class);

    /**
     * Seed for this profile's noise. Fixed rather than random so that a profile behaves the same way twice, which is
     * what makes tuning it possible at all; change it if you want a different-feeling instance of the same numbers.
     */
    private long seed = 0x5DEECE66DL;

    private boolean enabled = true;

    public MotionProfile(String name) {
        this.name = name;
        for (AimKnob knob : AimKnob.values()) {
            this.aim[knob.ordinal()] = knob.defaultValue();
        }
        for (GaitKnob knob : GaitKnob.values()) {
            this.gait[knob.ordinal()] = knob.defaultValue();
        }
    }

    // ------------------------------------------------------------------------------------------------------ values

    public double get(AimKnob knob) {
        return this.aim[knob.ordinal()];
    }

    /**
     * The value of an aim dial for a particular purpose, falling back to the profile's own value when that purpose
     * has no override.
     */
    public double get(AimKnob knob, RotationTarget.Purpose purpose) {
        double[] override = this.overrides.get(purpose);
        if (override == null || Double.isNaN(override[knob.ordinal()])) {
            return this.aim[knob.ordinal()];
        }
        return override[knob.ordinal()];
    }

    public double get(GaitKnob knob) {
        return this.gait[knob.ordinal()];
    }

    public MotionProfile set(AimKnob knob, double value) {
        this.aim[knob.ordinal()] = knob.clamp(value);
        return this;
    }

    public MotionProfile set(GaitKnob knob, double value) {
        this.gait[knob.ordinal()] = knob.clamp(value);
        return this;
    }

    /**
     * Overrides one aim dial for one purpose. Pass {@link Double#NaN} to drop the override and fall back to the
     * profile's own value.
     */
    public MotionProfile set(RotationTarget.Purpose purpose, AimKnob knob, double value) {
        double[] override = this.overrides.computeIfAbsent(purpose, ignored -> {
            double[] fresh = new double[AimKnob.values().length];
            java.util.Arrays.fill(fresh, Double.NaN);
            return fresh;
        });
        override[knob.ordinal()] = Double.isNaN(value) ? Double.NaN : knob.clamp(value);
        return this;
    }

    public boolean hasOverride(RotationTarget.Purpose purpose, AimKnob knob) {
        double[] override = this.overrides.get(purpose);
        return override != null && !Double.isNaN(override[knob.ordinal()]);
    }

    public MotionProfile clearOverrides(RotationTarget.Purpose purpose) {
        this.overrides.remove(purpose);
        return this;
    }

    // ---------------------------------------------------------------------------------------------- named access

    /**
     * Sets a value by name. Accepted forms:
     * <ul>
     *     <li>{@code maxYawSpeed} or {@code aim.maxYawSpeed} - an aim dial</li>
     *     <li>{@code gait.speed} or a gait dial's bare name</li>
     *     <li>{@code movement.jitter}, {@code block.maxYawSpeed} - an aim dial for one purpose</li>
     * </ul>
     *
     * @return Whether the name was recognised
     */
    public boolean setByName(String name, double value) {
        String key = name.trim();
        int dot = key.indexOf('.');
        if (dot > 0) {
            String prefix = key.substring(0, dot);
            String rest = key.substring(dot + 1);
            if (prefix.equalsIgnoreCase("aim")) {
                AimKnob knob = AimKnob.byKey(rest);
                if (knob != null) {
                    set(knob, value);
                    return true;
                }
                return false;
            }
            if (prefix.equalsIgnoreCase("gait")) {
                GaitKnob knob = GaitKnob.byKey(rest);
                if (knob != null) {
                    set(knob, value);
                    return true;
                }
                return false;
            }
            RotationTarget.Purpose purpose = purposeByName(prefix);
            AimKnob knob = AimKnob.byKey(rest);
            if (purpose != null && knob != null) {
                set(purpose, knob, value);
                return true;
            }
            return false;
        }
        AimKnob aimKnob = AimKnob.byKey(key);
        if (aimKnob != null) {
            set(aimKnob, value);
            return true;
        }
        GaitKnob gaitKnob = GaitKnob.byKey(key);
        if (gaitKnob != null) {
            set(gaitKnob, value);
            return true;
        }
        return false;
    }

    /**
     * @return The value of a named dial, or {@link Double#NaN} if the name is not recognised
     */
    public double getByName(String name) {
        String key = name.trim();
        int dot = key.indexOf('.');
        if (dot > 0) {
            String prefix = key.substring(0, dot);
            String rest = key.substring(dot + 1);
            if (prefix.equalsIgnoreCase("aim")) {
                AimKnob knob = AimKnob.byKey(rest);
                return knob == null ? Double.NaN : get(knob);
            }
            if (prefix.equalsIgnoreCase("gait")) {
                GaitKnob knob = GaitKnob.byKey(rest);
                return knob == null ? Double.NaN : get(knob);
            }
            RotationTarget.Purpose purpose = purposeByName(prefix);
            AimKnob knob = AimKnob.byKey(rest);
            return purpose == null || knob == null ? Double.NaN : get(knob, purpose);
        }
        AimKnob aimKnob = AimKnob.byKey(key);
        if (aimKnob != null) {
            return get(aimKnob);
        }
        GaitKnob gaitKnob = GaitKnob.byKey(key);
        return gaitKnob == null ? Double.NaN : get(gaitKnob);
    }

    public static RotationTarget.Purpose purposeByName(String name) {
        for (RotationTarget.Purpose purpose : RotationTarget.Purpose.values()) {
            String simple = purpose.name().toLowerCase().replace("_interact", "");
            if (purpose.name().equalsIgnoreCase(name) || simple.equalsIgnoreCase(name)) {
                return purpose;
            }
        }
        return null;
    }

    /**
     * Every dial with its current value and description, for a listing.
     */
    public List<String> describe() {
        List<String> lines = new ArrayList<>();
        lines.add("profile '" + this.name + "'" + (this.enabled ? "" : " (disabled)") + ", seed " + this.seed);
        lines.add("gait:");
        for (GaitKnob knob : GaitKnob.values()) {
            lines.add(String.format("  gait.%-18s %-8s %s", knob.key(), trim(get(knob)), knob.description()));
        }
        lines.add("aim:");
        for (AimKnob knob : AimKnob.values()) {
            lines.add(String.format("  aim.%-19s %-8s %s", knob.key(), trim(get(knob)), knob.description()));
        }
        for (Map.Entry<RotationTarget.Purpose, double[]> entry : this.overrides.entrySet()) {
            String prefix = entry.getKey().name().toLowerCase().replace("_interact", "");
            for (AimKnob knob : AimKnob.values()) {
                if (!Double.isNaN(entry.getValue()[knob.ordinal()])) {
                    lines.add(String.format("  %s.%-" + Math.max(1, 22 - prefix.length()) + "s %s",
                            prefix, knob.key(), trim(entry.getValue()[knob.ordinal()])));
                }
            }
        }
        return lines;
    }

    private static String trim(double value) {
        if (value == Math.rint(value) && Math.abs(value) < 1e9) {
            return String.valueOf((long) value);
        }
        return String.format("%.4g", value);
    }

    // ------------------------------------------------------------------------------------------- serialisation

    /**
     * Writes the profile as {@code key = value} lines. Only values that differ from the default are written, so a
     * saved profile reads as a description of what you changed rather than a wall of defaults.
     */
    public String save() {
        StringBuilder sb = new StringBuilder();
        sb.append("name = ").append(this.name).append('\n');
        sb.append("seed = ").append(this.seed).append('\n');
        if (!this.enabled) {
            sb.append("enabled = 0\n");
        }
        for (GaitKnob knob : GaitKnob.values()) {
            if (get(knob) != knob.defaultValue()) {
                sb.append("gait.").append(knob.key()).append(" = ").append(get(knob)).append('\n');
            }
        }
        for (AimKnob knob : AimKnob.values()) {
            if (get(knob) != knob.defaultValue()) {
                sb.append("aim.").append(knob.key()).append(" = ").append(get(knob)).append('\n');
            }
        }
        for (Map.Entry<RotationTarget.Purpose, double[]> entry : this.overrides.entrySet()) {
            String prefix = entry.getKey().name().toLowerCase().replace("_interact", "");
            for (AimKnob knob : AimKnob.values()) {
                double value = entry.getValue()[knob.ordinal()];
                if (!Double.isNaN(value)) {
                    sb.append(prefix).append('.').append(knob.key()).append(" = ").append(value).append('\n');
                }
            }
        }
        return sb.toString();
    }

    /**
     * Reads a profile back. Unknown keys are collected rather than thrown, so a profile written by a newer build
     * still loads everything it can - losing a dial you have never heard of is better than losing the file.
     *
     * @param problems Receives a line per unrecognised or unparseable entry; may be null
     */
    public static MotionProfile load(String text, String fallbackName, List<String> problems) {
        MotionProfile profile = new MotionProfile(fallbackName);
        int lineNumber = 0;
        for (String line : text.split("\n")) {
            lineNumber++;
            String trimmed = line.trim();
            if (trimmed.isEmpty() || trimmed.startsWith("#") || trimmed.startsWith("//")) {
                continue;
            }
            int equals = trimmed.indexOf('=');
            if (equals < 0) {
                if (problems != null) {
                    problems.add("line " + lineNumber + ": expected 'key = value'");
                }
                continue;
            }
            String key = trimmed.substring(0, equals).trim();
            String value = trimmed.substring(equals + 1).trim();
            if (key.equalsIgnoreCase("name")) {
                profile.name = value;
                continue;
            }
            try {
                double parsed = Double.parseDouble(value);
                if (key.equalsIgnoreCase("seed")) {
                    profile.seed = (long) parsed;
                } else if (key.equalsIgnoreCase("enabled")) {
                    profile.enabled = parsed != 0;
                } else if (!profile.setByName(key, parsed) && problems != null) {
                    problems.add("line " + lineNumber + ": no dial called '" + key + "'");
                }
            } catch (NumberFormatException e) {
                if (problems != null) {
                    problems.add("line " + lineNumber + ": '" + value + "' is not a number");
                }
            }
        }
        return profile;
    }

    public MotionProfile copy() {
        MotionProfile copy = new MotionProfile(this.name);
        System.arraycopy(this.aim, 0, copy.aim, 0, this.aim.length);
        System.arraycopy(this.gait, 0, copy.gait, 0, this.gait.length);
        for (Map.Entry<RotationTarget.Purpose, double[]> entry : this.overrides.entrySet()) {
            copy.overrides.put(entry.getKey(), entry.getValue().clone());
        }
        copy.seed = this.seed;
        copy.enabled = this.enabled;
        return copy;
    }

    /**
     * Every dial's name mapped to its description, for a config screen.
     */
    public static Map<String, String> vocabulary() {
        Map<String, String> all = new LinkedHashMap<>();
        for (GaitKnob knob : GaitKnob.values()) {
            all.put("gait." + knob.key(), knob.description());
        }
        for (AimKnob knob : AimKnob.values()) {
            all.put("aim." + knob.key(), knob.description());
        }
        return all;
    }

    // -------------------------------------------------------------------------------------------------- metadata

    public String getName() {
        return this.name;
    }

    public MotionProfile setName(String name) {
        this.name = name;
        return this;
    }

    public long getSeed() {
        return this.seed;
    }

    public MotionProfile setSeed(long seed) {
        this.seed = seed;
        return this;
    }

    public boolean isEnabled() {
        return this.enabled;
    }

    public MotionProfile setEnabled(boolean enabled) {
        this.enabled = enabled;
        return this;
    }

    /**
     * Whether every dial is still at its default.
     * <p>
     * Note that this is not the same as "does nothing": the defaults describe a deliberate, human-ish gait and aim,
     * and applying them changes how the bot moves. Whether a profile is applied at all is {@link #isEnabled()},
     * which is why the shipped default profile starts disabled - installing the mod should not silently change how
     * it walks.
     */
    public boolean isAtDefaults() {
        for (AimKnob knob : AimKnob.values()) {
            if (get(knob) != knob.defaultValue()) {
                return false;
            }
        }
        for (GaitKnob knob : GaitKnob.values()) {
            if (get(knob) != knob.defaultValue()) {
                return false;
            }
        }
        return this.overrides.isEmpty();
    }

    @Override
    public String toString() {
        return "MotionProfile[" + this.name + "]";
    }
}
