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

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The scripts written into {@code baritone/control/} the first time it is used.
 * <p>
 * They exist because an empty directory teaches nobody anything. Each one is a complete, working controller that does
 * something worth doing, and between them they use every part of the language: guards, locals, world queries, angle
 * arithmetic and both movement and aim outputs. All four ship disabled behind a {@code when} that is easy to widen,
 * so installing the mod does not silently change how it walks.
 *
 * @author Barelentless
 */
public final class ControlScriptExamples {

    private ControlScriptExamples() {}

    public static Map<String, String> all() {
        Map<String, String> examples = new LinkedHashMap<>();
        examples.put("00-readme.txt", README);
        examples.put("smooth-aim.bar", SMOOTH_AIM);
        examples.put("careful-edges.bar", CAREFUL_EDGES);
        examples.put("sprint-discipline.bar", SPRINT_DISCIPLINE);
        examples.put("corner-cutting.bar", CORNER_CUTTING);
        return examples;
    }

    private static final String README =
            "Control scripts\n"
            + "===============\n\n"
            + "Every .bar file in this folder is a stage of Baritone's movement and aim pipeline. Edit one, run\n"
            + "'control script reload' in game, and it takes effect immediately - no rebuild, no restart.\n\n"
            + "A script is a list of assignments evaluated once per tick, in order:\n\n"
            + "    name      what it is called, shown by 'control list'\n"
            + "    priority  when it runs; higher runs later and wins. Built-ins occupy 0 to 1000.\n"
            + "    when      a condition; the script does nothing on ticks where it is false\n"
            + "    let x =   a temporary value you can use further down\n"
            + "    forward = one of the outputs, listed by 'control vars'\n\n"
            + "Outputs it can set:\n"
            + "    forward, strafe   movement vector, -1 to 1, in your own frame (forward, and left)\n"
            + "    jump, sprint, sneak, sneakScale\n"
            + "    yaw, pitch        absolute angles to look at\n"
            + "    yawDelta, pitchDelta   angles relative to where you are looking now\n\n"
            + "Setting only some of them leaves the rest exactly as the previous stage decided.\n\n"
            + "Run 'control vars' for every variable and function with a description, and\n"
            + "'control script eval <expression>' to see what any expression evaluates to right now.\n\n"
            + "Two things a script can never do, however it is written: break a precise aim (the pipeline clamps\n"
            + "every result back inside the tolerance of whatever asked for the rotation), or hang the game (there\n"
            + "are no loops in the language). A script that produces a NaN is disabled and reported.\n\n"
            + "Delete examples.written to have these examples restored.\n";

    private static final String SMOOTH_AIM =
            "# Aim like a hand on a mouse instead of teleporting to the angle.\n"
            + "#\n"
            + "# The view accelerates into a turn and eases out of it, with a little overshoot on big corrections.\n"
            + "# This is the same idea as the built-in humanAim, written in eight lines you can actually tune.\n"
            + "#\n"
            + "# Turn it on by widening the guard: remove the 'precise' test to shape block-breaking aim too (the\n"
            + "# pipeline will still keep it inside tolerance, so it cannot break the interaction).\n"
            + "\n"
            + "name smooth aim\n"
            + "priority 450\n"
            + "when !precise && false          # <- change 'false' to 'true' to enable\n"
            + "\n"
            + "let error     = wrapDegrees(targetYaw - lookYaw)\n"
            + "let urgency   = smoothstep(0, 40, abs(error))\n"
            + "let rate      = lerp(0.25, 0.6, urgency)\n"
            + "let overshoot = abs(error) > 25 ? 1.06 : 1\n"
            + "\n"
            + "yawDelta   = clamp(error * rate * overshoot, -22, 22)\n"
            + "pitchDelta = clamp((targetPitch - lookPitch) * 0.4, -14, 14)\n";

    private static final String CAREFUL_EDGES =
            "# Slow down when there is nothing to stand on just ahead.\n"
            + "#\n"
            + "# solid/standable take coordinates in your own frame: forward, up, left. So standable(1, -1, 0) asks\n"
            + "# 'is there floor one block ahead of me', whichever way I happen to be facing.\n"
            + "#\n"
            + "# This is the kind of thing that was impossible before analog movement existed: the answer is not\n"
            + "# 'stop' or 'go', it is 'go at forty percent'.\n"
            + "\n"
            + "name careful edges\n"
            + "priority 700\n"
            + "when onGround && !flying && false     # <- change 'false' to 'true' to enable\n"
            + "\n"
            + "let floorAhead  = standable(1, -1, 0)\n"
            + "let floorFuther = standable(2, -1, 0)\n"
            + "let exposure    = (1 - floorAhead) + (1 - floorFuther) * 0.5\n"
            + "let caution     = clamp(exposure / 1.5, 0, 1)\n"
            + "\n"
            + "forward = inForward * lerp(1, 0.35, caution)\n"
            + "sprint  = inSprint && caution < 0.3\n"
            + "sneak   = caution > 0.8 && !inJump\n";

    private static final String SPRINT_DISCIPLINE =
            "# Stop sprinting when sprinting is a bad idea.\n"
            + "#\n"
            + "# overrun is how far past its own estimate the current movement has run. Above 1.5 something is going\n"
            + "# wrong, and charging at it faster rarely helps.\n"
            + "\n"
            + "name sprint discipline\n"
            + "priority 600\n"
            + "when false                            # <- change to 'true' to enable\n"
            + "\n"
            + "let hurt    = health < 8\n"
            + "let starved = hunger <= 6\n"
            + "let stuck   = overrun > 1.5\n"
            + "let ledge   = !standable(1, -1, 0)\n"
            + "\n"
            + "sprint = inSprint && !hurt && !starved && !stuck && !ledge\n";

    private static final String CORNER_CUTTING =
            "# Lean into corners instead of turning on the spot.\n"
            + "#\n"
            + "# dx and dz are how far ahead and how far to the left the destination is, so the angle to it is\n"
            + "# atan2(dz, dx). Feeding a fraction of that into strafe makes the bot arc through a turn while still\n"
            + "# walking forward, which is both faster and much less obviously a robot.\n"
            + "\n"
            + "name corner cutting\n"
            + "priority 650\n"
            + "when onGround && speed > 0.08 && false   # <- change 'false' to 'true' to enable\n"
            + "\n"
            + "let heading = deg(atan2(dz, dx))\n"
            + "let lean    = clamp(heading / 60, -1, 1)\n"
            + "\n"
            + "forward = inForward * (1 - 0.25 * abs(lean))\n"
            + "strafe  = clamp(inStrafe + lean * 0.5, -1, 1)\n";
}
