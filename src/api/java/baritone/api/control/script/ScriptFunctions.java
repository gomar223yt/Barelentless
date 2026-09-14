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

package baritone.api.control.script;

import java.util.Random;

/**
 * The standard function library available to every control script.
 * <p>
 * Chosen for what writing a movement by hand actually needs rather than for mathematical completeness. {@code clamp},
 * {@code lerp}, {@code smoothstep} and {@code approach} are here because almost every hand-written controller is
 * built out of them; {@code wrapDegrees} is here because getting angle wrapping wrong is the single most common bug
 * in aim code, and it should not be something each script has to rediscover.
 *
 * @author Barelentless
 */
public final class ScriptFunctions {

    private static final Random RANDOM = new Random();

    private ScriptFunctions() {}

    /**
     * Adds every standard function to a context.
     */
    public static ScriptContext install(ScriptContext context) {
        context.function("abs", 1, "abs(x)", "absolute value", a -> Math.abs(a[0]));
        context.function("sign", 1, "sign(x)", "-1, 0 or 1", a -> Math.signum(a[0]));
        context.function("floor", 1, "floor(x)", "round down", a -> Math.floor(a[0]));
        context.function("ceil", 1, "ceil(x)", "round up", a -> Math.ceil(a[0]));
        context.function("round", 1, "round(x)", "round to nearest", a -> Math.round(a[0]));
        context.function("sqrt", 1, "sqrt(x)", "square root", a -> Math.sqrt(a[0]));
        context.function("exp", 1, "exp(x)", "e to the power x", a -> Math.exp(a[0]));
        context.function("log", 1, "log(x)", "natural logarithm", a -> Math.log(a[0]));

        context.function("min", 2, "min(a, b)", "smaller of two", a -> Math.min(a[0], a[1]));
        context.function("max", 2, "max(a, b)", "larger of two", a -> Math.max(a[0], a[1]));
        context.function("clamp", 3, "clamp(x, low, high)", "keep x within a range",
                a -> Math.max(a[1], Math.min(a[2], a[0])));
        context.function("hypot", 2, "hypot(a, b)", "length of a right triangle's hypotenuse",
                a -> Math.sqrt(a[0] * a[0] + a[1] * a[1]));
        context.function("pow", 2, "pow(x, y)", "x to the power y", a -> Math.pow(a[0], a[1]));
        context.function("mod", 2, "mod(x, y)", "remainder, always non-negative",
                a -> ((a[0] % a[1]) + a[1]) % a[1]);

        context.function("sin", 1, "sin(radians)", "sine", a -> Math.sin(a[0]));
        context.function("cos", 1, "cos(radians)", "cosine", a -> Math.cos(a[0]));
        context.function("tan", 1, "tan(radians)", "tangent", a -> Math.tan(a[0]));
        context.function("asin", 1, "asin(x)", "inverse sine, in radians", a -> Math.asin(a[0]));
        context.function("acos", 1, "acos(x)", "inverse cosine, in radians", a -> Math.acos(a[0]));
        context.function("atan", 1, "atan(x)", "inverse tangent, in radians", a -> Math.atan(a[0]));
        context.function("atan2", 2, "atan2(y, x)", "angle of a vector, in radians", a -> Math.atan2(a[0], a[1]));
        context.function("deg", 1, "deg(radians)", "radians to degrees", a -> Math.toDegrees(a[0]));
        context.function("rad", 1, "rad(degrees)", "degrees to radians", a -> Math.toRadians(a[0]));

        context.function("lerp", 3, "lerp(a, b, t)", "blend from a to b as t goes 0 to 1",
                a -> a[0] + (a[1] - a[0]) * a[2]);
        context.function("step", 2, "step(edge, x)", "1 when x is past edge, otherwise 0",
                a -> a[1] >= a[0] ? 1 : 0);
        context.function("smoothstep", 3, "smoothstep(low, high, x)",
                "0 below low, 1 above high, smooth in between", a -> {
                    double t = Math.max(0, Math.min(1, (a[2] - a[0]) / (a[1] - a[0])));
                    return t * t * (3 - 2 * t);
                });
        context.function("approach", 3, "approach(current, target, rate)",
                "move current towards target by at most rate", a -> {
                    double difference = a[1] - a[0];
                    double limit = Math.abs(a[2]);
                    return a[0] + Math.max(-limit, Math.min(limit, difference));
                });
        context.function("wrapDegrees", 1, "wrapDegrees(angle)",
                "fold an angle into -180 to 180, which is what makes yaw arithmetic work", a -> {
                    double wrapped = a[0] % 360;
                    if (wrapped >= 180) {
                        wrapped -= 360;
                    }
                    if (wrapped < -180) {
                        wrapped += 360;
                    }
                    return wrapped;
                });
        context.function("select", 3, "select(condition, whenTrue, whenFalse)",
                "the value of one of two branches; both are evaluated, unlike ? :",
                a -> a[0] != 0 ? a[1] : a[2]);

        context.function("random", 0, 0, "random()", "a value in [0, 1)", (frame, a) -> RANDOM.nextDouble());
        context.function("noise", 1, "noise(x)",
                "a smooth pseudo-random value in [-1, 1], the same for the same x", a -> {
                    // value noise: hash the integer lattice and interpolate, so it is smooth and repeatable
                    double floor = Math.floor(a[0]);
                    double t = a[0] - floor;
                    double smooth = t * t * (3 - 2 * t);
                    return hash((long) floor) * (1 - smooth) + hash((long) floor + 1) * smooth;
                });
        return context;
    }

    private static double hash(long value) {
        long x = value * 0x9E3779B97F4A7C15L;
        x ^= x >>> 29;
        x *= 0xBF58476D1CE4E5B9L;
        x ^= x >>> 32;
        // 53 significant bits shifted out, so the divisor is 2^53 to land in [0, 1) before the rescale;
        // dividing by 2^52 would quietly produce [-1, 3] and a "random" spread biased upward
        return (x >>> 11) / (double) (1L << 53) * 2 - 1;
    }
}
