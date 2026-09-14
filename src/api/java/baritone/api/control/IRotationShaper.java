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

package baritone.api.control;

import baritone.api.utils.Rotation;

/**
 * A stage in the aim pipeline: given where the bot is looking and where it wants to look, decide where it looks now.
 * <p>
 * Shapers run in ascending priority order, each receiving the previous stage's output, and the result is clamped back
 * inside the target's tolerance afterwards, so no shaper can break a precise interaction. Whatever survives the chain
 * is then passed through the aim processor, which applies mouse quantization - a shaper works in angles and does not
 * have to know about sensitivity or GCD.
 *
 * @author Barelentless
 */
@FunctionalInterface
public interface IRotationShaper {

    /**
     * @param context The tick's context
     * @param target  What was asked for, and why
     * @param current Where the chain has decided to look so far; starts as the target's own rotation
     * @return Where to look. Returning {@code current} unchanged is a valid no-op.
     */
    Rotation shape(ControlContext context, RotationTarget target, Rotation current);

    default String name() {
        return getClass().getSimpleName();
    }

    default boolean isEnabled() {
        return true;
    }

    /**
     * Whether this shaper wants to run for a target of the given purpose. The default skips precise targets, which is
     * the right default for anything cosmetic: a shaper that only exists to make aim look human has no business
     * touching the rotation that a block break depends on.
     */
    default boolean appliesTo(RotationTarget.Purpose purpose) {
        return !purpose.isPrecise();
    }
}
