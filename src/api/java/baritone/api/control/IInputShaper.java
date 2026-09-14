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

/**
 * A stage in the movement pipeline. Receives the command as the previous stage left it and may change anything about
 * it before it reaches the game.
 * <p>
 * This is the extension point for everything from a two-line tweak ("never sprint while holding a bucket") to a
 * learned controller that overwrites the movement vector entirely. Shapers run in ascending priority order, so a
 * shaper with a high priority sees, and can override, the decisions of everything below it.
 *
 * @author Barelentless
 */
@FunctionalInterface
public interface IInputShaper {

    /**
     * Adjusts the command for this tick. Called once per tick while this shaper is registered, whether or not the bot
     * is pathing - check {@link ControlContext#isPathing()} if that matters.
     *
     * @param context The tick's context
     * @param command The command so far, to be modified in place
     */
    void shape(ControlContext context, MovementCommand command);

    /**
     * A name for diagnostics. Shows up in {@code control list} and in the per-tick trace.
     */
    default String name() {
        return getClass().getSimpleName();
    }

    /**
     * Whether this shaper should run at all this tick. Cheaper than checking inside {@link #shape} because a
     * disabled shaper is skipped without building anything.
     */
    default boolean isEnabled() {
        return true;
    }
}
