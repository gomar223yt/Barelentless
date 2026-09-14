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

package baritone.api.utils;

import baritone.api.behavior.IBehavior;
import baritone.api.control.MovementCommand;
import baritone.api.utils.input.Input;

/**
 * @author Brady
 * @since 11/12/2018
 */
public interface IInputOverrideHandler extends IBehavior {

    boolean isInputForcedDown(Input input);

    void setInputForceState(Input input, boolean forced);

    void clearAllKeys();

    /**
     * Sets the shaped {@link MovementCommand} to apply for the current tick. Set by the control pipeline; anything
     * else writing here will simply be overwritten on the next tick.
     *
     * @param command The command, or {@code null} to fall back to plain key state
     */
    void setActiveCommand(MovementCommand command);

    /**
     * @return The command being applied this tick, or {@code null} if movement is running on key state alone
     */
    MovementCommand getActiveCommand();
}
