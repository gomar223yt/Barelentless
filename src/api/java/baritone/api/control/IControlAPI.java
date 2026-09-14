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

import java.util.List;

/**
 * The movement and aim pipeline: where every tick's keys, movement vector and rotation are assembled, and where
 * anything can be inserted to change them.
 * <p>
 * The point of routing all control through one place is that "how the bot moves" stops being scattered across forty
 * movement classes and becomes a thing you can hold: register a shaper and it affects every movement, in every
 * process, immediately, without touching pathing code. Unregister it and the behaviour is exactly gone.
 *
 * @author Barelentless
 */
public interface IControlAPI {

    /**
     * A registration handle. Keep it to remove the shaper later; addons should release theirs when they unload.
     */
    interface Registration extends AutoCloseable {

        String name();

        int priority();

        boolean isActive();

        /**
         * Removes the shaper. Idempotent.
         */
        @Override
        void close();
    }

    /**
     * Inserts a movement shaper. Shapers run in ascending priority, so a larger priority runs later and wins.
     *
     * @param name     Shown in diagnostics
     * @param priority Execution order; built-in shapers occupy 0 to 1000
     * @param shaper   The shaper
     * @return A handle that removes it again
     */
    Registration registerInputShaper(String name, int priority, IInputShaper shaper);

    /**
     * Inserts an aim shaper. Same ordering rules as {@link #registerInputShaper}.
     */
    Registration registerRotationShaper(String name, int priority, IRotationShaper shaper);

    List<Registration> inputShapers();

    List<Registration> rotationShapers();

    /**
     * Removes every shaper registered by anyone. Intended for a panic command, not for normal use.
     */
    void clearShapers();

    /**
     * The command that was applied on the most recent tick, after shaping. Never {@code null}.
     */
    MovementCommand lastCommand();

    /**
     * The rotation target that was requested on the most recent tick, or {@code null} if there was none.
     */
    RotationTarget lastRotationTarget();

    /**
     * The rotation that was actually applied on the most recent tick, or {@code null}.
     */
    Rotation lastAppliedRotation();

    /**
     * Requests a movement command for this tick from outside the pathing system. The command goes through the same
     * shaping pipeline as one produced by a movement, and is cleared at the end of the tick.
     * <p>
     * This is how an addon drives the player directly - a custom combat routine, a minigame bot, a script - while
     * still inheriting every registered shaper.
     *
     * @param command The desired command
     * @param priority Higher priority overrides a lower one requested in the same tick
     */
    void requestCommand(MovementCommand command, int priority);

    /**
     * Requests a rotation for this tick, competing with any other request by priority.
     */
    void requestRotation(RotationTarget target);

    /**
     * Whether analog movement is honoured. When off, a command's analog vector is reduced to the eight vanilla
     * directions - which is what a server with strict movement checks may require.
     */
    boolean isAnalogMovementEnabled();

    void setAnalogMovementEnabled(boolean enabled);

    /**
     * A human readable trace of what each stage did to this tick's command and rotation. Populated only while
     * tracing is enabled, since building it every tick is wasteful.
     */
    List<String> lastTrace();

    void setTracing(boolean tracing);

    boolean isTracing();
}
