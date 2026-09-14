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

import baritone.api.IBaritone;
import baritone.api.pathing.movement.IMovement;
import baritone.api.utils.IPlayerContext;
import baritone.api.utils.Rotation;

/**
 * Everything a shaper is told about the moment it is being asked to act on.
 * <p>
 * Shapers are called from inside the tick, so this carries the already-computed facts rather than making every shaper
 * recompute them: where the player is and how fast, what movement the path executor is currently running and how long
 * it has been trying, what was decided last tick. A shaper that needs more can reach the whole game through
 * {@link #baritone()}.
 *
 * @author Barelentless
 */
public final class ControlContext {

    private final IBaritone baritone;
    private final long tick;
    private final IMovement movement;
    private final int movementTicks;
    private final MovementCommand previousCommand;
    private final Rotation previousRotation;
    private final double speed;
    private final boolean onGround;

    public ControlContext(IBaritone baritone, long tick, IMovement movement, int movementTicks,
                          MovementCommand previousCommand, Rotation previousRotation,
                          double speed, boolean onGround) {
        this.baritone = baritone;
        this.tick = tick;
        this.movement = movement;
        this.movementTicks = movementTicks;
        this.previousCommand = previousCommand;
        this.previousRotation = previousRotation;
        this.speed = speed;
        this.onGround = onGround;
    }

    public IBaritone baritone() {
        return this.baritone;
    }

    public IPlayerContext player() {
        return this.baritone.getPlayerContext();
    }

    /**
     * A monotonically increasing tick counter for this Baritone instance. Shapers that need a phase - jitter,
     * periodic corrections, anything that should not repeat identically - should key off this rather than wall clock.
     */
    public long tick() {
        return this.tick;
    }

    /**
     * The movement currently being executed, or {@code null} when the bot is not following a path. Shapers use this
     * to behave differently per movement kind without having to guess from geometry.
     */
    public IMovement movement() {
        return this.movement;
    }

    /**
     * How many ticks the current movement has been running. A movement that has taken three times its estimate is a
     * movement in trouble, and that is exactly when a shaper should consider doing something different.
     */
    public int movementTicks() {
        return this.movementTicks;
    }

    /**
     * The command produced last tick, after all shaping. Never {@code null}; an idle command when there was none.
     */
    public MovementCommand previousCommand() {
        return this.previousCommand;
    }

    /**
     * The rotation actually applied last tick, or {@code null} if none was.
     */
    public Rotation previousRotation() {
        return this.previousRotation;
    }

    /**
     * Horizontal speed in blocks per tick.
     */
    public double speed() {
        return this.speed;
    }

    public boolean onGround() {
        return this.onGround;
    }

    /**
     * True when the bot is executing a path. A shaper registered globally will also be called while the player moves
     * on their own, and usually wants to check this first.
     */
    public boolean isPathing() {
        return this.baritone.getPathingBehavior().isPathing();
    }
}
