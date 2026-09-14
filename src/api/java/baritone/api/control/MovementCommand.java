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

import baritone.api.utils.input.Input;

import java.util.EnumMap;
import java.util.EnumSet;
import java.util.Map;
import java.util.Set;

/**
 * The movement the bot intends to perform on a single tick, in a form that can still be edited.
 * <p>
 * Vanilla movement input is eight booleans, and that is all Baritone has ever been able to express: full speed in one
 * of eight directions, or nothing. Real movement is not like that. Walking a one-block ledge wants forty percent
 * forward, not a stutter of on and off; lining up a parkour jump wants a specific approach speed; a strafe around a
 * corner wants both axes at once at different magnitudes. This class carries the discrete keys for the code that
 * thinks in keys, and an optional analog vector for the code that does not, and the two are reconciled when the
 * command is applied.
 * <p>
 * A command is built by whoever is driving movement (a path executor, a process, a learned policy), then handed
 * through every registered {@link IInputShaper} in priority order before it reaches the game. Shapers see what the
 * previous stage decided and may change any part of it.
 *
 * @author Barelentless
 */
public final class MovementCommand {

    private final Map<Input, Boolean> keys = new EnumMap<>(Input.class);

    private Float forwardImpulse;
    private Float strafeImpulse;
    private float sneakScale = 0.3f;
    private boolean analogEnabled;
    private boolean sprintAllowed = true;
    private String source = "";

    public MovementCommand() {}

    public MovementCommand(MovementCommand copyOf) {
        this.keys.putAll(copyOf.keys);
        this.forwardImpulse = copyOf.forwardImpulse;
        this.strafeImpulse = copyOf.strafeImpulse;
        this.sneakScale = copyOf.sneakScale;
        this.analogEnabled = copyOf.analogEnabled;
        this.sprintAllowed = copyOf.sprintAllowed;
        this.source = copyOf.source;
    }

    // ------------------------------------------------------------------------------------------------------- keys

    public MovementCommand set(Input input, boolean pressed) {
        this.keys.put(input, pressed);
        return this;
    }

    public MovementCommand press(Input input) {
        return set(input, true);
    }

    public MovementCommand release(Input input) {
        return set(input, false);
    }

    public boolean isPressed(Input input) {
        return input != null && Boolean.TRUE.equals(this.keys.get(input));
    }

    public Set<Input> pressed() {
        Set<Input> set = EnumSet.noneOf(Input.class);
        for (Map.Entry<Input, Boolean> entry : this.keys.entrySet()) {
            if (entry.getValue()) {
                set.add(entry.getKey());
            }
        }
        return set;
    }

    public MovementCommand clearKeys() {
        this.keys.clear();
        return this;
    }

    // ----------------------------------------------------------------------------------------------------- analog

    /**
     * Sets a continuous movement vector in the player's local frame, each component in {@code [-1, 1]}: forward is
     * positive, strafing left is positive, matching the game's own impulse convention.
     * <p>
     * Setting this enables analog mode for the tick, and the discrete direction keys are derived from the sign of
     * each axis so that anything reading key state still sees something sensible.
     */
    public MovementCommand setAnalog(float forward, float strafe) {
        this.forwardImpulse = clamp(forward);
        this.strafeImpulse = clamp(strafe);
        this.analogEnabled = true;
        this.keys.put(Input.MOVE_FORWARD, this.forwardImpulse > 0.05f);
        this.keys.put(Input.MOVE_BACK, this.forwardImpulse < -0.05f);
        this.keys.put(Input.MOVE_LEFT, this.strafeImpulse > 0.05f);
        this.keys.put(Input.MOVE_RIGHT, this.strafeImpulse < -0.05f);
        return this;
    }

    /**
     * Scales the analog vector, keeping its direction. The natural way for a shaper to say "same heading, slower" -
     * approaching a ledge, easing into a jump, holding back to let a mob pass.
     */
    public MovementCommand scaleAnalog(float factor) {
        if (this.analogEnabled) {
            return setAnalog(this.forwardImpulse * factor, this.strafeImpulse * factor);
        }
        // promote the current key state into an analog vector first, so this works on a purely discrete command too
        float forward = (isPressed(Input.MOVE_FORWARD) ? 1f : 0f) - (isPressed(Input.MOVE_BACK) ? 1f : 0f);
        float strafe = (isPressed(Input.MOVE_LEFT) ? 1f : 0f) - (isPressed(Input.MOVE_RIGHT) ? 1f : 0f);
        return setAnalog(forward * factor, strafe * factor);
    }

    /**
     * Drops any analog vector, returning to plain key control.
     */
    public MovementCommand clearAnalog() {
        this.analogEnabled = false;
        this.forwardImpulse = null;
        this.strafeImpulse = null;
        return this;
    }

    public boolean isAnalog() {
        return this.analogEnabled;
    }

    /**
     * The forward impulse that will be applied, derived from the keys when not in analog mode.
     */
    public float getForwardImpulse() {
        if (this.analogEnabled) {
            return this.forwardImpulse;
        }
        return (isPressed(Input.MOVE_FORWARD) ? 1f : 0f) - (isPressed(Input.MOVE_BACK) ? 1f : 0f);
    }

    /**
     * The strafe impulse that will be applied, positive to the left.
     */
    public float getStrafeImpulse() {
        if (this.analogEnabled) {
            return this.strafeImpulse;
        }
        return (isPressed(Input.MOVE_LEFT) ? 1f : 0f) - (isPressed(Input.MOVE_RIGHT) ? 1f : 0f);
    }

    /**
     * How much sneaking scales the movement vector. Vanilla uses 0.3; exposing it lets a shaper model the slower
     * approach of a careful player, or keep full speed while sneaking for a tick to cancel fall damage.
     */
    public MovementCommand setSneakScale(float sneakScale) {
        this.sneakScale = Math.max(0f, Math.min(1f, sneakScale));
        return this;
    }

    public float getSneakScale() {
        return this.sneakScale;
    }

    /**
     * Whether sprinting may be enabled this tick. A shaper can veto sprint without having to fight whatever else
     * wants to set the sprint key.
     */
    public MovementCommand setSprintAllowed(boolean sprintAllowed) {
        this.sprintAllowed = sprintAllowed;
        return this;
    }

    public boolean isSprintAllowed() {
        return this.sprintAllowed;
    }

    public String getSource() {
        return this.source;
    }

    public MovementCommand setSource(String source) {
        this.source = source;
        return this;
    }

    /**
     * The magnitude of the movement vector, before the sneak scale is applied.
     */
    public float magnitude() {
        float forward = getForwardImpulse();
        float strafe = getStrafeImpulse();
        return (float) Math.sqrt(forward * forward + strafe * strafe);
    }

    public boolean isIdle() {
        return magnitude() < 1e-4f && !isPressed(Input.JUMP) && !isPressed(Input.SNEAK);
    }

    private static float clamp(float value) {
        if (Float.isNaN(value)) {
            return 0f;
        }
        return Math.max(-1f, Math.min(1f, value));
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(this.analogEnabled ? "analog" : "keys").append('(');
        sb.append(String.format("%.2f, %.2f", getForwardImpulse(), getStrafeImpulse())).append(')');
        for (Input input : pressed()) {
            if (input != Input.MOVE_FORWARD && input != Input.MOVE_BACK
                    && input != Input.MOVE_LEFT && input != Input.MOVE_RIGHT) {
                sb.append(' ').append(input.name().toLowerCase());
            }
        }
        if (!this.sprintAllowed) {
            sb.append(" no-sprint");
        }
        if (!this.source.isEmpty()) {
            sb.append(" <- ").append(this.source);
        }
        return sb.toString();
    }
}
