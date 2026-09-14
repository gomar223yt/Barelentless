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

package baritone.utils;

import baritone.api.control.MovementCommand;
import baritone.api.utils.input.Input;
import net.minecraft.client.player.ClientInput;
import net.minecraft.world.phys.Vec2;

public class PlayerMovementInput extends ClientInput {

    private final InputOverrideHandler handler;

    PlayerMovementInput(InputOverrideHandler handler) {
        this.handler = handler;
    }

    @Override
    public void tick() {
        MovementCommand command = handler.getActiveCommand();
        if (command != null && command.isAnalog()) {
            tickAnalog(command);
            return;
        }
        float leftImpulse = 0.0F;
        float forwardImpulse = 0.0F;
        boolean jumping = handler.isInputForcedDown(Input.JUMP); // oppa gangnam style

        boolean up = handler.isInputForcedDown(Input.MOVE_FORWARD);
        if (up) {
            forwardImpulse++;
        }

        boolean down = handler.isInputForcedDown(Input.MOVE_BACK);
        if (down) {
            forwardImpulse--;
        }

        boolean left = handler.isInputForcedDown(Input.MOVE_LEFT);
        if (left) {
            leftImpulse++;
        }

        boolean right = handler.isInputForcedDown(Input.MOVE_RIGHT);
        if (right) {
            leftImpulse--;
        }

        boolean sneaking = handler.isInputForcedDown(Input.SNEAK);
        if (sneaking) {
            leftImpulse *= 0.3D;
            forwardImpulse *= 0.3D;
        }
        this.moveVector = new Vec2(leftImpulse, forwardImpulse);

        boolean sprinting = handler.isInputForcedDown(Input.SPRINT);

        this.keyPresses = new net.minecraft.world.entity.player.Input(up, down, left, right, jumping, sneaking, sprinting);
    }

    /**
     * Applies a command that carries a continuous movement vector.
     * <p>
     * The key presses are still reported honestly - the game, and anything mixing into it, sees forward held when the
     * bot is moving forward - but the impulse vector is taken verbatim instead of being rebuilt from those booleans.
     * That is the whole difference between "walk forward" and "walk forward at forty percent", which is what makes
     * ledges, precise jump approaches and smooth cornering expressible at all.
     */
    private void tickAnalog(MovementCommand command) {
        float forwardImpulse = command.getForwardImpulse();
        float leftImpulse = command.getStrafeImpulse();
        boolean sneaking = command.isPressed(Input.SNEAK);
        if (sneaking) {
            float scale = command.getSneakScale();
            leftImpulse *= scale;
            forwardImpulse *= scale;
        }
        this.moveVector = new Vec2(leftImpulse, forwardImpulse);
        this.keyPresses = new net.minecraft.world.entity.player.Input(
                command.isPressed(Input.MOVE_FORWARD),
                command.isPressed(Input.MOVE_BACK),
                command.isPressed(Input.MOVE_LEFT),
                command.isPressed(Input.MOVE_RIGHT),
                command.isPressed(Input.JUMP),
                sneaking,
                command.isSprintAllowed() && command.isPressed(Input.SPRINT)
        );
    }
}
