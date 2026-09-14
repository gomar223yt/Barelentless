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

package baritone.ml;

import baritone.Baritone;
import baritone.api.control.ControlContext;
import baritone.api.control.IInputShaper;
import baritone.api.control.MovementCommand;
import baritone.api.ml.memory.EpisodicMemory;
import baritone.api.pathing.movement.IMovement;
import baritone.api.utils.BetterBlockPos;
import baritone.api.utils.input.Input;
import baritone.utils.BlockStateInterface;

/**
 * Moves carefully where experience says care is warranted.
 * <p>
 * This is the other half of what the episodic memory is for. Adjusting path costs changes which route is chosen;
 * this changes how the chosen route is walked. If a particular kind of movement in a particular kind of place has a
 * history of taking far longer than estimated, or of failing outright, the bot approaches it slower and without
 * sprinting - the same thing a player does at a jump they have missed before.
 * <p>
 * Nothing here is a rule about jumps or ledges or ice. The behaviour comes entirely from the recorded outcomes, so a
 * situation nobody anticipated - a modded block that is slippery in an unusual way, a redstone contraption that
 * intermittently blocks a doorway - produces caution for exactly the same reason a vanilla one does.
 *
 * @author Barelentless
 */
public final class LearnedCautionShaper implements IInputShaper {

    private final MlManager manager;

    private double lastCaution;
    private String lastReason = "";

    public LearnedCautionShaper(MlManager manager) {
        this.manager = manager;
    }

    @Override
    public String name() {
        return "learnedCaution";
    }

    @Override
    public boolean isEnabled() {
        return Baritone.settings().mlEnabled.value && Baritone.settings().mlCaution.value;
    }

    @Override
    public void shape(ControlContext context, MovementCommand command) {
        IMovement movement = context.movement();
        if (movement == null || command.isIdle()) {
            this.lastCaution = 0;
            return;
        }
        BlockStateInterface bsi = ((Baritone) context.baritone()).bsi;
        if (bsi == null) {
            return;
        }
        BetterBlockPos source = movement.getSrc();
        BetterBlockPos destination = movement.getDest();
        if (!bsi.worldContainsLoadedChunk(destination.x, destination.z)) {
            return;
        }
        EpisodicMemory.Estimate estimate;
        try {
            float[] key = LearnedCostAdjuster.key(
                    destination.y - source.y,
                    bsi.get0(source.x, source.y - 1, source.z),
                    bsi.get0(destination.x, destination.y, destination.z),
                    bsi.get0(destination.x, destination.y - 1, destination.z),
                    bsi.get0(destination.x, destination.y + 1, destination.z));
            estimate = this.manager.getMemory().estimate(key, movementKind(movement), 8);
        } catch (RuntimeException e) {
            this.manager.onInferenceFailure("caution", e);
            return;
        }
        if (estimate.isEmpty()) {
            this.lastCaution = 0;
            this.lastReason = "";
            return;
        }

        // two independent reasons to be careful: it has failed here before, and it has been unpredictable here before
        double failureRisk = Math.max(0, 0.95 - estimate.successRate);
        double unpredictability = Math.min(1, estimate.deviation);
        double caution = Math.min(1, (failureRisk * 2 + unpredictability * 0.5) * estimate.confidence
                * Math.max(0, Math.min(1, Baritone.settings().mlCautionStrength.value)));
        this.lastCaution = caution;
        if (caution < 0.05) {
            this.lastReason = "";
            return;
        }
        this.lastReason = String.format("success %.0f%%, spread %.2f, confidence %.2f",
                estimate.successRate * 100, estimate.deviation, estimate.confidence);

        if (caution > 0.25) {
            command.setSprintAllowed(false);
        }
        // never below a third speed: crawling into a movement can fail it just as surely as charging into one, and
        // the bot still has to get there
        command.scaleAnalog((float) (1 - caution * 0.66));
        if (caution > 0.6 && !command.isPressed(Input.JUMP)) {
            // at this point the bot is close to expecting a fall; sneaking off an edge is survivable, walking is not
            command.set(Input.SNEAK, true);
            command.setSneakScale(1f);
        }
    }

    private static int movementKind(IMovement movement) {
        return MovementKinds.of(movement);
    }

    /**
     * How much caution was applied on the last tick, in {@code [0, 1]}, and why. For the status readout.
     */
    public String describe() {
        if (this.lastCaution < 0.05) {
            return "none";
        }
        return String.format("%.0f%% (%s)", this.lastCaution * 100, this.lastReason);
    }
}
