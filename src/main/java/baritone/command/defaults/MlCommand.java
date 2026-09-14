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

package baritone.command.defaults;

import baritone.Baritone;
import baritone.api.IBaritone;
import baritone.api.command.Command;
import baritone.api.command.argument.IArgConsumer;
import baritone.api.command.exception.CommandException;
import baritone.api.command.exception.CommandInvalidStateException;
import baritone.api.ml.MlDiagnostics;
import baritone.api.ml.memory.MemoryRecord;
import baritone.ml.MlManager;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

/**
 * Inspects and controls the learning subsystem.
 * <p>
 * Every part of this exists because a learned controller you cannot interrogate is one you cannot debug, and a
 * learned controller you cannot switch off is one you cannot ship. Status shows what the models are and what they
 * have seen; selftest proves the maths still works on this machine; memory shows the specific situations the bot
 * believes it knows about; reset throws all of it away.
 *
 * @author Barelentless
 */
public class MlCommand extends Command {

    public MlCommand(IBaritone baritone) {
        super(baritone, "ml");
    }

    @Override
    public void execute(String label, IArgConsumer args) throws CommandException {
        MlManager manager = ((Baritone) this.baritone).getMlManager();
        if (!args.hasAny()) {
            manager.status().forEach(this::logDirect);
            return;
        }
        String action = args.getString().toLowerCase();
        switch (action) {
            case "status": {
                args.requireMax(0);
                manager.status().forEach(this::logDirect);
                break;
            }
            case "start": {
                args.requireMax(0);
                Baritone.settings().mlEnabled.value = true;
                manager.start();
                logDirect("Learning started. Recording is on; the bot's behaviour only changes once mlAim or "
                        + "mlLearnedCosts is enabled too.");
                break;
            }
            case "stop": {
                args.requireMax(0);
                manager.stop();
                Baritone.settings().mlEnabled.value = false;
                logDirect("Learning stopped and saved.");
                break;
            }
            case "save": {
                args.requireMax(0);
                manager.save();
                logDirect("Saved models and memory.");
                break;
            }
            case "reset": {
                String confirmation = args.hasAny() ? args.getString() : "";
                if (!confirmation.equals("confirm")) {
                    throw new CommandInvalidStateException(
                            "This erases every trained weight and remembered situation. Run 'ml reset confirm'.");
                }
                manager.reset();
                logDirect("Everything learned has been discarded.");
                break;
            }
            case "selftest": {
                args.requireMax(0);
                logDirect("Running learning self-checks, this takes a moment...");
                int failed = 0;
                for (MlDiagnostics.Result result : MlDiagnostics.runAll()) {
                    logDirect(result.toString());
                    if (!result.passed) {
                        failed++;
                    }
                }
                logDirect(failed == 0 ? "All checks passed." : failed + " checks FAILED.");
                break;
            }
            case "memory": {
                int count = args.hasAny() ? args.getAs(Integer.class) : 10;
                List<MemoryRecord> records = manager.getMemory().records();
                records.sort(Comparator.comparingDouble((MemoryRecord record) -> -record.getVisits()));
                logDirect(records.size() + " remembered situations, showing the " + Math.min(count, records.size())
                        + " most visited:");
                records.stream().limit(count).forEach(record -> logDirect("  " + record));
                break;
            }
            case "consolidate": {
                args.requireMax(0);
                int removed = manager.getMemory().consolidate(System.currentTimeMillis(), 2);
                logDirect("Consolidated memory, removed " + removed + " records, "
                        + manager.getMemory().size() + " remain.");
                break;
            }
            default:
                throw new CommandInvalidStateException("Unknown action '" + action + "'");
        }
    }

    @Override
    public Stream<String> tabComplete(String label, IArgConsumer args) {
        if (args.hasExactlyOne()) {
            return new baritone.api.command.helpers.TabCompleteHelper()
                    .append("status", "start", "stop", "save", "reset", "selftest", "memory", "consolidate")
                    .filterPrefix(args.getString())
                    .stream();
        }
        return Stream.empty();
    }

    @Override
    public String getShortDesc() {
        return "Inspect and control learning";
    }

    @Override
    public List<String> getLongDesc() {
        return Arrays.asList(
                "Baritone can learn from what actually happens when it moves: how long movements really take, what",
                "they cost in damage, and how a human moves the view. What it learns lives in the baritone/ml folder.",
                "",
                "Nothing is recorded and no model runs until mlEnabled is on, and recording alone changes no",
                "behaviour - a trained model only affects the bot once mlAim or mlLearnedCosts is on as well.",
                "",
                "Usage:",
                "> ml - Show what the models are and what they have seen",
                "> ml start - Enable learning and start the background trainer",
                "> ml stop - Stop training and save",
                "> ml save - Write models and memory to disk now",
                "> ml selftest - Verify every gradient and layer on this machine",
                "> ml memory [n] - Show the situations the bot has remembered most often",
                "> ml consolidate - Merge duplicate memories and drop ones that never mattered",
                "> ml reset confirm - Discard everything learned"
        );
    }
}
