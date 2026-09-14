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
import baritone.api.command.helpers.TabCompleteHelper;
import baritone.api.control.IControlAPI;
import baritone.api.control.script.ScriptContext;
import baritone.api.control.script.ScriptException;
import baritone.control.script.ScriptBindings;
import baritone.control.script.ScriptManager;
import baritone.control.script.ScriptedShaper;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

/**
 * Inspects the movement and aim pipeline.
 *
 * @author Barelentless
 */
public class ControlCommand extends Command {

    public ControlCommand(IBaritone baritone) {
        super(baritone, "control");
    }

    @Override
    public void execute(String label, IArgConsumer args) throws CommandException {
        IControlAPI control = this.baritone.getControlAPI();
        String action = args.hasAny() ? args.getString().toLowerCase() : "status";
        switch (action) {
            case "status": {
                args.requireMax(0);
                logDirect("command: " + control.lastCommand());
                logDirect("aim target: " + (control.lastRotationTarget() == null ? "none" : control.lastRotationTarget()));
                logDirect("aim applied: " + (control.lastAppliedRotation() == null ? "none" : control.lastAppliedRotation()));
                logDirect("analog movement: " + (control.isAnalogMovementEnabled() ? "on" : "off"));
                break;
            }
            case "list": {
                args.requireMax(0);
                logDirect("Movement shapers, in the order they run:");
                if (control.inputShapers().isEmpty()) {
                    logDirect("  (none)");
                }
                for (IControlAPI.Registration registration : control.inputShapers()) {
                    logDirect("  " + registration.priority() + "  " + registration.name());
                }
                logDirect("Aim shapers, in the order they run:");
                if (control.rotationShapers().isEmpty()) {
                    logDirect("  (none)");
                }
                for (IControlAPI.Registration registration : control.rotationShapers()) {
                    logDirect("  " + registration.priority() + "  " + registration.name());
                }
                break;
            }
            case "trace": {
                args.requireMax(0);
                if (!control.isTracing()) {
                    control.setTracing(true);
                    logDirect("Tracing enabled; run 'control trace' again to see the last tick.");
                    break;
                }
                List<String> trace = control.lastTrace();
                if (trace.isEmpty()) {
                    logDirect("Nothing traced yet.");
                }
                trace.forEach(line -> logDirect("  " + line));
                break;
            }
            case "untrace": {
                args.requireMax(0);
                control.setTracing(false);
                logDirect("Tracing disabled.");
                break;
            }
            case "costs": {
                args.requireMax(0);
                logDirect("Registered path cost adjusters, in the order they run:");
                if (this.baritone.getCostRegistry().registrations().isEmpty()) {
                    logDirect("  (none)");
                }
                for (baritone.api.pathing.calc.ICostRegistry.Registration registration
                        : this.baritone.getCostRegistry().registrations()) {
                    logDirect("  " + registration.name());
                }
                long[] statistics = this.baritone.getCostRegistry().lastCalculationStatistics();
                logDirect("Last calculation: adjusted " + statistics[0] + " of " + statistics[1] + " candidates");
                break;
            }
            case "vars": {
                args.requireMax(0);
                ScriptContext vocabulary = new ScriptBindings().getContext();
                logDirect("Variables a script can read, and (marked =) write:");
                vocabulary.describeVariables().forEach(line -> logDirect("  " + line));
                logDirect("Functions:");
                vocabulary.describeFunctions().forEach(line -> logDirect("  " + line));
                break;
            }
            case "script": {
                ScriptManager scripts = ((Baritone) this.baritone).getScriptManager();
                String sub = args.hasAny() ? args.getString().toLowerCase() : "list";
                switch (sub) {
                    case "list": {
                        args.requireMax(0);
                        logDirect("Scripts in " + scripts.getDirectory() + ":");
                        scripts.describe().forEach(line -> logDirect("  " + line));
                        break;
                    }
                    case "reload": {
                        args.requireMax(0);
                        List<String> report = scripts.reload();
                        report.forEach(this::logDirect);
                        logDirect(scripts.size() + " script" + (scripts.size() == 1 ? "" : "s") + " active");
                        break;
                    }
                    case "unload": {
                        args.requireMax(0);
                        scripts.unloadAll();
                        logDirect("All scripts removed from the pipeline; the files are untouched.");
                        break;
                    }
                    case "enable":
                    case "disable": {
                        String scriptName = args.getString();
                        ScriptedShaper shaper = scripts.find(scriptName);
                        if (shaper == null) {
                            throw new CommandInvalidStateException("no script called '" + scriptName + "'");
                        }
                        shaper.setEnabled(sub.equals("enable"));
                        logDirect(shaper.name() + " is now " + (shaper.isEnabled() ? "enabled" : "disabled"));
                        break;
                    }
                    case "show": {
                        String scriptName = args.getString();
                        ScriptedShaper shaper = scripts.find(scriptName);
                        if (shaper == null) {
                            throw new CommandInvalidStateException("no script called '" + scriptName + "'");
                        }
                        logDirect(shaper.describe());
                        for (String line : shaper.getScript().getSource().split("\n")) {
                            logDirect("  " + line);
                        }
                        break;
                    }
                    case "eval": {
                        String expression = args.rawRest();
                        if (expression.isEmpty()) {
                            throw new CommandInvalidStateException("give an expression to evaluate");
                        }
                        try {
                            logDirect(expression + " = " + scripts.evaluate(expression));
                        } catch (ScriptException e) {
                            for (String line : e.describe().split("\n")) {
                                logDirect(line);
                            }
                        }
                        break;
                    }
                    default:
                        throw new CommandInvalidStateException("unknown script action '" + sub + "'");
                }
                break;
            }
            case "remove": {
                String name = args.getString();
                boolean removed = false;
                for (IControlAPI.Registration registration : control.inputShapers()) {
                    if (registration.name().equalsIgnoreCase(name)) {
                        registration.close();
                        removed = true;
                    }
                }
                for (IControlAPI.Registration registration : control.rotationShapers()) {
                    if (registration.name().equalsIgnoreCase(name)) {
                        registration.close();
                        removed = true;
                    }
                }
                logDirect(removed ? "Removed " + name : "No shaper named " + name);
                break;
            }
            default:
                throw new CommandInvalidStateException("Unknown action '" + action + "'");
        }
    }

    @Override
    public Stream<String> tabComplete(String label, IArgConsumer args) throws CommandException {
        if (args.hasExactlyOne()) {
            return new TabCompleteHelper()
                    .append("status", "list", "vars", "costs", "script", "trace", "untrace", "remove")
                    .filterPrefix(args.getString())
                    .stream();
        }
        return Stream.empty();
    }

    @Override
    public String getShortDesc() {
        return "Inspect the movement and aim pipeline";
    }

    @Override
    public List<String> getLongDesc() {
        return Arrays.asList(
                "Every tick's keys, movement vector and aim are assembled in one pipeline, and anything registered",
                "into it can change them. This command shows what is in that pipeline and what it did.",
                "",
                "Usage:",
                "> control - What was applied on the last tick",
                "> control list - Every registered shaper, in the order it runs",
                "> control trace - Enable tracing, then show what each stage changed",
                "> control untrace - Stop tracing",
                "> control remove <name> - Remove a shaper by name",
                "> control costs - Registered path cost adjusters, and what they did last calculation",
                "",
                "Scripts let you write movement and aim yourself, as formulas in a text file, with no rebuild:",
                "> control vars - Every variable and function a script can use",
                "> control script list - Loaded scripts",
                "> control script reload - Re-read baritone/control/*.bar and apply them",
                "> control script show <name> - Print a script's source and how often it has run",
                "> control script enable|disable <name> - Toggle one without editing the file",
                "> control script eval <expression> - Evaluate an expression against the game right now",
                "> control script unload - Remove every script from the pipeline"
        );
    }
}
