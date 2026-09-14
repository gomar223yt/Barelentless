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

import baritone.api.IBaritone;
import baritone.api.command.Command;
import baritone.api.command.argument.IArgConsumer;
import baritone.api.command.exception.CommandException;
import baritone.api.command.exception.CommandInvalidStateException;
import baritone.api.command.helpers.TabCompleteHelper;
import baritone.api.control.profile.AimKnob;
import baritone.api.control.profile.GaitKnob;
import baritone.api.control.profile.IProfileAPI;
import baritone.api.control.profile.MotionProfile;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

/**
 * Tunes how the bot walks and turns, live.
 *
 * @author Barelentless
 */
public class ProfileCommand extends Command {

    public ProfileCommand(IBaritone baritone) {
        super(baritone, "profile");
    }

    @Override
    public void execute(String label, IArgConsumer args) throws CommandException {
        IProfileAPI profiles = this.baritone.getProfileAPI();
        String action = args.hasAny() ? args.getString().toLowerCase() : "show";
        switch (action) {
            case "show": {
                args.requireMax(0);
                profiles.active().describe().forEach(this::logDirect);
                break;
            }
            case "on":
            case "off": {
                args.requireMax(0);
                profiles.active().setEnabled(action.equals("on"));
                logDirect("Profile '" + profiles.active().getName() + "' is "
                        + (profiles.active().isEnabled() ? "on" : "off"));
                break;
            }
            case "set": {
                String key = args.getString();
                double value = args.getAs(Double.class);
                if (!profiles.active().setByName(key, value)) {
                    throw new CommandInvalidStateException("no dial called '" + key
                            + "'; run 'profile dials' for the list");
                }
                profiles.active().setEnabled(true);
                logDirect(key + " = " + profiles.active().getByName(key)
                        + (profiles.active().isEnabled() ? "" : " (profile is off)"));
                break;
            }
            case "get": {
                String key = args.getString();
                double value = profiles.active().getByName(key);
                if (Double.isNaN(value)) {
                    throw new CommandInvalidStateException("no dial called '" + key + "'");
                }
                logDirect(key + " = " + value);
                break;
            }
            case "dials": {
                args.requireMax(0);
                logDirect("How it walks:");
                for (GaitKnob knob : GaitKnob.values()) {
                    logDirect(String.format("  gait.%-18s %s", knob.key(), knob.description()));
                }
                logDirect("How it turns:");
                for (AimKnob knob : AimKnob.values()) {
                    logDirect(String.format("  aim.%-19s %s", knob.key(), knob.description()));
                }
                logDirect("Prefix an aim dial with a purpose to override it there, e.g. 'movement.jitter 2' or");
                logDirect("'block.maxYawSpeed 60'. Purposes: block, entity, movement, flight, cosmetic.");
                break;
            }
            case "list": {
                args.requireMax(0);
                List<String> names = profiles.available();
                logDirect("Profiles in " + profiles.getDirectory() + ":");
                if (names.isEmpty()) {
                    logDirect("  (none)");
                }
                names.forEach(name -> logDirect("  " + name
                        + (name.equals(profiles.active().getName()) ? "  <- active" : "")));
                break;
            }
            case "use": {
                String name = args.getString();
                if (!profiles.activate(name)) {
                    throw new CommandInvalidStateException("no profile called '" + name + "'");
                }
                profiles.active().setEnabled(true);
                logDirect("Using profile '" + profiles.active().getName() + "'");
                profiles.active().describe().forEach(this::logDirect);
                break;
            }
            case "save": {
                if (args.hasAny()) {
                    profiles.active().setName(args.getString());
                }
                profiles.save();
                logDirect("Saved profile '" + profiles.active().getName() + "'");
                break;
            }
            case "reset": {
                args.requireMax(0);
                profiles.reset();
                logDirect("Profile '" + profiles.active().getName() + "' is back to defaults");
                break;
            }
            default:
                throw new CommandInvalidStateException("unknown action '" + action + "'");
        }
    }

    @Override
    public Stream<String> tabComplete(String label, IArgConsumer args) throws CommandException {
        if (args.hasExactlyOne()) {
            return new TabCompleteHelper()
                    .append("show", "on", "off", "set", "get", "dials", "list", "use", "save", "reset")
                    .filterPrefix(args.getString())
                    .stream();
        }
        if (args.has(2)) {
            String action = args.getString();
            if (action.equalsIgnoreCase("set") || action.equalsIgnoreCase("get")) {
                List<String> keys = new ArrayList<>(MotionProfile.vocabulary().keySet());
                return new TabCompleteHelper().append(keys.toArray(new String[0]))
                        .filterPrefix(args.getString())
                        .stream();
            }
            if (action.equalsIgnoreCase("use")) {
                return new TabCompleteHelper()
                        .append(this.baritone.getProfileAPI().available().toArray(new String[0]))
                        .filterPrefix(args.getString())
                        .stream();
            }
        }
        return Stream.empty();
    }

    @Override
    public String getShortDesc() {
        return "Tune how the bot walks and turns";
    }

    @Override
    public List<String> getLongDesc() {
        return Arrays.asList(
                "A motion profile is the set of dials that decide how the bot moves: how fast it turns, how hard it",
                "accelerates, whether it overshoots and settles, how much spread it has, how it approaches an edge.",
                "",
                "Aim dials can be overridden per purpose, which is what lets you have obvious spread while walking",
                "and none at all on the rotation a block break depends on. Whatever a profile does, the pipeline",
                "still clamps aim inside the tolerance of whatever asked for it, so no profile can break an",
                "interaction.",
                "",
                "Usage:",
                "> profile - Show the active profile",
                "> profile dials - Every dial, with a description",
                "> profile set <dial> <value> - Change one, e.g. 'profile set aim.jitter 1.5'",
                "> profile set movement.jitter 3 - Change it only for a purpose",
                "> profile on|off - Apply this profile, or stop applying it",
                "> profile list - Profiles saved on disk",
                "> profile use <name> - Load and apply one",
                "> profile save [name] - Save the active profile",
                "> profile reset - Back to defaults"
        );
    }
}
