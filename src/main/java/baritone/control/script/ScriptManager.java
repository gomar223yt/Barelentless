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

package baritone.control.script;

import baritone.Baritone;
import baritone.api.control.ControlContext;
import baritone.api.control.IControlAPI;
import baritone.api.control.script.ControlScript;
import baritone.api.control.script.ExpressionParser;
import baritone.api.control.script.IScriptAPI;
import baritone.api.control.script.ScriptContext;
import baritone.api.control.script.ScriptException;
import baritone.api.pathing.movement.IMovement;
import baritone.api.pathing.path.IPathExecutor;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.world.phys.Vec3;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Loads control scripts from disk and keeps them registered in the pipeline.
 * <p>
 * Scripts live in {@code baritone/control/} as {@code .bar} files, one script per file, and are reloaded on demand -
 * the whole point is that writing a movement means editing a text file and typing a command, not rebuilding a mod.
 * A file that fails to compile is reported with the line and a caret under the problem, and the previously loaded
 * version of that script stays registered, so a typo cannot leave the bot with no controller at all.
 *
 * @author Barelentless
 */
public final class ScriptManager implements IScriptAPI {

    private static final String EXTENSION = ".bar";

    private final Baritone baritone;
    private final Path directory;

    private final Map<String, ScriptedShaper> shapers = new LinkedHashMap<>();
    private final Map<String, List<IControlAPI.Registration>> registrations = new LinkedHashMap<>();
    private final List<String> problems = new ArrayList<>();

    public ScriptManager(Baritone baritone) {
        this.baritone = baritone;
        this.directory = baritone.getDirectory().resolve("control");
    }

    @Override
    public Path getDirectory() {
        return this.directory;
    }

    /**
     * Reloads every script in the directory, replacing what is registered.
     *
     * @return A report, one line per script loaded or rejected
     */
    @Override
    public synchronized List<String> reload() {
        this.problems.clear();
        List<String> report = new ArrayList<>();
        try {
            Files.createDirectories(this.directory);
            writeExamplesIfMissing(report);
        } catch (IOException e) {
            report.add("could not use " + this.directory + ": " + e);
            return report;
        }

        List<Path> files = new ArrayList<>();
        try (Stream<Path> stream = Files.list(this.directory)) {
            stream.filter(path -> path.getFileName().toString().endsWith(EXTENSION)).sorted().forEach(files::add);
        } catch (IOException e) {
            report.add("could not list " + this.directory + ": " + e);
            return report;
        }

        for (Path file : files) {
            String key = file.getFileName().toString();
            try {
                String source = new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
                ScriptBindings bindings = new ScriptBindings();
                ControlScript script = ControlScript.parse(source,
                        key.substring(0, key.length() - EXTENSION.length()), bindings.getContext());
                ScriptedShaper shaper = new ScriptedShaper(script, bindings);
                if (!shaper.shapesMovement() && !shaper.shapesAim()) {
                    report.add(key + ": sets nothing the pipeline uses, ignored");
                    continue;
                }
                replace(key, shaper);
                report.add(key + ": " + shaper.describe());
            } catch (ScriptException e) {
                String message = key + ": " + e.describe();
                this.problems.add(message);
                report.add(message);
                report.add(this.shapers.containsKey(key)
                        ? "  (keeping the previously loaded version of " + key + ")"
                        : "  (not loaded)");
            } catch (IOException | UncheckedIOException e) {
                report.add(key + ": could not read - " + e);
            }
        }

        // drop scripts whose files are gone
        List<String> removed = new ArrayList<>();
        for (String key : new ArrayList<>(this.shapers.keySet())) {
            if (files.stream().noneMatch(path -> path.getFileName().toString().equals(key))) {
                unregister(key);
                this.shapers.remove(key);
                removed.add(key);
            }
        }
        for (String key : removed) {
            report.add(key + ": file is gone, unregistered");
        }
        if (report.isEmpty()) {
            report.add("no scripts in " + this.directory);
        }
        return report;
    }

    private void replace(String key, ScriptedShaper shaper) {
        unregister(key);
        this.shapers.put(key, shaper);
        List<IControlAPI.Registration> registered = new ArrayList<>(2);
        IControlAPI control = this.baritone.getControlAPI();
        if (shaper.shapesMovement()) {
            registered.add(control.registerInputShaper(shaper.name(), shaper.getScript().getPriority(), shaper));
        }
        if (shaper.shapesAim()) {
            registered.add(control.registerRotationShaper(shaper.name(), shaper.getScript().getPriority(), shaper));
        }
        this.registrations.put(key, registered);
    }

    private void unregister(String key) {
        List<IControlAPI.Registration> previous = this.registrations.remove(key);
        if (previous != null) {
            previous.forEach(IControlAPI.Registration::close);
        }
    }

    /**
     * Removes every script from the pipeline without deleting anything on disk.
     */
    @Override
    public synchronized void unloadAll() {
        for (String key : new ArrayList<>(this.shapers.keySet())) {
            unregister(key);
        }
        this.shapers.clear();
    }

    @Override
    public synchronized List<String> describe() {
        List<String> lines = new ArrayList<>();
        for (Map.Entry<String, ScriptedShaper> entry : this.shapers.entrySet()) {
            lines.add(entry.getKey() + "  ->  " + entry.getValue().describe());
        }
        if (lines.isEmpty()) {
            lines.add("no scripts loaded; put " + EXTENSION + " files in " + this.directory
                    + " and run 'control script reload'");
        }
        return lines;
    }

    public synchronized ScriptedShaper find(String name) {
        for (Map.Entry<String, ScriptedShaper> entry : this.shapers.entrySet()) {
            if (entry.getKey().equalsIgnoreCase(name)
                    || entry.getKey().equalsIgnoreCase(name + EXTENSION)
                    || entry.getValue().name().equalsIgnoreCase(name)) {
                return entry.getValue();
            }
        }
        return null;
    }

    public synchronized List<String> getProblems() {
        return new ArrayList<>(this.problems);
    }

    public synchronized int size() {
        return this.shapers.size();
    }

    /**
     * Evaluates a single expression against the current game state, for {@code control script eval}. Useful for
     * checking what a variable actually holds right now before building a script around it.
     */
    @Override
    public double evaluate(String expression) {
        ScriptBindings bindings = new ScriptBindings();
        ControlScript script = ControlScript.parse("let result = " + expression, "eval", bindings.getContext());
        double[] frame = new double[script.frameSize()];
        try {
            bindings.fill(frame, contextForEval(), this.baritone.getControlAPI().lastCommand());
            script.run(frame);
        } finally {
            bindings.release();
        }
        return frame[bindings.getContext().slotOf("result")];
    }

    /**
     * A context describing right now, for one-off evaluation outside the tick. The movement is whatever the path
     * executor is on, so {@code dx} and friends mean the same thing they would inside a script.
     */
    private ControlContext contextForEval() {
        IPathExecutor executor = this.baritone.getPathingBehavior().getCurrent();
        IMovement movement = executor == null || executor.getPosition() >= executor.getPath().movements().size()
                ? null
                : executor.getPath().movements().get(executor.getPosition());
        LocalPlayer player = this.baritone.getPlayerContext().player();
        double speed = 0;
        if (player != null) {
            Vec3 velocity = player.getDeltaMovement();
            speed = Math.sqrt(velocity.x * velocity.x + velocity.z * velocity.z);
        }
        return new ControlContext(this.baritone, this.baritone.getControlAPI().getTick(), movement, 0,
                this.baritone.getControlAPI().lastCommand(),
                this.baritone.getControlAPI().lastAppliedRotation(),
                speed, player != null && player.onGround());
    }

    /**
     * Writes the example scripts on first use, so the directory is never empty and mysterious. Existing files are
     * never touched - if you edit an example, it stays edited.
     */
    private void writeExamplesIfMissing(List<String> report) throws IOException {
        Path marker = this.directory.resolve("examples.written");
        if (Files.exists(marker)) {
            return;
        }
        for (Map.Entry<String, String> example : ControlScriptExamples.all().entrySet()) {
            Path path = this.directory.resolve(example.getKey());
            if (!Files.exists(path)) {
                Files.write(path, example.getValue().getBytes(StandardCharsets.UTF_8));
                report.add("wrote example " + example.getKey());
            }
        }
        Files.write(marker, ("Delete this file to have the examples written again.\n")
                .getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Registers a script that lives only in memory. See {@link IScriptAPI#register}.
     */
    @Override
    public synchronized IControlAPI.Registration register(String name, String source) {
        ScriptBindings bindings = new ScriptBindings();
        ControlScript script = ControlScript.parse(source, name, bindings.getContext());
        ScriptedShaper shaper = new ScriptedShaper(script, bindings);
        if (!shaper.shapesMovement() && !shaper.shapesAim()) {
            throw new ScriptException("this script sets nothing the pipeline uses");
        }
        IControlAPI control = this.baritone.getControlAPI();
        List<IControlAPI.Registration> registered = new ArrayList<>(2);
        if (shaper.shapesMovement()) {
            registered.add(control.registerInputShaper(shaper.name(), script.getPriority(), shaper));
        }
        if (shaper.shapesAim()) {
            registered.add(control.registerRotationShaper(shaper.name(), script.getPriority(), shaper));
        }
        return new IControlAPI.Registration() {

            @Override
            public String name() {
                return shaper.name();
            }

            @Override
            public int priority() {
                return script.getPriority();
            }

            @Override
            public boolean isActive() {
                return registered.stream().anyMatch(IControlAPI.Registration::isActive);
            }

            @Override
            public void close() {
                registered.forEach(IControlAPI.Registration::close);
            }
        };
    }

    @Override
    public ScriptContext newContext() {
        return new ScriptBindings().getContext();
    }

    /**
     * Checks an expression compiles, without running it.
     */
    @Override
    public void check(String expression) {
        ScriptBindings bindings = new ScriptBindings();
        ExpressionParser.compile(expression, bindings.getContext());
    }
}
