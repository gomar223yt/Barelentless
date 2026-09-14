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

package baritone.api.control.script;

import baritone.api.control.IControlAPI;

import java.nio.file.Path;
import java.util.List;

/**
 * Control scripts, as seen from outside.
 * <p>
 * Everything the {@code control script} command can do is here, plus the one thing a command cannot: registering a
 * script that exists only in memory. An addon can ship its movement logic as a string, register it on load and drop
 * it on unload, without touching the player's script folder or asking them to install a file.
 * <p>
 * Scripts registered this way are ordinary pipeline stages. They show up in {@code control list}, obey the same
 * tolerance clamps, and are removed by closing the handle.
 *
 * @author Barelentless
 */
public interface IScriptAPI {

    /**
     * Where {@code .bar} files are read from.
     */
    Path getDirectory();

    /**
     * Re-reads every script file and applies the result.
     *
     * @return One line per script loaded, rejected or removed
     */
    List<String> reload();

    /**
     * One line per currently loaded script: its name, priority, which pipelines it is in, and how often it has run.
     */
    List<String> describe();

    /**
     * Removes every file-backed script from the pipeline. The files themselves are untouched.
     */
    void unloadAll();

    /**
     * Compiles a script from source and registers it, without going near the filesystem.
     *
     * @param name   A name for diagnostics; overridden by a {@code name} line in the source
     * @param source The script
     * @return A handle that removes it again. Close it when your addon unloads.
     * @throws ScriptException if the script does not compile, with the line and column of the problem
     */
    IControlAPI.Registration register(String name, String source);

    /**
     * Compiles an expression against the script vocabulary and throws if it is not valid. Useful for validating
     * configuration a user typed before storing it.
     *
     * @throws ScriptException if it does not compile
     */
    void check(String expression);

    /**
     * Evaluates an expression against the game as it is right now. Intended for diagnostics and for configuration
     * screens, not for per-tick use - a script is the cheap way to do that.
     *
     * @throws ScriptException if it does not compile
     */
    double evaluate(String expression);

    /**
     * A fresh vocabulary: every variable and function a script may use, with descriptions. Compile your own
     * expressions against it with {@link ExpressionParser#compile}, or print it to show a user what is available.
     */
    ScriptContext newContext();
}
