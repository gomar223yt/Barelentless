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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The vocabulary a script is compiled against: which names are readable, which are writable, and which functions
 * exist.
 * <p>
 * Names are resolved to array slots at compile time, so the compiled script never looks anything up by string. The
 * context also carries the documentation for each name, which is what lets {@code control vars} print the whole
 * vocabulary in game - a scripting surface nobody can enumerate is a scripting surface nobody will use.
 *
 * @author Barelentless
 */
public final class ScriptContext {

    /**
     * A function callable from a script. Arguments arrive already evaluated.
     */
    @FunctionalInterface
    public interface Function {

        double apply(double[] arguments);
    }

    /**
     * A function that needs the whole frame, for things like querying the world at a computed position.
     */
    @FunctionalInterface
    public interface ContextualFunction {

        double apply(double[] frame, double[] arguments);
    }

    private static final class Slot {

        final int index;
        final String description;
        final boolean writable;

        Slot(int index, String description, boolean writable) {
            this.index = index;
            this.description = description;
            this.writable = writable;
        }
    }

    private static final class FunctionEntry {

        final String signature;
        final String description;
        final int minimumArguments;
        final int maximumArguments;
        final ContextualFunction function;

        FunctionEntry(String signature, String description, int minimumArguments, int maximumArguments,
                      ContextualFunction function) {
            this.signature = signature;
            this.description = description;
            this.minimumArguments = minimumArguments;
            this.maximumArguments = maximumArguments;
            this.function = function;
        }
    }

    private final Map<String, Slot> slots = new LinkedHashMap<>();
    private final Map<String, FunctionEntry> functions = new LinkedHashMap<>();
    private int nextSlot;

    /**
     * Declares a readable variable.
     *
     * @param name        The name a script uses
     * @param description What it means, for {@code control vars}
     * @return This context
     */
    public ScriptContext variable(String name, String description) {
        return declare(name, description, false);
    }

    /**
     * Declares a variable a script may assign to - the outputs, such as {@code forward} or {@code yaw}.
     */
    public ScriptContext output(String name, String description) {
        return declare(name, description, true);
    }

    private ScriptContext declare(String name, String description, boolean writable) {
        if (this.slots.containsKey(name)) {
            throw new IllegalArgumentException("duplicate variable " + name);
        }
        this.slots.put(name, new Slot(this.nextSlot++, description, writable));
        return this;
    }

    public ScriptContext function(String name, int arguments, String signature, String description, Function function) {
        return function(name, arguments, arguments, signature, description, (frame, args) -> function.apply(args));
    }

    public ScriptContext function(String name, int minimum, int maximum, String signature, String description,
                                  ContextualFunction function) {
        if (this.functions.containsKey(name)) {
            throw new IllegalArgumentException("duplicate function " + name);
        }
        this.functions.put(name, new FunctionEntry(signature, description, minimum, maximum, function));
        return this;
    }

    /**
     * @return The slot for a name, or -1 if it is not declared
     */
    public int slotOf(String name) {
        Slot slot = this.slots.get(name);
        return slot == null ? -1 : slot.index;
    }

    public boolean isWritable(String name) {
        Slot slot = this.slots.get(name);
        return slot != null && slot.writable;
    }

    /**
     * Declares a name that only this script uses, for {@code let} bindings. Locals live in the same frame as
     * everything else, so reading one is the same array access as reading a built-in.
     *
     * @return The slot allocated
     */
    public int declareLocal(String name) {
        Slot existing = this.slots.get(name);
        if (existing != null) {
            return existing.index;
        }
        Slot slot = new Slot(this.nextSlot++, "local", true);
        this.slots.put(name, slot);
        return slot.index;
    }

    ContextualFunction functionOf(String name, int argumentCount) {
        FunctionEntry entry = this.functions.get(name);
        if (entry == null) {
            return null;
        }
        if (argumentCount < entry.minimumArguments || argumentCount > entry.maximumArguments) {
            throw new ScriptException(name + " takes "
                    + (entry.minimumArguments == entry.maximumArguments
                    ? String.valueOf(entry.minimumArguments)
                    : entry.minimumArguments + " to " + entry.maximumArguments)
                    + " arguments, got " + argumentCount + " (" + entry.signature + ")");
        }
        return entry.function;
    }

    public boolean hasFunction(String name) {
        return this.functions.containsKey(name);
    }

    /**
     * How large a frame this context needs. Grows as locals are declared, so a frame must be allocated after the
     * script is compiled, not before.
     */
    public int frameSize() {
        return this.nextSlot;
    }

    /**
     * Every declared name with its description, for the in-game listing.
     */
    public List<String> describeVariables() {
        List<String> lines = new ArrayList<>();
        for (Map.Entry<String, Slot> entry : this.slots.entrySet()) {
            if ("local".equals(entry.getValue().description)) {
                continue;
            }
            lines.add((entry.getValue().writable ? "= " : "  ") + entry.getKey() + " - " + entry.getValue().description);
        }
        return lines;
    }

    public List<String> describeFunctions() {
        List<String> lines = new ArrayList<>();
        for (FunctionEntry entry : this.functions.values()) {
            lines.add(entry.signature + " - " + entry.description);
        }
        return lines;
    }

    /**
     * The names closest to a misspelling, so that an unknown name can suggest what was probably meant.
     */
    public List<String> suggest(String name) {
        List<String> best = new ArrayList<>();
        int bestDistance = Integer.MAX_VALUE;
        for (String candidate : this.slots.keySet()) {
            int distance = distance(name.toLowerCase(), candidate.toLowerCase());
            if (distance < bestDistance) {
                bestDistance = distance;
                best.clear();
                best.add(candidate);
            } else if (distance == bestDistance && best.size() < 3) {
                best.add(candidate);
            }
        }
        return bestDistance <= Math.max(2, name.length() / 2) ? best : new ArrayList<>();
    }

    private static int distance(String a, String b) {
        int[] previous = new int[b.length() + 1];
        int[] current = new int[b.length() + 1];
        for (int j = 0; j <= b.length(); j++) {
            previous[j] = j;
        }
        for (int i = 1; i <= a.length(); i++) {
            current[0] = i;
            for (int j = 1; j <= b.length(); j++) {
                int cost = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1;
                current[j] = Math.min(Math.min(current[j - 1] + 1, previous[j] + 1), previous[j - 1] + cost);
            }
            int[] swap = previous;
            previous = current;
            current = swap;
        }
        return previous[b.length()];
    }
}
