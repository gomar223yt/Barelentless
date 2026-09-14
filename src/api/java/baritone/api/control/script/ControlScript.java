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
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * A compiled control script: a guard, and a sequence of assignments evaluated in order every tick.
 * <p>
 * The language is deliberately tiny. There are no loops, no functions to define, no state that survives a tick, and
 * no way to call into the game beyond the variables and functions the host declares. That is not a limitation to
 * apologise for - it is what makes a script safe to run inside the tick of a game the player is in the middle of. A
 * script cannot hang, cannot leak, and cannot do anything its {@link ScriptContext} did not offer it.
 * <p>
 * What it can do is all of the arithmetic you want over the full movement state, which turns out to be most of what
 * writing a movement by hand actually is:
 * <pre>
 *   name  careful ledges
 *   priority 750
 *   when  onGround &amp;&amp; !standable(0, -1, 1)
 *
 *   let edge = clamp(distXZ, 0, 1)
 *   forward = 0.3 + 0.4 * edge
 *   sneak   = edge &lt; 0.4
 *   yawDelta = yawError * 0.5
 * </pre>
 *
 * @author Barelentless
 */
public final class ControlScript {

    /**
     * One {@code name = expression} line.
     */
    public static final class Assignment {

        public final String name;
        public final int slot;
        public final Expression expression;

        Assignment(String name, int slot, Expression expression) {
            this.name = name;
            this.slot = slot;
            this.expression = expression;
        }
    }

    private final String name;
    private final int priority;
    private final Expression guard;
    private final List<Assignment> assignments;
    private final Set<String> outputs;
    private final ScriptContext context;
    private final String source;

    private ControlScript(String name, int priority, Expression guard, List<Assignment> assignments,
                          Set<String> outputs, ScriptContext context, String source) {
        this.name = name;
        this.priority = priority;
        this.guard = guard;
        this.assignments = assignments;
        this.outputs = outputs;
        this.context = context;
        this.source = source;
    }

    /**
     * Compiles a script against a context. The context must be freshly built for this script, because locals are
     * allocated into it.
     *
     * @throws ScriptException if the script does not compile, with a message that points at the problem
     */
    public static ControlScript parse(String source, String defaultName, ScriptContext context) {
        ExpressionParser parser = new ExpressionParser(source, context);
        List<Assignment> assignments = new ArrayList<>();
        Set<String> outputs = new LinkedHashSet<>();
        String name = defaultName;
        int priority = 500;
        Expression guard = null;

        while (true) {
            skipBlankLines(parser, source);
            if (parser.getPosition() >= source.length()) {
                break;
            }
            String word = readName(parser, source);
            if (word.isEmpty()) {
                throw parser.error("expected a name at the start of the line");
            }
            switch (word) {
                case "name": {
                    name = readRestOfLine(parser, source).trim();
                    if (name.isEmpty()) {
                        throw parser.error("'name' needs a value");
                    }
                    break;
                }
                case "priority": {
                    String text = readRestOfLine(parser, source).trim();
                    try {
                        priority = Integer.parseInt(text);
                    } catch (NumberFormatException e) {
                        throw parser.error("'priority' needs a whole number, got '" + text + "'");
                    }
                    break;
                }
                case "when": {
                    if (guard != null) {
                        throw parser.error("a script can only have one 'when'");
                    }
                    guard = parser.parseExpression();
                    break;
                }
                case "let": {
                    String local = readName(parser, source);
                    if (local.isEmpty()) {
                        throw parser.error("'let' needs a name");
                    }
                    if (context.slotOf(local) >= 0 && !context.isWritable(local)) {
                        throw parser.error("'" + local + "' is a built-in variable and cannot be redefined");
                    }
                    expect(parser, '=');
                    Expression value = parser.parseExpression();
                    assignments.add(new Assignment(local, context.declareLocal(local), value));
                    break;
                }
                default: {
                    int slot = context.slotOf(word);
                    if (slot < 0) {
                        List<String> suggestions = context.suggest(word);
                        throw parser.error("nothing here is called '" + word + "'"
                                + (suggestions.isEmpty() ? "" : " (did you mean " + String.join(", ", suggestions) + "?)")
                                + "; use 'let " + word + " = ...' for a temporary value");
                    }
                    if (!context.isWritable(word)) {
                        throw parser.error("'" + word + "' is read-only");
                    }
                    expect(parser, '=');
                    Expression value = parser.parseExpression();
                    assignments.add(new Assignment(word, slot, value));
                    outputs.add(word);
                    break;
                }
            }
            endOfLine(parser, source);
        }
        if (assignments.isEmpty()) {
            throw new ScriptException("this script does not set anything");
        }
        return new ControlScript(name, priority, guard, assignments, outputs, context, source);
    }

    /**
     * Evaluates the script into the frame.
     *
     * @return Whether the guard allowed it to run
     */
    public boolean run(double[] frame) {
        if (this.guard != null && this.guard.evaluate(frame) == 0) {
            return false;
        }
        for (int i = 0; i < this.assignments.size(); i++) {
            Assignment assignment = this.assignments.get(i);
            frame[assignment.slot] = assignment.expression.evaluate(frame);
        }
        return true;
    }

    public String getName() {
        return this.name;
    }

    public int getPriority() {
        return this.priority;
    }

    /**
     * The output variables this script assigns. The host uses this to decide which pipelines the script belongs in -
     * a script that only sets {@code yaw} has no business being called for movement.
     */
    public Set<String> getOutputs() {
        return this.outputs;
    }

    public boolean assigns(String output) {
        return this.outputs.contains(output);
    }

    public boolean hasGuard() {
        return this.guard != null;
    }

    public int frameSize() {
        return this.context.frameSize();
    }

    public ScriptContext getContext() {
        return this.context;
    }

    public String getSource() {
        return this.source;
    }

    public List<Assignment> getAssignments() {
        return this.assignments;
    }

    // ---------------------------------------------------------------------------------------------- line handling

    private static void skipBlankLines(ExpressionParser parser, String source) {
        while (parser.getPosition() < source.length()) {
            parser.skipWhitespace();
            if (parser.getPosition() < source.length() && source.charAt(parser.getPosition()) == '\n') {
                parser.newLine();
            } else {
                return;
            }
        }
    }

    private static String readName(ExpressionParser parser, String source) {
        parser.skipWhitespace();
        int start = parser.getPosition();
        int at = start;
        while (at < source.length() && (Character.isLetterOrDigit(source.charAt(at)) || source.charAt(at) == '_')) {
            at++;
        }
        parser.setPosition(at);
        return source.substring(start, at);
    }

    private static String readRestOfLine(ExpressionParser parser, String source) {
        int start = parser.getPosition();
        int at = start;
        while (at < source.length() && source.charAt(at) != '\n') {
            at++;
        }
        parser.setPosition(at);
        String text = source.substring(start, at);
        int comment = text.indexOf('#');
        return comment >= 0 ? text.substring(0, comment) : text;
    }

    private static void expect(ExpressionParser parser, char character) {
        parser.skipWhitespace();
        if (parser.getPosition() >= parser.getSource().length()
                || parser.getSource().charAt(parser.getPosition()) != character) {
            throw parser.error("expected '" + character + "'");
        }
        parser.setPosition(parser.getPosition() + 1);
    }

    private static void endOfLine(ExpressionParser parser, String source) {
        parser.skipWhitespace();
        if (parser.getPosition() >= source.length()) {
            return;
        }
        char c = source.charAt(parser.getPosition());
        if (c == '\n') {
            parser.newLine();
            return;
        }
        if (c == ';') {
            parser.setPosition(parser.getPosition() + 1);
            return;
        }
        throw parser.error("unexpected '" + c + "' - one statement per line");
    }
}
