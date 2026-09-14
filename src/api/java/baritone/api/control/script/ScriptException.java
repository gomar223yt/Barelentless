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

/**
 * Thrown when a script cannot be compiled.
 * <p>
 * Carries the line and column so the message can point at the character that went wrong. A scripting surface lives
 * or dies on its error messages: the difference between "syntax error" and a caret under the offending token is the
 * difference between a feature people use and one they give up on.
 *
 * @author Barelentless
 */
public class ScriptException extends RuntimeException {

    private final int line;
    private final int column;
    private final String source;

    public ScriptException(String message) {
        this(message, -1, -1, null);
    }

    public ScriptException(String message, int line, int column, String source) {
        super(message);
        this.line = line;
        this.column = column;
        this.source = source;
    }

    public int getLine() {
        return this.line;
    }

    public int getColumn() {
        return this.column;
    }

    /**
     * A multi-line report: the message, the offending line, and a caret under the offending column.
     */
    public String describe() {
        if (this.source == null || this.line < 1) {
            return getMessage();
        }
        String[] lines = this.source.split("\n", -1);
        if (this.line > lines.length) {
            return getMessage();
        }
        String text = lines[this.line - 1];
        StringBuilder caret = new StringBuilder();
        for (int i = 0; i < Math.max(0, this.column - 1); i++) {
            caret.append(text.charAt(i) == '\t' ? '\t' : ' ');
        }
        caret.append('^');
        return "line " + this.line + ": " + getMessage() + "\n  " + text + "\n  " + caret;
    }
}
