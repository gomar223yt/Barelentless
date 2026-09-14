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
import java.util.List;

/**
 * A recursive descent parser for the control expression language.
 * <p>
 * The grammar is the familiar C-like one, with the precedence people expect:
 * <pre>
 *   expression  := ternary
 *   ternary     := or ('?' expression ':' expression)?
 *   or          := and ('||' and)*
 *   and         := equality ('&amp;&amp;' equality)*
 *   equality    := comparison (('==' | '!=') comparison)*
 *   comparison  := additive (('&lt;' | '&gt;' | '&lt;=' | '&gt;=') additive)*
 *   additive    := multiplicative (('+' | '-') multiplicative)*
 *   multiply    := unary (('*' | '/' | '%') unary)*
 *   unary       := ('-' | '!') unary | power
 *   power       := primary ('^' unary)?
 *   primary     := NUMBER | NAME | NAME '(' arguments ')' | '(' expression ')'
 * </pre>
 * Booleans are doubles: zero is false, anything else is true, and comparisons produce exactly one or zero. That
 * avoids a type system entirely, which for a language whose whole purpose is arithmetic over game state is the right
 * trade - {@code jump = distXZ < 0.6 && onGround} reads naturally and compiles to arithmetic.
 * <p>
 * Constant subexpressions are folded as they are parsed, so writing {@code 20 * 0.05} for readability costs nothing.
 *
 * @author Barelentless
 */
public final class ExpressionParser {

    private final String source;
    private final ScriptContext context;
    private int position;
    private int line = 1;
    private int lineStart;

    public ExpressionParser(String source, ScriptContext context) {
        this.source = source;
        this.context = context;
    }

    /**
     * Compiles a single expression. Throws {@link ScriptException} if anything is wrong with it.
     */
    public static Expression compile(String source, ScriptContext context) {
        ExpressionParser parser = new ExpressionParser(source, context);
        Expression expression = parser.parseExpression();
        parser.skipWhitespace();
        if (parser.position < parser.source.length()) {
            throw parser.error("unexpected '" + parser.source.charAt(parser.position) + "'");
        }
        return expression;
    }

    // ------------------------------------------------------------------------------------------------- the grammar

    public Expression parseExpression() {
        return parseTernary();
    }

    private Expression parseTernary() {
        Expression condition = parseOr();
        skipWhitespace();
        if (!match('?')) {
            return condition;
        }
        Expression whenTrue = parseExpression();
        skipWhitespace();
        if (!match(':')) {
            throw error("expected ':' to complete the conditional");
        }
        Expression whenFalse = parseExpression();
        Double constant = Expression.constantValue(condition);
        if (constant != null) {
            return constant != 0 ? whenTrue : whenFalse;
        }
        return frame -> condition.evaluate(frame) != 0 ? whenTrue.evaluate(frame) : whenFalse.evaluate(frame);
    }

    private Expression parseOr() {
        Expression left = parseAnd();
        while (true) {
            skipWhitespace();
            if (!matchSequence("||")) {
                return left;
            }
            Expression right = parseAnd();
            Expression a = left;
            // short circuit, which matters because the right side may divide by something the left side guards
            left = frame -> a.evaluate(frame) != 0 || right.evaluate(frame) != 0 ? 1 : 0;
        }
    }

    private Expression parseAnd() {
        Expression left = parseEquality();
        while (true) {
            skipWhitespace();
            if (!matchSequence("&&")) {
                return left;
            }
            Expression right = parseEquality();
            Expression a = left;
            left = frame -> a.evaluate(frame) != 0 && right.evaluate(frame) != 0 ? 1 : 0;
        }
    }

    private Expression parseEquality() {
        Expression left = parseComparison();
        while (true) {
            skipWhitespace();
            if (matchSequence("==")) {
                Expression right = parseComparison();
                Expression a = left;
                left = frame -> a.evaluate(frame) == right.evaluate(frame) ? 1 : 0;
            } else if (matchSequence("!=")) {
                Expression right = parseComparison();
                Expression a = left;
                left = frame -> a.evaluate(frame) != right.evaluate(frame) ? 1 : 0;
            } else {
                return left;
            }
        }
    }

    private Expression parseComparison() {
        Expression left = parseAdditive();
        while (true) {
            skipWhitespace();
            if (matchSequence("<=")) {
                Expression right = parseAdditive();
                Expression a = left;
                left = frame -> a.evaluate(frame) <= right.evaluate(frame) ? 1 : 0;
            } else if (matchSequence(">=")) {
                Expression right = parseAdditive();
                Expression a = left;
                left = frame -> a.evaluate(frame) >= right.evaluate(frame) ? 1 : 0;
            } else if (peek() == '<' && peek(1) != '=') {
                this.position++;
                Expression right = parseAdditive();
                Expression a = left;
                left = frame -> a.evaluate(frame) < right.evaluate(frame) ? 1 : 0;
            } else if (peek() == '>' && peek(1) != '=') {
                this.position++;
                Expression right = parseAdditive();
                Expression a = left;
                left = frame -> a.evaluate(frame) > right.evaluate(frame) ? 1 : 0;
            } else {
                return left;
            }
        }
    }

    private Expression parseAdditive() {
        Expression left = parseMultiplicative();
        while (true) {
            skipWhitespace();
            if (peek() == '+') {
                this.position++;
                Expression right = parseMultiplicative();
                left = fold(left, right, (a, b) -> a + b);
            } else if (peek() == '-' && peek(1) != '-') {
                this.position++;
                Expression right = parseMultiplicative();
                left = fold(left, right, (a, b) -> a - b);
            } else {
                return left;
            }
        }
    }

    private Expression parseMultiplicative() {
        Expression left = parseUnary();
        while (true) {
            skipWhitespace();
            char c = peek();
            if (c == '*') {
                this.position++;
                Expression right = parseUnary();
                left = fold(left, right, (a, b) -> a * b);
            } else if (c == '/') {
                this.position++;
                Expression right = parseUnary();
                left = fold(left, right, (a, b) -> a / b);
            } else if (c == '%') {
                this.position++;
                Expression right = parseUnary();
                left = fold(left, right, (a, b) -> a % b);
            } else {
                return left;
            }
        }
    }

    private Expression parseUnary() {
        skipWhitespace();
        if (peek() == '-') {
            this.position++;
            Expression operand = parseUnary();
            Double constant = Expression.constantValue(operand);
            return constant != null ? Expression.constant(-constant) : frame -> -operand.evaluate(frame);
        }
        if (peek() == '!') {
            this.position++;
            Expression operand = parseUnary();
            return frame -> operand.evaluate(frame) == 0 ? 1 : 0;
        }
        return parsePower();
    }

    private Expression parsePower() {
        Expression base = parsePrimary();
        skipWhitespace();
        if (peek() != '^') {
            return base;
        }
        this.position++;
        // right associative, so 2^3^2 is 2^(3^2), and unary binds tighter on the right so 2^-1 works
        Expression exponent = parseUnary();
        return fold(base, exponent, Math::pow);
    }

    private Expression parsePrimary() {
        skipWhitespace();
        char c = peek();
        if (c == '(') {
            this.position++;
            Expression inner = parseExpression();
            skipWhitespace();
            if (!match(')')) {
                throw error("expected ')'");
            }
            return inner;
        }
        if (Character.isDigit(c) || (c == '.' && Character.isDigit(peek(1)))) {
            return parseNumber();
        }
        if (Character.isLetter(c) || c == '_') {
            return parseName();
        }
        if (c == 0) {
            throw error("unexpected end of expression");
        }
        throw error("unexpected '" + c + "'");
    }

    private Expression parseNumber() {
        int start = this.position;
        while (Character.isDigit(peek()) || peek() == '.') {
            this.position++;
        }
        // exponent form, e.g. 1e-3
        if ((peek() == 'e' || peek() == 'E')
                && (Character.isDigit(peek(1)) || ((peek(1) == '-' || peek(1) == '+') && Character.isDigit(peek(2))))) {
            this.position += 2;
            while (Character.isDigit(peek())) {
                this.position++;
            }
        }
        String text = this.source.substring(start, this.position);
        try {
            return Expression.constant(Double.parseDouble(text));
        } catch (NumberFormatException e) {
            throw error("'" + text + "' is not a number");
        }
    }

    private Expression parseName() {
        int start = this.position;
        while (Character.isLetterOrDigit(peek()) || peek() == '_') {
            this.position++;
        }
        String name = this.source.substring(start, this.position);
        int nameColumn = start - this.lineStart + 1;

        skipWhitespace();
        if (peek() == '(') {
            this.position++;
            List<Expression> arguments = new ArrayList<>();
            skipWhitespace();
            if (peek() != ')') {
                while (true) {
                    arguments.add(parseExpression());
                    skipWhitespace();
                    if (match(',')) {
                        continue;
                    }
                    break;
                }
            }
            if (!match(')')) {
                throw error("expected ')' to close the call to " + name);
            }
            ScriptContext.ContextualFunction function;
            try {
                function = this.context.functionOf(name, arguments.size());
            } catch (ScriptException e) {
                throw new ScriptException(e.getMessage(), this.line, nameColumn, this.source);
            }
            if (function == null) {
                throw new ScriptException("no function named '" + name + "'", this.line, nameColumn, this.source);
            }
            Expression[] argumentArray = arguments.toArray(new Expression[0]);
            final int count = argumentArray.length;
            // the argument buffer is allocated per call site, not per evaluation: a script runs on one thread, and
            // reusing it is the difference between garbage every tick and none
            final double[] buffer = new double[count];
            return frame -> {
                for (int i = 0; i < count; i++) {
                    buffer[i] = argumentArray[i].evaluate(frame);
                }
                return function.apply(frame, buffer);
            };
        }

        switch (name) {
            case "true":
                return Expression.constant(1);
            case "false":
                return Expression.constant(0);
            case "pi":
                return Expression.constant(Math.PI);
            case "e":
                return Expression.constant(Math.E);
            default:
                break;
        }
        int slot = this.context.slotOf(name);
        if (slot < 0) {
            List<String> suggestions = this.context.suggest(name);
            String hint = suggestions.isEmpty() ? "" : " (did you mean " + String.join(", ", suggestions) + "?)";
            if (this.context.hasFunction(name)) {
                hint = " (that is a function - it needs parentheses)";
            }
            throw new ScriptException("no variable named '" + name + "'" + hint, this.line, nameColumn, this.source);
        }
        return Expression.variable(slot);
    }

    // ----------------------------------------------------------------------------------------------------- lexing

    private interface DoubleOperator {

        double apply(double a, double b);
    }

    private static Expression fold(Expression left, Expression right, DoubleOperator operator) {
        Double a = Expression.constantValue(left);
        Double b = Expression.constantValue(right);
        if (a != null && b != null) {
            return Expression.constant(operator.apply(a, b));
        }
        return frame -> operator.apply(left.evaluate(frame), right.evaluate(frame));
    }

    private char peek() {
        return peek(0);
    }

    private char peek(int offset) {
        int index = this.position + offset;
        return index < this.source.length() ? this.source.charAt(index) : 0;
    }

    private boolean match(char expected) {
        if (peek() == expected) {
            this.position++;
            return true;
        }
        return false;
    }

    private boolean matchSequence(String expected) {
        if (this.source.startsWith(expected, this.position)) {
            this.position += expected.length();
            return true;
        }
        return false;
    }

    /**
     * Skips whitespace and comments. Newlines are significant to the script parser, so this stops at them unless
     * told otherwise.
     */
    void skipWhitespace() {
        while (this.position < this.source.length()) {
            char c = this.source.charAt(this.position);
            if (c == '\n') {
                break;
            }
            if (Character.isWhitespace(c)) {
                this.position++;
            } else if (c == '#' || (c == '/' && peek(1) == '/')) {
                while (this.position < this.source.length() && this.source.charAt(this.position) != '\n') {
                    this.position++;
                }
            } else {
                break;
            }
        }
    }

    ScriptException error(String message) {
        return new ScriptException(message, this.line, this.position - this.lineStart + 1, this.source);
    }

    int getPosition() {
        return this.position;
    }

    void setPosition(int position) {
        this.position = position;
    }

    int getLine() {
        return this.line;
    }

    void newLine() {
        this.line++;
        this.position++;
        this.lineStart = this.position;
    }

    String getSource() {
        return this.source;
    }

    ScriptContext getContext() {
        return this.context;
    }
}
