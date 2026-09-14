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
 * A compiled arithmetic expression, evaluated against a frame of variable values.
 * <p>
 * Expressions are compiled once, when a script is loaded, into a tree of small closures that read variables by array
 * index rather than by name. That is the whole reason this exists instead of an off-the-shelf expression library:
 * these run inside the tick, several times per tick, and a hash lookup per variable reference would show up. After
 * compilation, evaluating {@code 0.4 * clamp(distXZ, 0, 1)} is a handful of array reads and arithmetic.
 * <p>
 * There are no statements, no loops and no allocation in an expression, so evaluation cannot hang and cannot produce
 * garbage. What it can produce is a non-finite number - dividing by zero is not an error here - which the caller is
 * expected to check before feeding it to movement.
 *
 * @author Barelentless
 */
@FunctionalInterface
public interface Expression {

    /**
     * @param frame The variable values, indexed as the {@link ScriptContext} that compiled this expression decided
     * @return The value of this expression
     */
    double evaluate(double[] frame);

    /**
     * A literal value. Kept as its own type rather than a lambda so that the parser can recognise it and fold
     * constant subexpressions: arithmetic written out for readability costs nothing at runtime.
     */
    final class Constant implements Expression {

        private final double value;

        public Constant(double value) {
            this.value = value;
        }

        public double value() {
            return this.value;
        }

        @Override
        public double evaluate(double[] frame) {
            return this.value;
        }

        @Override
        public String toString() {
            return Double.toString(this.value);
        }
    }

    static Expression constant(double value) {
        return new Constant(value);
    }

    /**
     * Reads a variable by its slot in the frame.
     */
    static Expression variable(int slot) {
        return frame -> frame[slot];
    }

    /**
     * @return The constant value of this expression, or {@code null} if it is not constant
     */
    static Double constantValue(Expression expression) {
        return expression instanceof Constant ? ((Constant) expression).value() : null;
    }
}
