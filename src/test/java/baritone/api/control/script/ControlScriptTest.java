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

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The control scripting language.
 * <p>
 * Two kinds of property are worth testing here, and the second matters more. The first is that correct scripts
 * produce correct numbers. The second is that incorrect ones produce a message a person can act on: a typo should
 * suggest the name that was meant, a wrong argument count should say what the function takes, and every error should
 * carry the line it happened on. A scripting surface is only as usable as its errors.
 *
 * @author Barelentless
 */
public class ControlScriptTest {

    private static ScriptContext context() {
        ScriptContext context = new ScriptContext();
        context.variable("distXZ", "horizontal distance");
        context.variable("speed", "speed");
        context.variable("onGround", "on ground");
        context.variable("yawError", "yaw error");
        context.output("forward", "forward impulse");
        context.output("strafe", "strafe impulse");
        context.output("jump", "jump");
        context.output("sneak", "sneak");
        context.output("yawDelta", "yaw delta");
        return ScriptFunctions.install(context);
    }

    private static double evaluate(String expression) {
        ScriptContext context = context();
        return ExpressionParser.compile(expression, context).evaluate(new double[context.frameSize()]);
    }

    // ------------------------------------------------------------------------------------------------ expressions

    @Test
    public void operatorPrecedenceMatchesExpectation() {
        assertEquals(49, evaluate("2 + 3 * 4 ^ 2 - 1"), 1e-9);
        assertEquals(-8, evaluate("-2 ^ 3"), 1e-9);
        assertEquals(512, evaluate("2 ^ 3 ^ 2"), 1e-9);  // right associative
        assertEquals(0.5, evaluate("2 ^ -1"), 1e-9);
        assertEquals(14, evaluate("2 * (3 + 4)"), 1e-9);
    }

    @Test
    public void comparisonsAndLogicProduceOneOrZero() {
        assertEquals(1, evaluate("1 < 2"), 0);
        assertEquals(0, evaluate("2 < 2"), 0);
        assertEquals(1, evaluate("2 <= 2"), 0);
        assertEquals(1, evaluate("1 < 2 && 3 >= 3"), 0);
        assertEquals(0, evaluate("1 > 2 || 3 != 3"), 0);
        assertEquals(1, evaluate("!0"), 0);
        assertEquals(7, evaluate("1 ? 7 : 9"), 0);
        assertEquals(9, evaluate("0 ? 7 : 9"), 0);
    }

    @Test
    public void anglesWrapTheWayAimCodeNeeds() {
        assertEquals(10, evaluate("wrapDegrees(370)"), 1e-9);
        assertEquals(170, evaluate("wrapDegrees(-190)"), 1e-9);
        assertEquals(-170, evaluate("wrapDegrees(190)"), 1e-9);
        assertEquals(0, evaluate("wrapDegrees(720)"), 1e-9);
    }

    @Test
    public void helperFunctionsBehave() {
        assertEquals(1, evaluate("clamp(5, 0, 1)"), 0);
        assertEquals(2.5, evaluate("lerp(0, 10, 0.25)"), 1e-9);
        assertEquals(0.5, evaluate("smoothstep(0, 1, 0.5)"), 1e-9);
        assertEquals(0, evaluate("smoothstep(0, 1, -3)"), 0);
        assertEquals(1, evaluate("smoothstep(0, 1, 3)"), 0);
        assertEquals(3, evaluate("approach(0, 10, 3)"), 1e-9);
        assertEquals(10, evaluate("approach(0, 10, 30)"), 1e-9);
        assertEquals(2, evaluate("mod(-7, 3)"), 1e-9);  // non-negative, unlike java's %
    }

    @Test
    public void constantSubexpressionsAreFolded() {
        // folding is an optimisation, so the observable property is that the result is a Constant rather than a tree
        Expression folded = ExpressionParser.compile("20 * 0.05 + 2 ^ 3", context());
        assertTrue("expected a folded constant", folded instanceof Expression.Constant);
        assertEquals(9, folded.evaluate(new double[0]), 1e-9);
    }

    @Test
    public void shortCircuitProtectsTheRightHandSide() {
        // if && evaluated both sides, this would divide by zero and produce a non-finite result
        assertEquals(0, evaluate("0 && (1 / 0) > 0"), 0);
        assertEquals(1, evaluate("1 || (1 / 0) > 0"), 0);
    }

    // ---------------------------------------------------------------------------------------------------- scripts

    @Test
    public void aScriptComputesItsOutputs() {
        ScriptContext context = context();
        ControlScript script = ControlScript.parse(String.join("\n",
                "name careful",
                "priority 750",
                "when onGround && speed < 0.3",
                "",
                "let edge = clamp(distXZ, 0, 1)   # a comment",
                "forward = 0.3 + 0.4 * edge",
                "sneak = edge < 0.4",
                "yawDelta = yawError * 0.5"), "fallback", context);

        assertEquals("careful", script.getName());
        assertEquals(750, script.getPriority());
        assertTrue(script.hasGuard());
        assertEquals("[forward, sneak, yawDelta]", script.getOutputs().toString());

        double[] frame = new double[script.frameSize()];
        frame[context.slotOf("distXZ")] = 0.2;
        frame[context.slotOf("onGround")] = 1;
        frame[context.slotOf("speed")] = 0.1;
        frame[context.slotOf("yawError")] = 30;

        assertTrue(script.run(frame));
        assertEquals(0.38, frame[context.slotOf("forward")], 1e-9);
        assertEquals(1, frame[context.slotOf("sneak")], 0);
        assertEquals(15, frame[context.slotOf("yawDelta")], 1e-9);
    }

    @Test
    public void theGuardStopsTheWholeScript() {
        ScriptContext context = context();
        ControlScript script = ControlScript.parse(String.join("\n",
                "when speed < 0.3",
                "forward = 1"), "guarded", context);
        double[] frame = new double[script.frameSize()];
        frame[context.slotOf("speed")] = 0.9;
        assertFalse(script.run(frame));
        assertEquals("nothing should have been written", 0, frame[context.slotOf("forward")], 0);
    }

    @Test
    public void localsAreVisibleToLaterLines() {
        ScriptContext context = context();
        ControlScript script = ControlScript.parse(String.join("\n",
                "let a = 2",
                "let b = a * 3",
                "forward = a + b"), "locals", context);
        double[] frame = new double[script.frameSize()];
        script.run(frame);
        assertEquals(8, frame[context.slotOf("forward")], 1e-9);
    }

    @Test
    public void statementsMayShareALineWithSemicolons() {
        ScriptContext context = context();
        ControlScript script = ControlScript.parse("let a = 3; forward = a / 2", "semis", context);
        double[] frame = new double[script.frameSize()];
        script.run(frame);
        assertEquals(1.5, frame[context.slotOf("forward")], 1e-9);
    }

    // ----------------------------------------------------------------------------------------------------- errors

    private static ScriptException rejected(String source) {
        try {
            ControlScript.parse(source, "bad", context());
        } catch (ScriptException e) {
            return e;
        }
        fail("expected '" + source + "' to be rejected");
        return null;
    }

    @Test
    public void aTypoSuggestsWhatWasMeant() {
        ScriptException e = rejected("forward = forwrd * 2");
        assertTrue(e.getMessage(), e.getMessage().contains("did you mean"));
        assertTrue(e.getMessage(), e.getMessage().contains("forward"));
    }

    @Test
    public void writingToAReadOnlyVariableIsRejected() {
        ScriptException e = rejected("speed = 1");
        assertTrue(e.getMessage(), e.getMessage().contains("read-only"));
    }

    @Test
    public void wrongArgumentCountSaysWhatTheFunctionTakes() {
        ScriptException e = rejected("forward = clamp(1, 2)");
        assertTrue(e.getMessage(), e.getMessage().contains("takes 3"));
        assertTrue(e.getMessage(), e.getMessage().contains("clamp(x, low, high)"));
    }

    @Test
    public void aFunctionUsedWithoutParenthesesSaysSo() {
        ScriptException e = rejected("forward = clamp");
        assertTrue(e.getMessage(), e.getMessage().contains("needs parentheses"));
    }

    @Test
    public void errorsCarryTheLineAndRenderACaret() {
        ScriptException e = rejected(String.join("\n",
                "forward = 1",
                "strafe = 2",
                "jump = nonsense"));
        assertEquals(3, e.getLine());
        String description = e.describe();
        assertTrue(description, description.contains("line 3"));
        assertTrue(description, description.contains("jump = nonsense"));
        assertTrue(description, description.contains("^"));
    }

    @Test
    public void unfinishedExpressionsAreRejected() {
        rejected("forward = 2 +");
        rejected("forward = (2");
        rejected("forward = clamp(1, 2, 3");
        rejected("forward = 1 ? 2");
    }

    @Test
    public void anEmptyScriptIsRejected() {
        ScriptException e = rejected("# nothing but a comment\n\n");
        assertTrue(e.getMessage(), e.getMessage().contains("does not set anything"));
    }

    @Test
    public void noiseStaysInItsAdvertisedRange() {
        // the documented range is [-1, 1]; getting the divisor wrong in the hash would widen it silently, and a
        // script using noise() for spread would produce a bias instead of a wander
        ScriptContext context = context();
        Expression noise = ExpressionParser.compile("noise(distXZ)", context);
        double[] frame = new double[context.frameSize()];
        int slot = context.slotOf("distXZ");
        double sum = 0;
        final int samples = 20000;
        for (int i = 0; i < samples; i++) {
            frame[slot] = i * 0.013;
            double value = noise.evaluate(frame);
            assertTrue("noise out of range: " + value, value >= -1 && value <= 1);
            sum += value;
        }
        assertTrue("noise is biased, mean " + sum / samples, Math.abs(sum / samples) < 0.05);
    }

    @Test
    public void evaluationIsFastEnoughForEveryTick() {
        ScriptContext context = context();
        ControlScript script = ControlScript.parse(String.join("\n",
                "let edge = clamp(distXZ, 0, 1)",
                "forward = 0.3 + 0.4 * edge * smoothstep(0, 1, speed)",
                "yawDelta = wrapDegrees(yawError) * 0.5",
                "jump = distXZ < 0.6 && onGround"), "perf", context);
        double[] frame = new double[script.frameSize()];
        for (int i = 0; i < 200_000; i++) {
            script.run(frame);
        }
        long start = System.nanoTime();
        final int runs = 1_000_000;
        for (int i = 0; i < runs; i++) {
            script.run(frame);
        }
        double nanoseconds = (System.nanoTime() - start) / (double) runs;
        // measured around 120 ns; the bound is loose because the build runs wherever it runs, but a regression into
        // per-evaluation allocation or name lookup would blow straight past it
        assertTrue(String.format("%.0f ns per evaluation is too slow", nanoseconds), nanoseconds < 3000);
    }
}
