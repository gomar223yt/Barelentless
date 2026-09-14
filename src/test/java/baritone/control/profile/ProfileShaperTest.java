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

package baritone.control.profile;

import baritone.api.control.RotationTarget;
import baritone.api.control.profile.AimKnob;
import baritone.api.control.profile.MotionProfile;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The turn model itself, without a game around it.
 * <p>
 * These are the properties that decide whether a profile is usable or whether the bot stands there vibrating: a turn
 * has to actually converge, it must not exceed the speed it was told, and it must not swing further past the target
 * than the overshoot dial allows. All three are easy to get wrong in a way that only shows up as "it feels weird",
 * which is the hardest kind of bug to act on.
 *
 * @author Barelentless
 */
public class ProfileShaperTest {

    private static final RotationTarget.Purpose PURPOSE = RotationTarget.Purpose.MOVEMENT;

    /**
     * Runs the model until it settles.
     *
     * @return ticks taken, worst overshoot, and the error left over
     */
    private static double[] simulate(MotionProfile profile, double startingError, int maxTicks) {
        double error = startingError;
        double velocity = 0;
        double worstOvershoot = 0;
        int ticks = 0;
        while (ticks < maxTicks && Math.abs(error) > 0.05) {
            velocity = ProfileShaper.step(velocity, error, profile, PURPOSE,
                    profile.get(AimKnob.MAX_YAW_SPEED, PURPOSE));
            error -= velocity;
            if (startingError != 0 && Math.signum(error) != Math.signum(startingError)) {
                worstOvershoot = Math.max(worstOvershoot, Math.abs(error));
            }
            ticks++;
        }
        return new double[]{ticks, worstOvershoot, error};
    }

    @Test
    public void aTurnConverges() {
        MotionProfile profile = new MotionProfile("test");
        double[] result = simulate(profile, 90, 400);
        assertTrue("did not settle within 400 ticks, error left " + result[2], result[0] < 400);
        assertTrue("settled off target by " + result[2], Math.abs(result[2]) <= 0.05);
    }

    @Test
    public void convergesFromEitherDirectionAndFromSmallErrors() {
        MotionProfile profile = new MotionProfile("test");
        assertTrue(simulate(profile, 90, 400)[0] < 400);
        assertTrue(simulate(profile, -90, 400)[0] < 400);
        assertTrue(simulate(profile, 0.5, 400)[0] < 400);
    }

    @Test
    public void speedNeverExceedsTheDial() {
        MotionProfile profile = new MotionProfile("test").set(AimKnob.MAX_YAW_SPEED, 5);
        double error = 180;
        double velocity = 0;
        for (int tick = 0; tick < 200; tick++) {
            velocity = ProfileShaper.step(velocity, error, profile, PURPOSE, 5);
            assertTrue("stepped " + velocity + " degrees with a 5 degree limit", Math.abs(velocity) <= 5 + 1e-9);
            error -= velocity;
        }
    }

    @Test
    public void overshootStaysWithinItsDial() {
        MotionProfile profile = new MotionProfile("test")
                .set(AimKnob.OVERSHOOT, 0.1)
                .set(AimKnob.OVERSHOOT_THRESHOLD, 10);
        double[] result = simulate(profile, 90, 400);
        assertTrue("overshot by " + result[1] + " degrees on a 90 degree turn", result[1] <= 90 * 0.1 + 1);
    }

    @Test
    public void noOvershootDialMeansNoOvershoot() {
        MotionProfile profile = new MotionProfile("test").set(AimKnob.OVERSHOOT, 0);
        assertEquals("a turn with overshoot off must approach from one side only",
                0, simulate(profile, 90, 400)[1], 1e-9);
    }

    @Test
    public void higherResponsivenessSettlesSooner() {
        MotionProfile lazy = new MotionProfile("lazy").set(AimKnob.RESPONSIVENESS, 0.1);
        MotionProfile snappy = new MotionProfile("snappy").set(AimKnob.RESPONSIVENESS, 0.9);
        assertTrue(simulate(snappy, 90, 400)[0] < simulate(lazy, 90, 400)[0]);
    }

    @Test
    public void accelerationMeansTheTurnBuildsUp() {
        MotionProfile profile = new MotionProfile("test").set(AimKnob.ACCELERATION, 0.3);
        double first = ProfileShaper.step(0, 90, profile, PURPOSE, 3600);
        double second = ProfileShaper.step(first, 90 - first, profile, PURPOSE, 3600);
        assertTrue("a turn should accelerate, got " + first + " then " + second, second > first);
    }

    @Test
    public void noiseIsBoundedAndRepeatable() {
        for (int i = 0; i < 2000; i++) {
            double phase = i * 0.37;
            double value = ProfileShaper.noise(phase, 12345);
            assertTrue("noise out of range: " + value, value >= -1 && value <= 1);
            assertEquals("noise must be repeatable for the same phase and seed",
                    value, ProfileShaper.noise(phase, 12345), 0);
        }
    }

    @Test
    public void noiseIsSmoothAndUnbiased() {
        double previous = ProfileShaper.noise(0, 7);
        double sum = 0;
        final int samples = 20000;
        for (int i = 1; i < samples; i++) {
            double value = ProfileShaper.noise(i * 0.01, 7);
            // smooth: a hundredth of a cycle cannot swing the value across its range. White noise would fail this,
            // and white noise is what drift must not be - a view that vibrates reads worse than one that snaps.
            assertTrue("noise jumped by " + Math.abs(value - previous), Math.abs(value - previous) < 0.2);
            previous = value;
            sum += value;
        }
        // unbiased: drift whose mean was not zero would slowly pull aim off target rather than wander around it
        assertTrue("noise is biased, mean " + sum / samples, Math.abs(sum / samples) < 0.05);
    }
}
