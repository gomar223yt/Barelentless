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

package baritone.api.control.profile;

import baritone.api.control.RotationTarget;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The profile's own behaviour: clamping, per-purpose overrides, naming, and surviving a round trip to text.
 * <p>
 * The clamping matters more than it looks. These values are typed by people into a chat command, and they end up
 * multiplying a movement vector or an angle every tick. A negative maximum speed or a NaN spread would not throw -
 * it would produce a bot that spins, and the player would have no idea why.
 *
 * @author Barelentless
 */
public class MotionProfileTest {

    @Test
    public void valuesAreClampedIntoTheirRange() {
        MotionProfile profile = new MotionProfile("test");
        profile.set(AimKnob.MAX_YAW_SPEED, -50);
        assertEquals(AimKnob.MAX_YAW_SPEED.minimum(), profile.get(AimKnob.MAX_YAW_SPEED), 1e-9);

        profile.set(AimKnob.JITTER, 9999);
        assertEquals(AimKnob.JITTER.maximum(), profile.get(AimKnob.JITTER), 1e-9);

        profile.set(GaitKnob.SPEED, Double.NaN);
        assertEquals("a NaN must fall back to the default, not poison every tick",
                GaitKnob.SPEED.defaultValue(), profile.get(GaitKnob.SPEED), 1e-9);
    }

    @Test
    public void perPurposeOverridesFallBackToTheProfile() {
        MotionProfile profile = new MotionProfile("test");
        profile.set(AimKnob.JITTER, 2);
        assertEquals(2, profile.get(AimKnob.JITTER, RotationTarget.Purpose.BLOCK_INTERACT), 1e-9);

        profile.set(RotationTarget.Purpose.BLOCK_INTERACT, AimKnob.JITTER, 0);
        assertEquals(0, profile.get(AimKnob.JITTER, RotationTarget.Purpose.BLOCK_INTERACT), 1e-9);
        assertEquals("other purposes must be unaffected",
                2, profile.get(AimKnob.JITTER, RotationTarget.Purpose.MOVEMENT), 1e-9);

        profile.set(RotationTarget.Purpose.BLOCK_INTERACT, AimKnob.JITTER, Double.NaN);
        assertFalse(profile.hasOverride(RotationTarget.Purpose.BLOCK_INTERACT, AimKnob.JITTER));
        assertEquals(2, profile.get(AimKnob.JITTER, RotationTarget.Purpose.BLOCK_INTERACT), 1e-9);
    }

    @Test
    public void dialsAreAddressableByName() {
        MotionProfile profile = new MotionProfile("test");
        assertTrue(profile.setByName("aim.maxYawSpeed", 30));
        assertTrue(profile.setByName("gait.speed", 0.5));
        assertTrue(profile.setByName("jitter", 1.5));              // bare aim name
        assertTrue(profile.setByName("movement.jitter", 3));       // per purpose
        assertTrue(profile.setByName("block.maxYawSpeed", 60));

        assertEquals(30, profile.getByName("aim.maxYawSpeed"), 1e-9);
        assertEquals(0.5, profile.getByName("gait.speed"), 1e-9);
        assertEquals(3, profile.get(AimKnob.JITTER, RotationTarget.Purpose.MOVEMENT), 1e-9);
        assertEquals(60, profile.get(AimKnob.MAX_YAW_SPEED, RotationTarget.Purpose.BLOCK_INTERACT), 1e-9);

        assertFalse(profile.setByName("nonsense", 1));
        assertFalse(profile.setByName("gait.nonsense", 1));
        assertTrue(Double.isNaN(profile.getByName("nonsense")));
    }

    @Test
    public void savingAndLoadingRoundTrips() {
        MotionProfile original = new MotionProfile("mine");
        original.set(AimKnob.MAX_YAW_SPEED, 17)
                .set(AimKnob.JITTER, 1.25)
                .set(GaitKnob.SPEED, 0.7)
                .set(GaitKnob.EDGE_CAUTION, 0.5)
                .set(RotationTarget.Purpose.BLOCK_INTERACT, AimKnob.JITTER, 0)
                .setSeed(12345);

        List<String> problems = new ArrayList<>();
        MotionProfile loaded = MotionProfile.load(original.save(), "fallback", problems);

        assertTrue(problems.toString(), problems.isEmpty());
        assertEquals("mine", loaded.getName());
        assertEquals(12345, loaded.getSeed());
        assertEquals(17, loaded.get(AimKnob.MAX_YAW_SPEED), 1e-9);
        assertEquals(1.25, loaded.get(AimKnob.JITTER), 1e-9);
        assertEquals(0.7, loaded.get(GaitKnob.SPEED), 1e-9);
        assertEquals(0.5, loaded.get(GaitKnob.EDGE_CAUTION), 1e-9);
        assertEquals(0, loaded.get(AimKnob.JITTER, RotationTarget.Purpose.BLOCK_INTERACT), 1e-9);
    }

    @Test
    public void onlyChangedValuesAreWritten() {
        MotionProfile profile = new MotionProfile("sparse");
        profile.set(GaitKnob.SPEED, 0.6);
        String text = profile.save();
        assertTrue(text, text.contains("gait.speed"));
        assertFalse("a saved profile should describe what you changed, not every default",
                text.contains("aim.maxYawSpeed"));
    }

    @Test
    public void loadingReportsProblemsInsteadOfThrowing() {
        List<String> problems = new ArrayList<>();
        MotionProfile loaded = MotionProfile.load(String.join("\n",
                "name = odd",
                "gait.speed = 0.5",
                "gait.nonsense = 3",
                "aim.jitter = notanumber",
                "no equals sign here"), "fallback", problems);

        assertEquals("the valid line must still have applied", 0.5, loaded.get(GaitKnob.SPEED), 1e-9);
        assertEquals(3, problems.size());
        assertTrue(problems.get(0), problems.get(0).contains("nonsense"));
        assertTrue(problems.get(1), problems.get(1).contains("not a number"));
    }

    @Test
    public void copyIsIndependent() {
        MotionProfile original = new MotionProfile("a").set(GaitKnob.SPEED, 0.5);
        original.set(RotationTarget.Purpose.MOVEMENT, AimKnob.JITTER, 2);
        MotionProfile copy = original.copy();

        copy.set(GaitKnob.SPEED, 1);
        copy.set(RotationTarget.Purpose.MOVEMENT, AimKnob.JITTER, 0);

        assertEquals(0.5, original.get(GaitKnob.SPEED), 1e-9);
        assertEquals(2, original.get(AimKnob.JITTER, RotationTarget.Purpose.MOVEMENT), 1e-9);
    }

    @Test
    public void everyDialHasASaneRangeAndAUniqueName() {
        List<String> seen = new ArrayList<>();
        for (AimKnob knob : AimKnob.values()) {
            assertTrue(knob.key(), knob.minimum() <= knob.defaultValue() && knob.defaultValue() <= knob.maximum());
            assertFalse("duplicate dial name " + knob.key(), seen.contains("aim." + knob.key()));
            seen.add("aim." + knob.key());
        }
        for (GaitKnob knob : GaitKnob.values()) {
            assertTrue(knob.key(), knob.minimum() <= knob.defaultValue() && knob.defaultValue() <= knob.maximum());
            assertFalse("duplicate dial name " + knob.key(), seen.contains("gait." + knob.key()));
            seen.add("gait." + knob.key());
        }
        assertEquals(seen.size(), MotionProfile.vocabulary().size());
    }

    @Test
    public void gaitDefaultsChangeNothing() {
        // the gait defaults have to be a no-op, because a profile that is switched on to add aim spread must not
        // silently also change how the bot walks
        MotionProfile profile = new MotionProfile("defaults");
        assertEquals(1, profile.get(GaitKnob.SPEED), 1e-9);
        assertEquals(0, profile.get(GaitKnob.ACCELERATION_TICKS), 1e-9);
        assertEquals(1, profile.get(GaitKnob.STRAFE), 1e-9);
        assertEquals(0, profile.get(GaitKnob.STRAFE_SMOOTHING), 1e-9);
        assertEquals(0, profile.get(GaitKnob.JITTER), 1e-9);
        assertEquals(0, profile.get(GaitKnob.CORNER_LEAN), 1e-9);
        assertEquals(0, profile.get(GaitKnob.EDGE_CAUTION), 1e-9);
        assertEquals(0, profile.get(GaitKnob.HESITATION), 1e-9);
        assertEquals("vanilla sneak slowdown", 0.3, profile.get(GaitKnob.SNEAK_SCALE), 1e-9);
    }

    @Test
    public void noiseIsOffByDefaultAndPreciseAimIsUntouched() {
        MotionProfile profile = new MotionProfile("defaults");
        assertEquals(0, profile.get(AimKnob.JITTER), 1e-9);
        assertEquals(0, profile.get(AimKnob.DRIFT), 1e-9);
        assertEquals("shaping exact aim must be opt-in", 0, profile.get(AimKnob.SHAPE_PRECISE), 1e-9);
        assertEquals("noise on exact aim must be opt-in even then", 0, profile.get(AimKnob.NOISE_ON_PRECISE), 1e-9);
    }
}
