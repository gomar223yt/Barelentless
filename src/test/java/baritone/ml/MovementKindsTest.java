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

package baritone.ml;

import baritone.api.utils.BetterBlockPos;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

/**
 * The identifiers the episodic memory files its records under.
 * <p>
 * These end up written to disk and read back weeks later, by a jar that has been through ProGuard in between. The
 * properties that matter are therefore not about elegance: the same movement must produce the same number forever,
 * movements that face the same problem must share a number, and movements that do not must not.
 *
 * @author Barelentless
 */
public class MovementKindsTest {

    private static int kind(int dx, int dy, int dz) {
        return MovementKinds.of(new BetterBlockPos(0, 64, 0), new BetterBlockPos(dx, 64 + dy, dz));
    }

    @Test
    public void directionDoesNotMatter() {
        // walking north and walking south are the same problem; splitting them would quarter the evidence behind
        // every estimate for nothing
        int north = kind(0, 0, -1);
        assertEquals(north, kind(0, 0, 1));
        assertEquals(north, kind(1, 0, 0));
        assertEquals(north, kind(-1, 0, 0));
    }

    @Test
    public void geometryThatBehavesDifferentlyIsDistinguished() {
        int level = kind(1, 0, 0);
        int ascend = kind(1, 1, 0);
        int descend = kind(1, -1, 0);
        int diagonal = kind(1, 0, 1);
        int pillar = kind(0, 1, 0);
        int downward = kind(0, -1, 0);
        int parkourTwo = kind(2, 0, 0);
        int parkourThree = kind(3, 0, 0);

        int[] all = {level, ascend, descend, diagonal, pillar, downward, parkourTwo, parkourThree};
        for (int i = 0; i < all.length; i++) {
            for (int j = i + 1; j < all.length; j++) {
                assertNotEquals("kinds " + i + " and " + j + " collided", all[i], all[j]);
            }
        }
    }

    @Test
    public void identifiersAreBoundedAndNonNegative() {
        for (int dy = -30; dy <= 30; dy++) {
            for (int dx = -10; dx <= 10; dx++) {
                for (int dz = -10; dz <= 10; dz++) {
                    int kind = kind(dx, dy, dz);
                    assertTrue("kind " + kind + " out of range", kind >= 0 && kind < MovementKinds.COUNT);
                }
            }
        }
    }

    @Test
    public void longFallsShareAKind() {
        // past a certain height a fall is a fall; distinguishing thirty blocks from forty would just split evidence
        assertEquals(kind(0, -20, 0), kind(0, -30, 0));
    }

    @Test
    public void descriptionsMatchTheGeometry() {
        assertEquals("pillar", MovementKinds.describe(kind(0, 1, 0)));
        assertEquals("downward", MovementKinds.describe(kind(0, -1, 0)));
        assertTrue(MovementKinds.describe(kind(1, 0, 0)).contains("straight"));
        assertTrue(MovementKinds.describe(kind(1, 0, 1)).contains("diagonal"));
        assertTrue(MovementKinds.describe(kind(1, 1, 0)).contains("ascend"));
        assertTrue(MovementKinds.describe(kind(3, 0, 0)).contains("jump 3"));
    }
}
