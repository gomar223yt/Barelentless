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

package baritone.api.ml;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Runs the shipped learning self-checks as part of the ordinary build.
 * <p>
 * The same checks are reachable in game through {@code ml selftest}, which is what makes them useful when a player
 * reports strange behaviour on their machine. Running them here as well means a broken gradient never reaches a
 * build in the first place.
 *
 * @author Barelentless
 */
public class MlDiagnosticsTest {

    @Test
    public void everyCheckPasses() {
        List<MlDiagnostics.Result> results = MlDiagnostics.runAll();
        assertTrue("no checks ran", results.size() >= 16);
        StringBuilder failures = new StringBuilder();
        for (MlDiagnostics.Result result : results) {
            if (!result.passed) {
                failures.append('\n').append(result);
            }
        }
        if (failures.length() > 0) {
            fail("learning self-checks failed:" + failures);
        }
    }
}
