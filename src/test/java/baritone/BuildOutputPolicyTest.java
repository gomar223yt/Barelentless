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

package baritone;

import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Guards the two properties that make this fork usable from another mod.
 * <p>
 * Both live in build configuration rather than in code, which is exactly why they need a test: nothing else would
 * notice them regressing. A merge from upstream, or an innocent-looking cleanup of the ProGuard config, would
 * silently produce a release whose classes are all called {@code a} and whose API is missing - and the failure would
 * only surface as somebody's addon crashing weeks later.
 *
 * @author Barelentless
 */
public class BuildOutputPolicyTest {

    private static String read(String relative) throws IOException {
        Path path = Paths.get(relative);
        if (!Files.exists(path)) {
            // tests may run from the root or from a subproject depending on how they are invoked
            path = Paths.get("..").resolve(relative);
        }
        assertTrue("cannot find " + relative + " from " + Paths.get("").toAbsolutePath(), Files.exists(path));
        return new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
    }

    @Test
    public void releasesAreNotObfuscated() throws IOException {
        String config = read("scripts/proguard.pro");
        assertTrue("proguard.pro must disable renaming: an addon compiled against the API cannot call a class that "
                + "has been renamed to 'a'", config.contains("-dontobfuscate"));
        assertFalse("-repackageclasses moves classes out of the packages addons import them from",
                config.contains("\n-repackageclasses"));
        assertFalse("-flattenpackagehierarchy moves classes out of the packages addons import them from",
                config.contains("\n-flattenpackagehierarchy"));
    }

    @Test
    public void everyBuildKeepsTheApi() throws IOException {
        String config = read("scripts/proguard.pro");
        assertTrue("proguard.pro must keep baritone.api", config.contains("-keep class baritone.api.** { *; }"));

        String task = read("buildSrc/src/main/java/baritone/gradle/task/ProguardTask.java");
        assertFalse("the standalone build must not strip the keep-api rule - that is the jar people install, and "
                        + "without the API in it an addon can compile but not run",
                task.contains("removeIf(s -> s.contains(\"# this is the keep api\"))"));
    }
}
