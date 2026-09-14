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

import java.nio.file.Path;
import java.util.List;

/**
 * How the bot walks and turns, as settings rather than code.
 * <p>
 * The active profile is applied by a built-in shaper at priority 300 - below the learning shapers and below anything
 * an addon is likely to register, so a profile sets the character of the movement and more specific stages still get
 * the last word.
 *
 * @author Barelentless
 */
public interface IProfileAPI {

    /**
     * The profile currently shaping movement and aim. Never null; mutating it takes effect on the next tick.
     */
    MotionProfile active();

    /**
     * Replaces the active profile wholesale.
     */
    void setActive(MotionProfile profile);

    /**
     * Loads a saved profile by name and makes it active.
     *
     * @return Whether a profile of that name exists
     */
    boolean activate(String name);

    /**
     * Saves the active profile under its own name, in the profiles folder.
     */
    void save();

    /**
     * Names of every profile on disk.
     */
    List<String> available();

    /**
     * Where profiles are stored.
     */
    Path getDirectory();

    /**
     * Resets the active profile to defaults, keeping its name.
     */
    void reset();

    /**
     * Every dial's qualified name mapped to its description, for a config screen or a listing.
     *
     * @see MotionProfile#vocabulary()
     */
    default java.util.Map<String, String> vocabulary() {
        return MotionProfile.vocabulary();
    }
}
