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

import baritone.api.pathing.movement.IMovement;
import baritone.api.utils.BetterBlockPos;

/**
 * Stable identifiers for kinds of movement, used as the episodic memory's context.
 * <p>
 * The obvious implementation - hash the movement's class name - is wrong for the build people actually install. The
 * released jar is run through ProGuard, so {@code MovementParkour} becomes {@code a}, and the identifier a memory
 * file was written with would not survive the next release, let alone the difference between a development build and
 * a shipped one. The failure would be silent: every remembered situation would simply stop matching, and the bot
 * would quietly forget everything it had learned while still reporting a full memory.
 * <p>
 * So the identifier is derived from what the movement does rather than what it is called: how far it goes
 * horizontally, how far vertically, and whether it moves on one axis or two. That is stable under obfuscation,
 * stable across versions, and stable even if the movement classes are reorganised - and it carries real meaning,
 * because two movements with the same geometry face the same problem whatever their class is called.
 * <p>
 * Direction is deliberately not part of it. Walking north and walking south are the same problem, and splitting them
 * would quarter the evidence behind every estimate for nothing.
 *
 * @author Barelentless
 */
public final class MovementKinds {

    /**
     * Largest vertical change distinguished; anything beyond is a long fall and behaves the same way.
     */
    private static final int MAX_VERTICAL = 8;

    /**
     * Largest horizontal reach distinguished: a three block gap is the longest parkour jump in the game.
     */
    private static final int MAX_HORIZONTAL = 3;

    /**
     * One past the largest identifier {@link #of} can produce: seventeen vertical steps by four horizontal reaches
     * by four minor reaches.
     */
    public static final int COUNT = (MAX_VERTICAL * 2 + 1) * (MAX_HORIZONTAL + 1) * (MAX_HORIZONTAL + 1);

    private MovementKinds() {}

    /**
     * @return An identifier in {@code [0, 272)}, stable across builds
     */
    public static int of(IMovement movement) {
        if (movement == null) {
            return 0;
        }
        return of(movement.getSrc(), movement.getDest());
    }

    public static int of(BetterBlockPos source, BetterBlockPos destination) {
        int vertical = clamp(destination.y - source.y, -MAX_VERTICAL, MAX_VERTICAL);
        int alongX = Math.abs(destination.x - source.x);
        int alongZ = Math.abs(destination.z - source.z);
        int major = Math.min(Math.max(alongX, alongZ), MAX_HORIZONTAL);
        int minor = Math.min(Math.min(alongX, alongZ), MAX_HORIZONTAL);
        return ((vertical + MAX_VERTICAL) * (MAX_HORIZONTAL + 1) + major) * (MAX_HORIZONTAL + 1) + minor;
    }

    /**
     * A short human readable name for an identifier, for the {@code ml memory} listing. Derived from the same
     * geometry, so it stays accurate without depending on any class name.
     */
    public static String describe(int kind) {
        int minor = kind % (MAX_HORIZONTAL + 1);
        int rest = kind / (MAX_HORIZONTAL + 1);
        int major = rest % (MAX_HORIZONTAL + 1);
        int vertical = rest / (MAX_HORIZONTAL + 1) - MAX_VERTICAL;

        if (major == 0 && minor == 0) {
            return vertical > 0 ? "pillar" : vertical < 0 ? "downward" : "still";
        }
        String shape = minor > 0 ? "diagonal" : "straight";
        String direction = vertical > 0 ? "ascend" : vertical < 0 ? (vertical < -2 ? "fall" : "descend") : "level";
        String reach = major > 1 ? " jump " + major : "";
        return shape + " " + direction + reach;
    }

    private static int clamp(int value, int low, int high) {
        return Math.max(low, Math.min(high, value));
    }
}
