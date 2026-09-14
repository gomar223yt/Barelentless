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

package baritone.api.control;

import baritone.api.utils.Rotation;

/**
 * A request to look somewhere, together with why.
 * <p>
 * The why is the part Baritone never had, and the part that decides how the request may be treated. A rotation that
 * exists only so that a block can be broken must land exactly on that block or the interaction fails - it can be
 * slowed down but never rounded off. A rotation that exists because the bot is walking somewhere has no such
 * constraint and can be smoothed, delayed or ignored entirely. Telling a shaper which kind it is looking at is what
 * lets aim be made human without breaking the things that need precision.
 *
 * @author Barelentless
 */
public final class RotationTarget {

    /**
     * What the rotation is for.
     */
    public enum Purpose {

        /**
         * Needed to break or place a block. Must end up precise: the raytrace has to hit.
         */
        BLOCK_INTERACT(true),

        /**
         * Needed to attack or interact with an entity. Precise, but the target moves.
         */
        ENTITY_INTERACT(true),

        /**
         * Steering. The direction of travel; free to be shaped as long as it roughly points the right way.
         */
        MOVEMENT(false),

        /**
         * Elytra control, where pitch is the throttle and small errors compound over hundreds of blocks.
         */
        FLIGHT(true),

        /**
         * Looking around for its own sake - scanning, idling, appearing alive.
         */
        COSMETIC(false);

        private final boolean precise;

        Purpose(boolean precise) {
            this.precise = precise;
        }

        /**
         * Whether the final rotation has to match the request closely. Shapers must not deviate from a precise
         * target beyond {@link RotationTarget#getTolerance()}.
         */
        public boolean isPrecise() {
            return this.precise;
        }
    }

    private final Rotation rotation;
    private final Purpose purpose;
    private final String source;
    private final float tolerance;
    private final int priority;

    public RotationTarget(Rotation rotation, Purpose purpose, String source) {
        this(rotation, purpose, source, purpose.isPrecise() ? 0.05f : 8f, 0);
    }

    public RotationTarget(Rotation rotation, Purpose purpose, String source, float tolerance, int priority) {
        this.rotation = rotation;
        this.purpose = purpose;
        this.source = source;
        this.tolerance = tolerance;
        this.priority = priority;
    }

    public Rotation getRotation() {
        return this.rotation;
    }

    public Purpose getPurpose() {
        return this.purpose;
    }

    /**
     * Who asked. Free-form, for diagnostics.
     */
    public String getSource() {
        return this.source;
    }

    /**
     * How many degrees the final rotation may differ from the requested one.
     */
    public float getTolerance() {
        return this.tolerance;
    }

    /**
     * Higher wins when two targets are requested on the same tick.
     */
    public int getPriority() {
        return this.priority;
    }

    public RotationTarget withRotation(Rotation rotation) {
        return new RotationTarget(rotation, this.purpose, this.source, this.tolerance, this.priority);
    }

    public RotationTarget withTolerance(float tolerance) {
        return new RotationTarget(this.rotation, this.purpose, this.source, tolerance, this.priority);
    }

    /**
     * Clamps a shaped rotation back inside this target's tolerance, so a shaper cannot break a precise interaction
     * however aggressive it is. Applied automatically after every shaper runs.
     */
    public Rotation enforceTolerance(Rotation shaped) {
        if (this.tolerance >= 180f) {
            return shaped;
        }
        float yawError = Rotation.normalizeYaw(shaped.getYaw() - this.rotation.getYaw());
        float pitchError = shaped.getPitch() - this.rotation.getPitch();
        float yaw = shaped.getYaw();
        float pitch = shaped.getPitch();
        if (Math.abs(yawError) > this.tolerance) {
            yaw = this.rotation.getYaw() + Math.signum(yawError) * this.tolerance;
        }
        if (Math.abs(pitchError) > this.tolerance) {
            pitch = this.rotation.getPitch() + Math.signum(pitchError) * this.tolerance;
        }
        return new Rotation(yaw, pitch).normalizeAndClamp();
    }

    @Override
    public String toString() {
        return this.purpose + " " + this.rotation + " (+-" + this.tolerance + ") from " + this.source;
    }
}
