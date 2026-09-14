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

/**
 * What actually happened when a movement was executed.
 * <p>
 * Baritone estimates the cost of every movement before executing it and then never looks at whether that estimate was
 * right. This record closes that loop: the situation as it was at the start, the estimate that was made, and the
 * ticks it really took. Feeding that difference back is what turns a fixed cost table into something that knows this
 * particular staircase is slower than the formula says.
 *
 * @author Barelentless
 */
public final class MovementExperience {

    /**
     * The encoded situation at the moment the movement began.
     */
    public final float[] situation;

    /**
     * Ordinal of the movement kind, used as the memory's context so that kinds never answer for each other.
     */
    public final int movementKind;

    /**
     * What the cost model predicted, in ticks.
     */
    public final double estimatedTicks;

    /**
     * What it actually took, in ticks.
     */
    public final double actualTicks;

    /**
     * Whether the movement completed rather than failing or being cancelled.
     */
    public final boolean successful;

    /**
     * Damage taken during the movement, in half-hearts. A fast route that costs four hearts is not a fast route.
     */
    public final float damageTaken;

    public final long timestamp;

    public MovementExperience(float[] situation, int movementKind, double estimatedTicks, double actualTicks,
                              boolean successful, float damageTaken, long timestamp) {
        this.situation = situation;
        this.movementKind = movementKind;
        this.estimatedTicks = estimatedTicks;
        this.actualTicks = actualTicks;
        this.successful = successful;
        this.damageTaken = damageTaken;
        this.timestamp = timestamp;
    }

    /**
     * The ratio of real to estimated cost. One means the estimate was right; two means it took twice as long as
     * planned. This, rather than the raw tick count, is what the cost model learns, because it transfers across
     * movements of wildly different lengths.
     */
    public double costRatio() {
        if (this.estimatedTicks <= 0) {
            return 1;
        }
        return Math.max(0.1, Math.min(10.0, this.actualTicks / this.estimatedTicks));
    }

    /**
     * The outcome stored in episodic memory: the cost ratio, penalised for damage, so that a movement which
     * technically succeeded while costing health is remembered as expensive.
     */
    public double memoryOutcome() {
        return costRatio() + this.damageTaken * 0.25;
    }

    /**
     * How much this experience deserves to be replayed: proportional to how wrong the estimate turned out to be.
     * A movement that took exactly as long as predicted has nothing left to teach.
     */
    public float priority() {
        return (float) Math.abs(Math.log(costRatio())) + (this.successful ? 0f : 2f) + this.damageTaken * 0.2f;
    }

    @Override
    public String toString() {
        return String.format("kind=%d estimate=%.1f actual=%.1f ratio=%.2f%s%s",
                this.movementKind, this.estimatedTicks, this.actualTicks, costRatio(),
                this.successful ? "" : " FAILED",
                this.damageTaken > 0 ? String.format(" damage=%.1f", this.damageTaken) : "");
    }
}
