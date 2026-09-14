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

import baritone.api.ml.memory.EpisodicMemory;

import java.util.List;

/**
 * The learning subsystem, as seen from outside.
 * <p>
 * Exposed so that an addon can read what the bot has learned, contribute experience of its own, and reuse the same
 * memory rather than building a parallel one. An addon that adds a new kind of movement, or a new process, gets the
 * learning for free by filing its outcomes here under its own context id.
 *
 * @author Barelentless
 */
public interface ILearningAPI {

    /**
     * Whether recording and training are currently active.
     */
    boolean isRunning();

    /**
     * Starts recording and training. Equivalent to enabling the master setting and running {@code ml start}.
     */
    void start();

    /**
     * Stops training and writes everything to disk. Loaded models keep serving predictions.
     */
    void stop();

    /**
     * Writes models and memory to disk immediately.
     */
    void save();

    /**
     * Discards every trained weight and every remembered situation, and overwrites the files on disk.
     */
    void reset();

    /**
     * The shared episodic memory. Addons may both read from and write to it; use a context id of your own so that
     * your situations never answer somebody else's question.
     *
     * @see EpisodicMemory#remember
     */
    EpisodicMemory getMemory();

    /**
     * How many gradient steps the trainer has taken in total, across sessions.
     */
    long getTrainingSteps();

    /**
     * A human readable multi-line status report - the same text {@code ml status} prints.
     */
    List<String> status();
}
