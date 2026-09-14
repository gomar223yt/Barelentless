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
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * Checks the two claims the memory's design rests on, at the size it is actually expected to reach.
 * <p>
 * Approximate nearest neighbour search trades recall for speed, and the honest question is how much of each. A
 * memory that answers in a microsecond but only finds what it stored half the time would quietly make the bot worse
 * while looking like it was working; a memory with perfect recall that takes a millisecond per query would be
 * unusable from the pathfinder. Both are measured here rather than assumed.
 *
 * @author Barelentless
 */
public class EpisodicMemoryScaleTest {

    private static final int DIMENSION = 24;
    private static final int RECORDS = 20_000;

    private static float[] situation(Random random) {
        float[] key = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            key[i] = (float) random.nextGaussian();
        }
        return key;
    }

    @Test
    public void recallsWhatItStoredAndDoesSoQuickly() {
        Random random = new Random(20_260_914L);
        EpisodicMemory memory = new EpisodicMemory(DIMENSION, RECORDS * 2, new Random(7));

        // a hundred situations that will be queried later, buried among twenty thousand others
        List<float[]> probes = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            float[] key = situation(random);
            probes.add(key);
            for (int visit = 0; visit < 5; visit++) {
                memory.remember(key.clone(), 1, null, 4 + random.nextGaussian() * 0.1, true, 0);
            }
        }
        for (int i = 0; i < RECORDS; i++) {
            memory.remember(situation(random), 1, null, 20, true, 0);
        }

        int found = 0;
        for (float[] probe : probes) {
            EpisodicMemory.Estimate estimate = memory.estimate(probe, 1, 8);
            // the buried situations have an outcome of about 4, everything else about 20, so a wrong answer is
            // obvious rather than a matter of degree
            if (!estimate.isEmpty() && Math.abs(estimate.outcome - 4) < 2) {
                found++;
            }
        }
        assertTrue("recalled only " + found + " of 100 stored situations among " + memory.size() + " records",
                found >= 90);

        // warm up, then measure: the pathfinder calls this once per distinct situation, thousands of times a second
        for (int i = 0; i < 2000; i++) {
            memory.estimate(probes.get(i % probes.size()), 1, 8);
        }
        long start = System.nanoTime();
        final int queries = 20_000;
        for (int i = 0; i < queries; i++) {
            memory.estimate(probes.get(i % probes.size()), 1, 8);
        }
        double microsecondsPerQuery = (System.nanoTime() - start) / 1000.0 / queries;
        // measured at about 14 us per query at two hundred thousand records on the machine this was written on;
        // the bound is loose because the build runs wherever it runs, but tight enough to catch a regression back
        // into scanning a bucket proportional to the whole memory
        assertTrue(String.format("%.1f us per query at %d records is too slow", microsecondsPerQuery, memory.size()),
                microsecondsPerQuery < 60);
    }

    @Test
    public void consolidationShrinksWithoutLosingWhatMatters() {
        Random random = new Random(11);
        EpisodicMemory memory = new EpisodicMemory(DIMENSION, 50_000, new Random(13));
        long now = System.currentTimeMillis();

        float[] important = situation(random);
        for (int i = 0; i < 40; i++) {
            memory.remember(important.clone(), 3, null, 6, true, now);
        }
        // a thousand situations seen exactly once, an hour ago: the definition of what consolidation is for
        for (int i = 0; i < 1000; i++) {
            memory.remember(situation(random), 3, null, 15, true, now - 7_200_000L);
        }
        int before = memory.size();
        int removed = memory.consolidate(now, 2);

        assertTrue("consolidation removed " + removed + " of " + before, removed > 500);
        EpisodicMemory.Estimate estimate = memory.estimate(important, 3, 8);
        assertTrue("the frequently visited situation must survive consolidation", !estimate.isEmpty());
        assertTrue("and must keep its estimate, got " + estimate.outcome, Math.abs(estimate.outcome - 6) < 1);
    }
}
