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
import baritone.api.ml.memory.MemoryRecord;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The behaviour the memory is actually relied on for: that repeated exposure sharpens one record instead of
 * scattering duplicates, that confidence reflects evidence rather than luck, that capacity is respected, and that a
 * saved memory means the same thing when it is loaded back.
 *
 * @author Barelentless
 */
public class EpisodicMemoryTest {

    private static float[] situation(Random random, int dimension) {
        float[] key = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            key[i] = (float) random.nextGaussian();
        }
        return key;
    }

    @Test
    public void repeatedExposureSharpensOneRecord() {
        Random random = new Random(1);
        EpisodicMemory memory = new EpisodicMemory(16, 1000, 6, 10, random);
        float[] key = situation(random, 16);
        long now = 1_000_000L;
        for (int i = 0; i < 50; i++) {
            memory.remember(key.clone(), 1, null, 10 + (i % 2 == 0 ? 1 : -1), true, now);
        }
        assertEquals("identical situations must merge, not accumulate", 1, memory.size());
        MemoryRecord record = memory.records().get(0);
        assertEquals(50, record.getVisits());
        assertEquals(10.0, record.getOutcomeMean(), 0.05);
        assertEquals(1.0, record.getOutcomeDeviation(), 0.05);

        // Confidence is capped by the spread as well as the visit count, by design: fifty visits that disagree by a
        // tick each time are fifty pieces of evidence for an estimate that is genuinely only good to a tick. So the
        // meaningful assertion is relative - this must be worth far more than having seen it once.
        EpisodicMemory once = new EpisodicMemory(16, 1000, 6, 10, new Random(1));
        once.remember(key.clone(), 1, null, 10, true, now);
        double singleVisit = once.records().get(0).getConfidence();
        assertTrue("fifty visits (" + record.getConfidence() + ") should far outweigh one (" + singleVisit + ")",
                record.getConfidence() > singleVisit * 3);
        assertTrue("50 consistent visits should be trusted", record.getConfidence() > 0.8);
    }

    @Test
    public void oneVisitIsNotEvidence() {
        Random random = new Random(2);
        EpisodicMemory memory = new EpisodicMemory(16, 1000, 6, 10, random);
        float[] key = situation(random, 16);
        memory.remember(key.clone(), 1, null, 5, true, 0);
        EpisodicMemory.Estimate estimate = memory.estimate(key, 1, 4);
        assertFalse(estimate.isEmpty());
        assertTrue("a single visit must not read as certainty, got " + estimate.confidence,
                estimate.confidence < 0.5);
    }

    @Test
    public void differentContextsDoNotAnswerForEachOther() {
        Random random = new Random(3);
        EpisodicMemory memory = new EpisodicMemory(16, 1000, 6, 10, random);
        float[] key = situation(random, 16);
        for (int i = 0; i < 20; i++) {
            memory.remember(key.clone(), 7, null, 3, true, 0);
        }
        assertFalse(memory.estimate(key, 7, 4).isEmpty());
        assertTrue("a pillar memory must never answer a parkour question",
                memory.estimate(key, 8, 4).isEmpty());
    }

    @Test
    public void capacityIsRespected() {
        Random random = new Random(4);
        EpisodicMemory memory = new EpisodicMemory(16, 64, 6, 10, random);
        for (int i = 0; i < 500; i++) {
            memory.remember(situation(random, 16), 1, null, i, true, i);
        }
        assertTrue("size " + memory.size(), memory.size() <= 64);
        assertTrue("eviction should have happened", memory.getEvictions() > 0);
    }

    @Test
    public void survivesARoundTripToDisk() throws Exception {
        Random random = new Random(5);
        EpisodicMemory memory = new EpisodicMemory(16, 1000, 6, 10, random);
        float[] key = situation(random, 16);
        for (int i = 0; i < 30; i++) {
            memory.remember(key.clone(), 2, null, 7.5, i % 5 != 0, 12345L);
        }
        for (int i = 0; i < 40; i++) {
            memory.remember(situation(random, 16), 2, null, 20, true, 12345L);
        }
        EpisodicMemory.Estimate before = memory.estimate(key, 2, 8);

        Path file = Files.createTempDirectory("barelentless-memory").resolve("memory.brlm");
        memory.save(file);

        EpisodicMemory restored = new EpisodicMemory(16, 1000, 6, 10, new Random(99));
        int loaded = restored.load(file);
        Files.deleteIfExists(file);

        assertEquals(memory.size(), loaded);
        EpisodicMemory.Estimate after = restored.estimate(key, 2, 8);
        assertEquals(before.outcome, after.outcome, 1e-4);
        assertEquals(before.successRate, after.successRate, 1e-4);
        assertEquals(before.confidence, after.confidence, 1e-4);
    }

    @Test
    public void unrelatedSituationsAreNotRecalled() {
        Random random = new Random(6);
        EpisodicMemory memory = new EpisodicMemory(32, 1000, 8, 12, random);
        for (int i = 0; i < 200; i++) {
            memory.remember(situation(random, 32), 1, null, 4, true, 0);
        }
        // a brand new situation in a 32 dimensional space is essentially orthogonal to every stored one
        assertTrue("the memory must admit when it has not seen something",
                memory.estimate(situation(random, 32), 1, 8).isEmpty());
    }
}
