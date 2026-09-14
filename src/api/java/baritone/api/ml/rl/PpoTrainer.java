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

package baritone.api.ml.rl;

import baritone.api.ml.Losses;
import baritone.api.ml.Tensor;
import baritone.api.ml.optim.Optimizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Proximal policy optimization over recorded trajectories.
 * <p>
 * PPO rather than a vanilla policy gradient for one reason that matters specifically here: the data is collected by
 * the same agent that is being updated, while a player is watching. A single over-large update turns a bot that walks
 * into a bot that vibrates, and the player notices immediately. The clipped objective bounds how far one update can
 * move the policy away from the one that produced the data, and the early stop on KL divergence bounds it again when
 * the clip is not enough.
 *
 * @author Barelentless
 */
public final class PpoTrainer {

    /**
     * What one update pass did, for diagnostics and for the in-game status readout.
     */
    public static final class Report {

        public final float policyLoss;
        public final float valueLoss;
        public final float entropy;
        public final float approximateKl;
        public final float clipFraction;
        public final int samples;
        public final int epochsRun;

        Report(float policyLoss, float valueLoss, float entropy, float approximateKl,
               float clipFraction, int samples, int epochsRun) {
            this.policyLoss = policyLoss;
            this.valueLoss = valueLoss;
            this.entropy = entropy;
            this.approximateKl = approximateKl;
            this.clipFraction = clipFraction;
            this.samples = samples;
            this.epochsRun = epochsRun;
        }

        @Override
        public String toString() {
            return String.format("policy=%.4f value=%.4f entropy=%.3f kl=%.4f clipped=%.0f%% n=%d epochs=%d",
                    this.policyLoss, this.valueLoss, this.entropy, this.approximateKl,
                    this.clipFraction * 100, this.samples, this.epochsRun);
        }
    }

    private final Policy policy;
    private final Optimizer optimizer;
    private final Random random;

    private float clipRange = 0.2f;
    private float valueCoefficient = 0.5f;
    private float entropyCoefficient = 0.005f;
    private float targetKl = 0.02f;
    private int epochs = 4;
    private int miniBatch = 64;
    private boolean normalizeAdvantages = true;

    public PpoTrainer(Policy policy, Optimizer optimizer, Random random) {
        this.policy = policy;
        this.optimizer = optimizer;
        this.random = random;
    }

    /**
     * Runs the configured number of epochs over the given finished trajectories.
     */
    public Report update(List<Trajectory> trajectories) {
        List<Trajectory.Step> steps = new ArrayList<>();
        for (Trajectory trajectory : trajectories) {
            if (!trajectory.isFinished()) {
                throw new IllegalArgumentException("trajectory must be finished before training");
            }
            steps.addAll(trajectory.steps());
        }
        if (steps.isEmpty()) {
            return new Report(0, 0, 0, 0, 0, 0, 0);
        }
        float advantageMean = 0;
        float advantageDeviation = 1;
        if (this.normalizeAdvantages) {
            double sum = 0;
            for (Trajectory.Step step : steps) {
                sum += step.advantage;
            }
            advantageMean = (float) (sum / steps.size());
            double variance = 0;
            for (Trajectory.Step step : steps) {
                double d = step.advantage - advantageMean;
                variance += d * d;
            }
            advantageDeviation = (float) Math.max(1e-6, Math.sqrt(variance / steps.size()));
        }

        this.policy.module().train();
        float policyLossSum = 0;
        float valueLossSum = 0;
        float entropySum = 0;
        float klSum = 0;
        float clippedSum = 0;
        int batches = 0;
        int epochsRun = 0;

        for (int epoch = 0; epoch < this.epochs; epoch++) {
            Collections.shuffle(steps, this.random);
            float epochKl = 0;
            int epochBatches = 0;
            for (int start = 0; start < steps.size(); start += this.miniBatch) {
                List<Trajectory.Step> batch = steps.subList(start, Math.min(steps.size(), start + this.miniBatch));
                if (batch.size() < 2) {
                    continue;
                }
                Report batchReport = updateBatch(batch, advantageMean, advantageDeviation);
                policyLossSum += batchReport.policyLoss;
                valueLossSum += batchReport.valueLoss;
                entropySum += batchReport.entropy;
                klSum += batchReport.approximateKl;
                clippedSum += batchReport.clipFraction;
                epochKl += batchReport.approximateKl;
                batches++;
                epochBatches++;
            }
            epochsRun++;
            if (epochBatches > 0 && epochKl / epochBatches > this.targetKl * 1.5f) {
                // the policy has already moved as far as this data can justify; further epochs would be fitting noise
                break;
            }
        }
        this.policy.module().eval();
        if (batches == 0) {
            return new Report(0, 0, 0, 0, 0, steps.size(), 0);
        }
        return new Report(policyLossSum / batches, valueLossSum / batches, entropySum / batches,
                klSum / batches, clippedSum / batches, steps.size(), epochsRun);
    }

    private Report updateBatch(List<Trajectory.Step> batch, float advantageMean, float advantageDeviation) {
        final int n = batch.size();
        final int features = batch.get(0).observation.length;
        final int continuous = this.policy.continuousActions();
        final int discrete = this.policy.discreteActions();

        Tensor observations = new Tensor(n, features);
        Tensor actions = continuous > 0 ? new Tensor(n, continuous) : null;
        boolean[][] discreteActions = discrete > 0 ? new boolean[n][discrete] : null;
        Tensor oldLogProbabilities = new Tensor(n, 1);
        Tensor advantages = new Tensor(n, 1);
        Tensor valueTargets = new Tensor(n, 1);

        for (int i = 0; i < n; i++) {
            Trajectory.Step step = batch.get(i);
            System.arraycopy(step.observation, 0, observations.data, i * features, features);
            if (actions != null && step.continuousAction != null) {
                System.arraycopy(step.continuousAction, 0, actions.data, i * continuous, continuous);
            }
            if (discreteActions != null && step.discreteAction != null) {
                discreteActions[i] = step.discreteAction;
            }
            oldLogProbabilities.data[i] = step.logProbability;
            advantages.data[i] = (step.advantage - advantageMean) / advantageDeviation;
            valueTargets.data[i] = step.valueTarget;
        }

        this.optimizer.zeroGrad();
        PolicyOutput output = this.policy.forward(observations);

        Tensor logProbabilities = null;
        Tensor entropy = null;
        if (continuous > 0 && actions != null) {
            logProbabilities = Losses.gaussianLogProbability(output.mean, output.logStandardDeviation, actions);
            entropy = Losses.gaussianEntropy(output.logStandardDeviation);
        }
        if (discrete > 0 && output.hasDiscrete()) {
            Tensor discreteLog = Losses.bernoulliLogProbability(output.discreteLogits, discreteActions);
            Tensor discreteEntropy = Losses.bernoulliEntropy(output.discreteLogits);
            logProbabilities = logProbabilities == null ? discreteLog : logProbabilities.add(discreteLog);
            entropy = entropy == null ? discreteEntropy : entropy.add(discreteEntropy);
        }
        if (logProbabilities == null) {
            throw new IllegalStateException("policy emits neither continuous nor discrete actions");
        }

        Tensor logRatio = logProbabilities.sub(oldLogProbabilities);
        Tensor ratio = logRatio.exp();
        Tensor unclipped = ratio.mul(advantages);
        Tensor clipped = ratio.clamp(1 - this.clipRange, 1 + this.clipRange).mul(advantages);
        Tensor policyLoss = unclipped.min(clipped).mean().neg();

        Tensor loss = policyLoss;
        float valueLossValue = 0;
        if (output.hasValue()) {
            Tensor valueLoss = Losses.huber(output.value, valueTargets, 10f);
            valueLossValue = valueLoss.item();
            loss = loss.add(valueLoss.scale(this.valueCoefficient));
        }
        Tensor entropyMean = entropy.mean();
        loss = loss.sub(entropyMean.scale(this.entropyCoefficient));

        loss.backward();
        this.optimizer.step();

        // approximate KL using the unbiased k3 estimator; more stable than -mean(logRatio) near zero
        float kl = 0;
        float clippedFraction = 0;
        for (int i = 0; i < n; i++) {
            float lr = logRatio.data[i];
            kl += (float) (Math.exp(lr) - 1 - lr);
            if (Math.abs(ratio.data[i] - 1f) > this.clipRange) {
                clippedFraction++;
            }
        }
        return new Report(policyLoss.item(), valueLossValue, entropyMean.item(),
                kl / n, clippedFraction / n, n, 1);
    }

    public PpoTrainer setClipRange(float clipRange) {
        this.clipRange = clipRange;
        return this;
    }

    public PpoTrainer setValueCoefficient(float valueCoefficient) {
        this.valueCoefficient = valueCoefficient;
        return this;
    }

    public PpoTrainer setEntropyCoefficient(float entropyCoefficient) {
        this.entropyCoefficient = entropyCoefficient;
        return this;
    }

    public PpoTrainer setTargetKl(float targetKl) {
        this.targetKl = targetKl;
        return this;
    }

    public PpoTrainer setEpochs(int epochs) {
        this.epochs = epochs;
        return this;
    }

    public PpoTrainer setMiniBatch(int miniBatch) {
        this.miniBatch = miniBatch;
        return this;
    }

    public PpoTrainer setNormalizeAdvantages(boolean normalizeAdvantages) {
        this.normalizeAdvantages = normalizeAdvantages;
        return this;
    }

    public Policy getPolicy() {
        return this.policy;
    }

    public Optimizer getOptimizer() {
        return this.optimizer;
    }
}
