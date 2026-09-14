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

import baritone.api.ml.data.ReplayBuffer;
import baritone.api.ml.data.RunningStatistics;
import baritone.api.ml.io.ModelIO;
import baritone.api.ml.memory.EpisodicMemory;
import baritone.api.ml.nn.Activation;
import baritone.api.ml.nn.GruCell;
import baritone.api.ml.nn.LayerNorm;
import baritone.api.ml.nn.Linear;
import baritone.api.ml.nn.Mlp;
import baritone.api.ml.nn.Module;
import baritone.api.ml.nn.SequenceEncoder;
import baritone.api.ml.optim.Adam;
import baritone.api.ml.optim.LearningRateSchedule;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Self-checks for the learning stack.
 * <p>
 * A learning system that silently computes the wrong gradient does not crash - it just gets slowly, plausibly worse,
 * and by the time anyone notices, weeks of recorded play have been fit to a broken objective. So every differentiable
 * operation here is checked against a numerical estimate of its own derivative, and the checks ship with the mod
 * rather than living in a test folder nobody runs: {@code ml selftest} in game runs exactly this.
 * <p>
 * Also runnable standalone: {@code java baritone.api.ml.MlDiagnostics}.
 *
 * @author Barelentless
 */
public final class MlDiagnostics {

    /**
     * One check and what it found.
     */
    public static final class Result {

        public final String name;
        public final boolean passed;
        public final String detail;

        Result(String name, boolean passed, String detail) {
            this.name = name;
            this.passed = passed;
            this.detail = detail;
        }

        @Override
        public String toString() {
            return (this.passed ? "PASS " : "FAIL ") + this.name + " - " + this.detail;
        }
    }

    private MlDiagnostics() {}

    /**
     * Runs every check and returns the results, in order. Never throws: a failing check is reported, not raised.
     */
    public static List<Result> runAll() {
        List<Result> results = new ArrayList<>();
        results.add(guard("gradients/elementwise", MlDiagnostics::checkElementwiseGradients));
        results.add(guard("gradients/activations", MlDiagnostics::checkActivationGradients));
        results.add(guard("gradients/matmul", MlDiagnostics::checkMatmulGradients));
        results.add(guard("gradients/softmax", MlDiagnostics::checkSoftmaxGradients));
        results.add(guard("gradients/layernorm", MlDiagnostics::checkLayerNormGradients));
        results.add(guard("gradients/attention", MlDiagnostics::checkAttentionGradients));
        results.add(guard("gradients/gru", MlDiagnostics::checkGruGradients));
        results.add(guard("gradients/losses", MlDiagnostics::checkLossGradients));
        results.add(guard("learning/xor", MlDiagnostics::checkXorLearning));
        results.add(guard("learning/sequence", MlDiagnostics::checkSequenceLearning));
        results.add(guard("optim/clipping", MlDiagnostics::checkGradientClipping));
        results.add(guard("data/statistics", MlDiagnostics::checkRunningStatistics));
        results.add(guard("data/replay", MlDiagnostics::checkReplayBuffer));
        results.add(guard("memory/recall", MlDiagnostics::checkEpisodicMemory));
        results.add(guard("io/checkpoint", MlDiagnostics::checkCheckpointRoundTrip));
        return results;
    }

    private static Result guard(String name, Supplier<Result> check) {
        try {
            Result result = check.get();
            return new Result(name, result.passed, result.detail);
        } catch (Throwable t) {
            return new Result(name, false, t.getClass().getSimpleName() + ": " + t.getMessage());
        }
    }

    // ------------------------------------------------------------------------------------------- gradient checking

    /**
     * Compares analytic gradients against central differences.
     * <p>
     * Central differences rather than forward, and float tensors evaluated in double where it matters, because the
     * whole check is a fight against cancellation: at single precision, too small an epsilon measures rounding noise
     * and too large a one measures curvature. The tolerance below is relative, which is what makes it meaningful
     * across operations whose outputs differ by orders of magnitude.
     */
    private static Result gradientCheck(String label, Tensor[] inputs, Function<Tensor[], Tensor> objective,
                                        float epsilon, float tolerance) {
        Tensor loss = objective.apply(inputs);
        for (Tensor input : inputs) {
            input.zeroGrad();
        }
        loss.backward();

        // Below this magnitude a finite difference is measuring float rounding, not a derivative: the two evaluations
        // differ in their last few bits, and dividing that by 2*epsilon inflates it into a fake gradient. Anything
        // this small is reported as agreement regardless of its relative error, which is the only honest thing to do
        // at single precision - it is the reason a correct attention backward "fails" a naive relative-error check.
        final float noiseFloor = Math.max(1e-7f, Math.abs(loss.item()) * 8f * 6e-8f / epsilon);

        float worst = 0;
        String worstWhere = "";
        for (int t = 0; t < inputs.length; t++) {
            Tensor input = inputs[t];
            float[] analytic = input.gradient();
            if (analytic == null) {
                return new Result(label, false, "input " + t + " received no gradient at all");
            }
            for (int i = 0; i < input.size(); i++) {
                float original = input.data[i];
                input.data[i] = original + epsilon;
                float plus = objective.apply(inputs).item();
                input.data[i] = original - epsilon;
                float minus = objective.apply(inputs).item();
                input.data[i] = original;

                float numeric = (plus - minus) / (2 * epsilon);
                if (Math.abs(numeric) < noiseFloor && Math.abs(analytic[i]) < noiseFloor) {
                    continue;
                }
                float difference = Math.abs(numeric - analytic[i]);
                float scale = Math.max(noiseFloor, Math.max(Math.abs(numeric), Math.abs(analytic[i])));
                float relative = difference / scale;
                if (relative > worst) {
                    worst = relative;
                    worstWhere = "input " + t + "[" + i + "] analytic=" + analytic[i] + " numeric=" + numeric;
                }
            }
        }
        boolean passed = worst <= tolerance;
        return new Result(label, passed, passed
                ? String.format("max relative error %.2e", worst)
                : String.format("max relative error %.2e at %s", worst, worstWhere));
    }

    private static Result checkElementwiseGradients() {
        Random random = new Random(7);
        Tensor a = randomParameter(3, 4, random);
        Tensor b = randomParameter(3, 4, random);
        Tensor bias = randomParameter(1, 4, random);
        return gradientCheck("elementwise", new Tensor[]{a, b, bias}, in -> {
            Tensor x = in[0];
            Tensor y = in[1];
            Tensor c = in[2];
            return x.mul(y).add(x.sub(y).square()).addRowVector(c).div(y.abs().plus(2f)).sum();
        }, 1e-3f, 2e-2f);
    }

    private static Result checkActivationGradients() {
        Random random = new Random(11);
        Tensor a = randomParameter(4, 5, random);
        return gradientCheck("activations", new Tensor[]{a}, in -> {
            Tensor x = in[0];
            return x.relu().sum()
                    .add(x.gelu().sum())
                    .add(x.tanh().sum())
                    .add(x.sigmoid().sum())
                    .add(x.softplus().sum())
                    .add(x.leakyRelu(0.05f).sum())
                    .add(x.abs().plus(1.5f).log().sum())
                    .add(x.abs().plus(0.5f).sqrt().sum())
                    .add(x.scale(0.25f).exp().sum());
        }, 1e-3f, 3e-2f);
    }

    private static Result checkMatmulGradients() {
        Random random = new Random(13);
        Tensor a = randomParameter(3, 4, random);
        Tensor b = randomParameter(4, 2, random);
        return gradientCheck("matmul", new Tensor[]{a, b}, in ->
                in[0].matmul(in[1]).tanh().square().sum()
                        .add(in[0].transpose().matmul(in[0]).sum()), 1e-3f, 2e-2f);
    }

    private static Result checkSoftmaxGradients() {
        Random random = new Random(17);
        Tensor logits = randomParameter(3, 5, random);
        int[] targets = {0, 3, 4};
        return gradientCheck("softmax", new Tensor[]{logits}, in ->
                Losses.crossEntropy(in[0], targets)
                        .add(in[0].softmax().square().sum())
                        .add(Losses.categoricalEntropy(in[0]).mean()), 1e-3f, 2e-2f);
    }

    private static Result checkLayerNormGradients() {
        Random random = new Random(19);
        LayerNorm norm = new LayerNorm(6);
        Tensor input = randomParameter(4, 6, random);
        List<Tensor> parameters = norm.parameters();
        Tensor[] inputs = {input, parameters.get(0), parameters.get(1)};
        return gradientCheck("layernorm", inputs, in -> norm.forward(in[0]).tanh().square().sum(), 1e-3f, 3e-2f);
    }

    private static Result checkAttentionGradients() {
        Random random = new Random(23);
        SequenceEncoder encoder = new SequenceEncoder(4, 8, 2, 1, 16, 8, 0f, random);
        encoder.eval();
        Tensor window = randomParameter(5, 4, random);
        List<Tensor> parameters = encoder.parameters();
        Tensor[] inputs = {window, parameters.get(0), parameters.get(parameters.size() - 1)};
        return gradientCheck("attention", inputs, in -> encoder.encode(in[0]).square().sum(), 2e-3f, 5e-2f);
    }

    private static Result checkGruGradients() {
        Random random = new Random(29);
        GruCell cell = new GruCell(3, 4, random);
        Tensor input = randomParameter(2, 3, random);
        List<Tensor> parameters = cell.parameters();
        Tensor[] inputs = {input, parameters.get(0)};
        return gradientCheck("gru", inputs, in -> {
            Tensor state = cell.initialState(2);
            // unroll three steps so the recurrent path is exercised, not just one cell application
            for (int i = 0; i < 3; i++) {
                state = cell.forward(in[0], state);
            }
            return state.square().sum();
        }, 1e-3f, 4e-2f);
    }

    private static Result checkLossGradients() {
        Random random = new Random(31);
        Tensor mean = randomParameter(3, 2, random);
        Tensor logStandardDeviation = randomParameter(3, 2, random);
        Tensor target = randomParameter(3, 2, random);
        Tensor logits = randomParameter(3, 4, random);
        boolean[][] actions = {{true, false, true, true}, {false, false, true, false}, {true, true, false, false}};
        return gradientCheck("losses", new Tensor[]{mean, logStandardDeviation, logits}, in ->
                Losses.gaussianNll(in[0], in[1], target)
                        .add(Losses.gaussianLogProbability(in[0], in[1], target).mean())
                        .add(Losses.gaussianEntropy(in[1]).mean())
                        .add(Losses.huber(in[0], target, 1f))
                        .add(Losses.bernoulliLogProbability(in[2], actions).mean())
                        .add(Losses.bernoulliEntropy(in[2]).mean())
                        .add(Losses.binaryCrossEntropy(in[2], new float[]{1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1})),
                1e-3f, 3e-2f);
    }

    private static Tensor randomParameter(int rows, int cols, Random random) {
        Tensor tensor = Tensor.param(rows, cols);
        for (int i = 0; i < tensor.size(); i++) {
            tensor.data[i] = (float) (random.nextGaussian() * 0.7);
        }
        return tensor;
    }

    // --------------------------------------------------------------------------------------------------- learning

    /**
     * XOR is the smallest problem that a linear model provably cannot solve, so fitting it end to end proves the
     * hidden layers, their nonlinearity and the optimizer are all actually doing something.
     */
    private static Result checkXorLearning() {
        Random random = new Random(1234);
        Mlp model = new Mlp(2, new int[]{16, 16}, 1, Activation.RELU, Activation.IDENTITY, false, random);
        Adam optimizer = new Adam(model.parameters(), 0.05f);
        LearningRateSchedule schedule = LearningRateSchedule.cosineWithWarmup(0.05f, 0.005f, 20, 400);

        Tensor inputs = Tensor.of(4, 2, 0, 0, 0, 1, 1, 0, 1, 1);
        Tensor targets = Tensor.of(4, 1, 0, 1, 1, 0);

        float loss = Float.MAX_VALUE;
        for (int step = 0; step < 600; step++) {
            schedule.apply(optimizer);
            optimizer.zeroGrad();
            Tensor prediction = model.forward(inputs);
            Tensor objective = Losses.mse(prediction, targets);
            objective.backward();
            optimizer.step();
            loss = objective.item();
        }
        Tensor prediction = model.forward(inputs);
        boolean correct = prediction.data[0] < 0.5f && prediction.data[1] > 0.5f
                && prediction.data[2] > 0.5f && prediction.data[3] < 0.5f;
        return new Result("xor", correct && loss < 0.02f,
                String.format("loss=%.4f outputs=[%.2f %.2f %.2f %.2f]", loss,
                        prediction.data[0], prediction.data[1], prediction.data[2], prediction.data[3]));
    }

    /**
     * Fits a task that is impossible without memory: the answer depends on a value seen several ticks earlier, not on
     * the current tick at all. If the encoder's attention were broken this could not get below chance.
     */
    private static Result checkSequenceLearning() {
        Random random = new Random(4321);
        final int window = 6;
        final int features = 3;
        SequenceEncoder encoder = new SequenceEncoder(features, 16, 2, 1, 32, window, 0f, random);
        Linear head = new Linear(16, 1, true, false, random);

        Module container = new Module() {};
        List<Tensor> parameters = new ArrayList<>(encoder.parameters());
        parameters.addAll(head.parameters());
        Adam optimizer = new Adam(parameters, 0.01f);

        float loss = 0;
        for (int step = 0; step < 400; step++) {
            Tensor input = new Tensor(window, features);
            for (int i = 0; i < input.size(); i++) {
                input.data[i] = (float) random.nextGaussian() * 0.3f;
            }
            // the target is the first tick's first feature, which the model can only produce by looking back
            float flag = random.nextBoolean() ? 1f : -1f;
            input.data[0] = flag;
            Tensor target = Tensor.scalar(flag);

            optimizer.zeroGrad();
            Tensor prediction = head.forward(encoder.encode(input));
            Tensor objective = Losses.mse(prediction, target);
            objective.backward();
            optimizer.step();
            loss = 0.95f * loss + 0.05f * objective.item();
        }

        int correct = 0;
        encoder.eval();
        for (int trial = 0; trial < 50; trial++) {
            Tensor input = new Tensor(window, features);
            for (int i = 0; i < input.size(); i++) {
                input.data[i] = (float) random.nextGaussian() * 0.3f;
            }
            float flag = random.nextBoolean() ? 1f : -1f;
            input.data[0] = flag;
            float prediction = head.forward(encoder.encode(input)).item();
            if (Math.signum(prediction) == Math.signum(flag)) {
                correct++;
            }
        }
        return new Result("sequence", correct >= 45,
                String.format("recalled %d/50 with smoothed loss %.4f (%d parameters)",
                        correct, loss, encoder.parameterCount() + head.parameterCount()));
    }

    private static Result checkGradientClipping() {
        Tensor parameter = Tensor.param(1, 4);
        Adam optimizer = new Adam(List.of(parameter), 0.1f);
        optimizer.setMaxGradientNorm(1f);
        float[] gradient = parameter.gradientOrAllocate();
        java.util.Arrays.fill(gradient, 1000f);
        boolean applied = optimizer.step();
        float afterHugeGradient = parameter.data[0];

        optimizer.zeroGrad();
        gradient = parameter.gradientOrAllocate();
        gradient[0] = Float.NaN;
        boolean rejected = !optimizer.step();
        boolean unchanged = parameter.data[0] == afterHugeGradient;

        return new Result("clipping", applied && rejected && unchanged && Math.abs(afterHugeGradient) < 0.2f,
                String.format("clipped step moved weight by %.4f, non-finite step rejected=%b, weights intact=%b",
                        afterHugeGradient, rejected, unchanged));
    }

    // ------------------------------------------------------------------------------------------------------- data

    private static Result checkRunningStatistics() {
        Random random = new Random(5);
        RunningStatistics statistics = new RunningStatistics(2);
        for (int i = 0; i < 10_000; i++) {
            statistics.observe(new float[]{(float) (random.nextGaussian() * 3 + 10), (float) random.nextGaussian()});
        }
        float mean = statistics.meanOf(0);
        float deviation = statistics.deviationOf(0);
        float[] normalized = statistics.normalized(new float[]{10f, 0f});
        boolean ok = Math.abs(mean - 10) < 0.2f && Math.abs(deviation - 3) < 0.2f && Math.abs(normalized[0]) < 0.2f;
        return new Result("statistics", ok, String.format("mean=%.3f sd=%.3f normalized=%.3f", mean, deviation, normalized[0]));
    }

    private static Result checkReplayBuffer() {
        Random random = new Random(9);
        ReplayBuffer<Integer> buffer = new ReplayBuffer<>(64, random);
        for (int i = 0; i < 200; i++) {
            buffer.add(i, i == 150 ? 100f : 1f);
        }
        boolean evicted = buffer.size() == 64 && buffer.totalAdded() == 200;
        int hits = 0;
        for (int trial = 0; trial < 400; trial++) {
            for (ReplayBuffer.Sample<Integer> sample : buffer.sample(4)) {
                if (sample.value == 150) {
                    hits++;
                }
            }
        }
        // with a priority a hundred times the rest, the high priority item must dominate a uniform 1/64 share
        boolean prioritized = hits > 400 * 4 / 64 * 3;
        return new Result("replay", evicted && prioritized,
                String.format("size=%d added=%d high-priority draws=%d of %d", buffer.size(), buffer.totalAdded(), hits, 1600));
    }

    private static Result checkEpisodicMemory() {
        Random random = new Random(77);
        EpisodicMemory memory = new EpisodicMemory(8, 500, 6, 10, random);
        long now = System.currentTimeMillis();

        float[] situation = new float[8];
        for (int i = 0; i < 8; i++) {
            situation[i] = (float) random.nextGaussian();
        }
        // the same situation met twenty times, with a consistent outcome
        for (int i = 0; i < 20; i++) {
            float[] noisy = situation.clone();
            for (int j = 0; j < 8; j++) {
                noisy[j] += (float) (random.nextGaussian() * 0.001);
            }
            memory.remember(noisy, 3, new float[]{1f}, 12 + random.nextGaussian() * 0.5, true, now);
        }
        // plus a hundred unrelated ones
        for (int i = 0; i < 100; i++) {
            float[] other = new float[8];
            for (int j = 0; j < 8; j++) {
                other[j] = (float) random.nextGaussian();
            }
            memory.remember(other, 3, new float[]{0f}, 40 + random.nextGaussian(), false, now);
        }

        EpisodicMemory.Estimate estimate = memory.estimate(situation, 3, 8);
        boolean recalled = !estimate.isEmpty() && Math.abs(estimate.outcome - 12) < 3 && estimate.confidence > 0.3;
        boolean merged = memory.size() < 110 && memory.getMerges() >= 15;
        boolean contextIsolated = memory.estimate(situation, 99, 8).isEmpty();
        return new Result("memory", recalled && merged && contextIsolated,
                String.format("%s, %d records from 120 experiences, %d merges, context isolation=%b",
                        estimate, memory.size(), memory.getMerges(), contextIsolated));
    }

    private static Result checkCheckpointRoundTrip() {
        try {
            Random random = new Random(99);
            Mlp original = new Mlp(4, new int[]{8}, 2, Activation.GELU, Activation.TANH, true, random);
            Tensor probe = Tensor.of(1, 4, 0.3f, -0.2f, 0.9f, 0.1f);
            float[] before = original.forward(probe).data.clone();

            Path file = Files.createTempDirectory("barelentless-ml").resolve("model.brlm");
            LinkedHashMap<String, String> metadata = new LinkedHashMap<>();
            metadata.put("samples", "12345");
            ModelIO.save(original, file, metadata);

            Mlp restored = new Mlp(4, new int[]{8}, 2, Activation.GELU, Activation.TANH, true, new Random(1));
            ModelIO.LoadReport report = ModelIO.load(restored, file);
            float[] after = restored.forward(probe).data.clone();

            boolean identical = true;
            for (int i = 0; i < before.length; i++) {
                identical &= Math.abs(before[i] - after[i]) < 1e-6f;
            }
            boolean metadataKept = "12345".equals(ModelIO.peekMetadata(file).get("samples"));
            Files.deleteIfExists(file);
            return new Result("checkpoint", identical && report.isClean() && metadataKept,
                    report + ", outputs identical=" + identical + ", metadata kept=" + metadataKept);
        } catch (Exception e) {
            return new Result("checkpoint", false, e.toString());
        }
    }

    public static void main(String[] args) {
        List<Result> results = runAll();
        int failed = 0;
        for (Result result : results) {
            System.out.println(result);
            if (!result.passed) {
                failed++;
            }
        }
        System.out.println(failed == 0
                ? "all " + results.size() + " checks passed"
                : failed + " of " + results.size() + " checks FAILED");
        if (failed > 0) {
            System.exit(1);
        }
    }
}
