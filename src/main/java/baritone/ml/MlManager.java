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

import baritone.Baritone;
import baritone.api.event.events.TickEvent;
import baritone.api.event.events.WorldEvent;
import baritone.api.ml.ILearningAPI;
import baritone.api.ml.Losses;
import baritone.api.ml.Tensor;
import baritone.api.ml.data.ReplayBuffer;
import baritone.api.ml.data.RunningStatistics;
import baritone.api.ml.io.ModelIO;
import baritone.api.ml.memory.EpisodicMemory;
import baritone.api.ml.optim.Adam;
import baritone.api.ml.optim.LearningRateSchedule;
import baritone.api.ml.rl.PolicyOutput;
import baritone.api.ml.rl.PpoTrainer;
import baritone.api.ml.rl.Trajectory;
import baritone.api.pathing.movement.IMovement;
import baritone.api.pathing.path.IPathExecutor;
import baritone.api.utils.Helper;
import baritone.api.utils.BetterBlockPos;
import baritone.api.utils.Rotation;
import baritone.behavior.Behavior;
import baritone.pathing.movement.CalculationContext;
import baritone.utils.BlockStateInterface;
import net.minecraft.world.phys.Vec3;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Owns the learning: what gets recorded, what gets trained, what the live models are, and when any of it touches disk.
 * <p>
 * The hard constraint that shapes all of it is that training must never be visible in the frame time. So the client
 * thread only ever appends to lock-free-enough buffers and reads a volatile model reference; a single daemon thread
 * does every gradient step on its own copy of the model and publishes a snapshot when it is done. There is no lock
 * between them and no point at which a half-updated set of weights can be read mid-tick.
 * <p>
 * The second constraint is that nothing here may be able to make the bot worse in a way the player cannot undo. Every
 * model starts initialized near zero, so enabling learning changes nothing until it has actually learned something;
 * every file is written atomically; a model that goes non-finite is dropped and rebuilt rather than saved over a good
 * checkpoint; and all of it is behind settings that default to off.
 *
 * @author Barelentless
 */
public final class MlManager extends Behavior implements ILearningAPI, Helper {

    private static final String AIM_MODEL_FILE = "aim.brlm";
    private static final String MEMORY_FILE = "memory.brlm";
    private static final String MOVEMENT_MODEL_FILE = "movement.brlm";
    private static final String STATISTICS_FILE = "features.bin";

    /**
     * How many ticks ahead the human's eventual rotation is taken as the target when recording a demonstration.
     * Roughly a third of a second: long enough that it captures where the player was going, short enough that it is
     * still the same intention.
     */
    private static final int DEMONSTRATION_LOOKAHEAD = 6;

    private final Path directory;
    private final Random random = new Random();

    private final EpisodicMemory memory;
    private final ReplayBuffer<AimSample> aimSamples;
    private final ReplayBuffer<MovementExperience> movementSamples;
    private RunningStatistics featureStatistics;

    /**
     * The model the game thread reads. Replaced wholesale by the trainer; never mutated in place.
     */
    private volatile AimModel liveAimModel;

    // architecture, kept so that a published snapshot is built exactly like the model it copies
    private int modelWindow;
    private int modelDimension;
    private int modelHeads;
    private int modelDepth;

    /**
     * The trainer's own copy. Touched only by the training thread.
     */
    private AimModel trainingAimModel;
    private Adam aimOptimizer;
    private LearningRateSchedule aimSchedule;

    private Thread trainer;
    private final AtomicBoolean running = new AtomicBoolean();
    private final Object trainerSignal = new Object();

    /**
     * The movement policy the game thread reads, and the copy the trainer owns. Data collected under an older
     * snapshot is still valid to train on: each step stores the log probability of the policy that produced it, and
     * that is exactly what the importance ratio in PPO corrects for.
     */
    private volatile MovementPolicyModel liveMovementPolicy;
    private MovementPolicyModel trainingMovementPolicy;
    private PpoTrainer movementTrainer;
    private Adam movementOptimizer;

    private final java.util.concurrent.ConcurrentLinkedQueue<Trajectory> finishedTrajectories =
            new java.util.concurrent.ConcurrentLinkedQueue<>();
    private Trajectory movementTrajectory;
    private double previousDistance = Double.NaN;
    private float previousHealth;
    private long policySteps;
    private float lastPolicyReport = Float.NaN;
    private String lastPolicyReportText = "";

    private NeuralAimShaper aimShaper;
    private LearnedCautionShaper cautionShaper;
    private MovementPolicyShaper policyShaper;

    // per-tick state shared with the shapers
    private float[] stateFeatures = new float[StateEncoder.FEATURES];
    private long stateFeaturesTick = -1;
    private long tick;

    // movement outcome tracking
    private IMovement trackedMovement;
    private float[] trackedSituation;
    private int trackedTicks;
    private float trackedStartHealth;

    // demonstration recording
    private final Deque<DemonstrationFrame> demonstrationFrames = new ArrayDeque<>();

    // inference budget
    private double inferenceNanosEma;
    private boolean throttled;
    private long throttleChanges;

    // diagnostics
    private long trainingSteps;
    private float lastLoss = Float.NaN;
    private long inferenceFailures;
    private long demonstrationsRecorded;
    private long lastSaveMillis;
    private String lastError = "";

    private static final class DemonstrationFrame {

        final float[] state;
        final Rotation rotation;
        final Rotation nextRotation;

        DemonstrationFrame(float[] state, Rotation rotation, Rotation nextRotation) {
            this.state = state;
            this.rotation = rotation;
            this.nextRotation = nextRotation;
        }
    }

    public MlManager(Baritone baritone) {
        super(baritone);
        this.directory = baritone.getDirectory().resolve("ml");
        this.memory = new EpisodicMemory(LearnedCostAdjuster.KEY_FEATURES, 200_000, 8, 12, new Random(0xBA217091L));
        this.aimSamples = new ReplayBuffer<>(20_000, this.random);
        this.movementSamples = new ReplayBuffer<>(50_000, this.random);
        this.featureStatistics = new RunningStatistics(StateEncoder.FEATURES);
    }

    // -------------------------------------------------------------------------------------------------- lifecycle

    /**
     * Builds the models and starts the trainer. Safe to call repeatedly; only the first call does anything.
     */
    @Override
    public synchronized void start() {
        if (this.running.get()) {
            return;
        }
        if (this.liveAimModel == null) {
            createModels();
            loadFromDisk();
        }
        if (this.aimShaper == null) {
            this.aimShaper = new NeuralAimShaper(this);
            // above the human aim shaper: when the model is confident it should be the thing that decides, and when
            // it is not, its own blend hands control back rather than needing a lower priority to do it
            this.baritone.getControlAPI().registerRotationShaper("neuralAim", 600, this.aimShaper);
        }
        if (this.cautionShaper == null) {
            this.cautionShaper = new LearnedCautionShaper(this);
            this.baritone.getControlAPI().registerInputShaper("learnedCaution", 600, this.cautionShaper);
        }
        if (this.policyShaper == null) {
            this.policyShaper = new MovementPolicyShaper(this);
            // after caution: caution decides how fast is safe here, the policy works within that
            this.baritone.getControlAPI().registerInputShaper("movementPolicy", 700, this.policyShaper);
        }
        this.running.set(true);
        this.trainer = new Thread(this::trainLoop, "Barelentless ML trainer");
        this.trainer.setDaemon(true);
        this.trainer.setPriority(Thread.MIN_PRIORITY);
        this.trainer.start();
    }

    /**
     * Stops training and saves. The models stay loaded, so inference continues to work.
     */
    @Override
    public synchronized void stop() {
        this.running.set(false);
        synchronized (this.trainerSignal) {
            this.trainerSignal.notifyAll();
        }
        this.trainer = null;
        save();
    }

    private void createModels() {
        int window = Math.max(2, Baritone.settings().mlAimWindow.value);
        int dimension = Math.max(8, Baritone.settings().mlAimDimension.value);
        // heads have to divide the model dimension; pick the largest sensible divisor rather than rejecting the
        // setting, because a player who sets dimension to 48 should not have to also know about head counts
        int heads = 4;
        while (heads > 1 && dimension % heads != 0) {
            heads--;
        }
        int depth = Math.max(1, Baritone.settings().mlAimDepth.value);
        this.modelWindow = window;
        this.modelDimension = dimension;
        this.modelHeads = heads;
        this.modelDepth = depth;
        this.trainingAimModel = new AimModel(window, dimension, heads, depth, this.random);
        this.liveAimModel = new AimModel(window, dimension, heads, depth, this.random);
        this.liveAimModel.loadStateFrom(this.trainingAimModel);
        this.aimOptimizer = new Adam(this.trainingAimModel.parameters(), 3e-4f, 0.9f, 0.999f, 1e-8f, 1e-5f);
        this.aimSchedule = LearningRateSchedule.cosineWithWarmup(3e-4f, 3e-5f, 200, 20_000);

        int policyHidden = Math.max(16, Baritone.settings().mlMovementDimension.value);
        this.trainingMovementPolicy = new MovementPolicyModel(policyHidden, this.random);
        this.liveMovementPolicy = new MovementPolicyModel(policyHidden, this.random);
        this.liveMovementPolicy.loadStateFrom(this.trainingMovementPolicy);
        this.movementOptimizer = new Adam(this.trainingMovementPolicy.parameters(), 1e-4f);
        this.movementTrainer = new PpoTrainer(this.trainingMovementPolicy, this.movementOptimizer, this.random)
                .setEpochs(3)
                .setMiniBatch(64)
                // a small entropy bonus, because this policy only ever sees the data its own behaviour produces:
                // collapsing early means never discovering that a slower approach would have worked
                .setEntropyCoefficient(0.004f)
                .setTargetKl(0.02f);
    }

    @Override
    public boolean isRunning() {
        return this.running.get();
    }

    public AimModel getAimModel() {
        return this.liveAimModel;
    }

    public MovementPolicyModel getMovementPolicy() {
        return this.liveMovementPolicy;
    }

    @Override
    public EpisodicMemory getMemory() {
        return this.memory;
    }

    // ------------------------------------------------------------------------------------------------------- tick

    @Override
    public void onTick(TickEvent event) {
        if (event.getType() == TickEvent.Type.OUT || ctx.player() == null) {
            return;
        }
        if (!Baritone.settings().mlEnabled.value) {
            return;
        }
        if (!this.running.get()) {
            start();
        }
        this.tick++;
        try {
            float[] features = currentStateFeatures();
            this.featureStatistics.observe(features);
            trackMovement(features);
            recordDemonstration(features);
        } catch (RuntimeException e) {
            onInferenceFailure("tick", e);
        }
    }

    /**
     * The encoded state for this tick, computed once and shared by everything that needs it.
     */
    public float[] currentStateFeatures() {
        if (this.stateFeaturesTick == this.tick) {
            return this.stateFeatures;
        }
        BlockStateInterface bsi = ((Baritone) this.baritone).bsi;
        float[] raw = StateEncoder.encode(ctx, bsi, currentMovement(), this.trackedTicks);
        this.featureStatistics.normalize(raw);
        this.stateFeatures = raw;
        this.stateFeaturesTick = this.tick;
        return raw;
    }

    private IMovement currentMovement() {
        IPathExecutor executor = this.baritone.getPathingBehavior().getCurrent();
        if (executor == null || executor.getPosition() >= executor.getPath().movements().size()) {
            return null;
        }
        return executor.getPath().movements().get(executor.getPosition());
    }

    /**
     * Watches the executing movement and closes the loop when it ends: the estimate that was made, against the ticks
     * it actually took and the damage it cost.
     */
    private void trackMovement(float[] features) {
        IMovement movement = currentMovement();
        if (movement == this.trackedMovement) {
            if (movement != null) {
                this.trackedTicks++;
            }
            return;
        }
        if (this.trackedMovement != null && this.trackedSituation != null && this.trackedTicks > 0) {
            float damage = Math.max(0, this.trackedStartHealth - ctx.player().getHealth());
            boolean successful = this.trackedTicks < this.trackedMovement.getCost() * 3 + 20;
            finishMovementTrajectory(successful, this.policyShaper == null ? 0f : this.policyShaper.getLastValue());
            MovementExperience experience = new MovementExperience(
                    this.trackedSituation,
                    movementKind(this.trackedMovement),
                    this.trackedMovement.getCost(),
                    this.trackedTicks,
                    successful,
                    damage,
                    System.currentTimeMillis());
            this.movementSamples.add(experience, experience.priority());
            rememberMovement(experience);
        }
        this.trackedMovement = movement;
        this.trackedSituation = movement == null ? null : features.clone();
        this.trackedTicks = 0;
        this.trackedStartHealth = ctx.player().getHealth();
    }

    /**
     * Files a completed movement into the episodic memory under the same key the pathfinder will later look it up by.
     */
    private void rememberMovement(MovementExperience experience) {
        BlockStateInterface bsi = ((Baritone) this.baritone).bsi;
        if (bsi == null) {
            return;
        }
        IMovement movement = this.trackedMovement;
        if (movement == null) {
            return;
        }
        try {
            float[] key = LearnedCostAdjuster.key(
                    movement.getDest().y - movement.getSrc().y,
                    bsi.get0(movement.getSrc().x, movement.getSrc().y - 1, movement.getSrc().z),
                    bsi.get0(movement.getDest().x, movement.getDest().y, movement.getDest().z),
                    bsi.get0(movement.getDest().x, movement.getDest().y - 1, movement.getDest().z),
                    bsi.get0(movement.getDest().x, movement.getDest().y + 1, movement.getDest().z));
            this.memory.remember(key, experience.movementKind, null,
                    experience.memoryOutcome(), experience.successful, experience.timestamp);
        } catch (RuntimeException e) {
            onInferenceFailure("memory", e);
        }
    }

    /**
     * A stable id per movement class, used as the memory's context so a pillar memory can never answer a parkour
     * question. String hashing is specified by the language, so this id means the same thing in a memory file written
     * months ago as it does today.
     */
    private static int movementKind(IMovement movement) {
        return Math.abs(movement.getClass().getSimpleName().hashCode()) % 4096;
    }

    /**
     * Called by the movement policy shaper each tick it acts.
     * <p>
     * The reward for the previous step is settled here rather than when that step was taken, because the world state
     * visible now is the consequence of that action: progress made towards the destination, time spent, damage taken.
     * Rewarding a step at the moment it happens would be rewarding an intention.
     */
    public void recordPolicyStep(Trajectory.Step step) {
        IMovement movement = currentMovement();
        if (movement == null || ctx.player() == null) {
            return;
        }
        if (this.movementTrajectory == null) {
            this.movementTrajectory = new Trajectory(movementKind(movement));
            this.previousDistance = Double.NaN;
        }
        double distance = distanceTo(movement);
        float health = ctx.player().getHealth();
        if (!Double.isNaN(this.previousDistance) && this.movementTrajectory.size() > 0) {
            float progress = (float) (this.previousDistance - distance);
            float damage = Math.max(0f, this.previousHealth - health);
            // progress dominates, a small per-tick cost stops dithering from being free, damage is expensive
            this.movementTrajectory.reward(progress * 2f - 0.02f - damage * 0.5f);
        }
        this.previousDistance = distance;
        this.previousHealth = health;
        this.movementTrajectory.add(step);
        this.policySteps++;
        if (this.movementTrajectory.size() >= 400) {
            // something is badly stuck; close the trajectory rather than letting it grow without bound
            finishMovementTrajectory(false, step.value);
        }
    }

    private double distanceTo(IMovement movement) {
        BetterBlockPos destination = movement.getDest();
        Vec3 position = ctx.player().position();
        double dx = destination.x + 0.5 - position.x;
        double dy = destination.y - position.y;
        double dz = destination.z + 0.5 - position.z;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /**
     * Closes the current trajectory and hands it to the trainer.
     *
     * @param successful     Whether the movement completed rather than failing
     * @param bootstrapValue The policy's estimate of the state after the last step; zero when the episode truly ended
     */
    private void finishMovementTrajectory(boolean successful, float bootstrapValue) {
        Trajectory trajectory = this.movementTrajectory;
        this.movementTrajectory = null;
        this.previousDistance = Double.NaN;
        if (trajectory == null || trajectory.size() < 2) {
            return;
        }
        trajectory.reward(successful ? 2f : -3f);
        if (successful) {
            trajectory.steps().get(trajectory.size() - 1).terminal = true;
        }
        trajectory.finish(successful ? 0f : bootstrapValue, 0.99f, 0.95f);
        this.finishedTrajectories.add(trajectory);
        // bound the queue: if the trainer cannot keep up, the oldest data is also the most stale
        while (this.finishedTrajectories.size() > 64) {
            this.finishedTrajectories.poll();
        }
        signalTrainer();
    }

    /**
     * Records what the player does with their own view.
     * <p>
     * The label is the rotation the player actually ended up at a few ticks later, not the one they had at the time -
     * otherwise the sample would just say "you were where you were". Using the eventual rotation as the target makes
     * each sample a genuine demonstration of how a human closes a gap of a given size at a given speed.
     */
    private void recordDemonstration(float[] features) {
        if (!Baritone.settings().mlLearnFromPlayer.value) {
            return;
        }
        // only while the bot is not the one aiming; otherwise this would be learning from itself
        if (this.baritone.getControlAPI().lastAppliedRotation() != null && this.baritone.getPathingBehavior().isPathing()) {
            this.demonstrationFrames.clear();
            return;
        }
        Rotation now = ctx.playerRotations();
        Rotation previous = this.demonstrationFrames.isEmpty() ? now : this.demonstrationFrames.peekLast().rotation;
        this.demonstrationFrames.addLast(new DemonstrationFrame(features.clone(), now, null));

        AimModel model = this.liveAimModel;
        int window = model == null ? 8 : model.getWindow();
        int needed = window + DEMONSTRATION_LOOKAHEAD;
        while (this.demonstrationFrames.size() > needed) {
            this.demonstrationFrames.removeFirst();
        }
        if (this.demonstrationFrames.size() < needed) {
            return;
        }
        List<DemonstrationFrame> frames = new ArrayList<>(this.demonstrationFrames);
        DemonstrationFrame anchor = frames.get(window - 1);
        Rotation eventual = frames.get(frames.size() - 1).rotation;
        Rotation next = frames.get(window).rotation;

        float yawDelta = Rotation.normalizeYaw(next.getYaw() - anchor.rotation.getYaw());
        float pitchDelta = next.getPitch() - anchor.rotation.getPitch();
        if (Math.abs(yawDelta) < 0.01f && Math.abs(pitchDelta) < 0.01f) {
            // a tick where the player did not move the mouse teaches nothing about how they move it
            return;
        }
        float[][] rows = new float[window][AimModel.INPUT_FEATURES];
        for (int i = 0; i < window; i++) {
            DemonstrationFrame frame = frames.get(i);
            float yawError = Rotation.normalizeYaw(eventual.getYaw() - frame.rotation.getYaw());
            float pitchError = eventual.getPitch() - frame.rotation.getPitch();
            Rotation before = i == 0 ? frame.rotation : frames.get(i - 1).rotation;
            float previousYawError = Rotation.normalizeYaw(eventual.getYaw() - before.getYaw());
            float previousPitchError = eventual.getPitch() - before.getPitch();
            System.arraycopy(frame.state, 0, rows[i], 0, StateEncoder.FEATURES);
            int index = StateEncoder.FEATURES;
            rows[i][index++] = yawError / 45f;
            rows[i][index++] = pitchError / 45f;
            rows[i][index++] = (yawError - previousYawError) / 45f;
            rows[i][index++] = (pitchError - previousPitchError) / 45f;
            rows[i][index++] = 0f;
            rows[i][index++] = 1f;
            rows[i][index++] = 1f;
            rows[i][index] = 0f;
        }
        AimSample sample = new AimSample(rows, yawDelta, pitchDelta, true);
        this.aimSamples.add(sample, sample.priority());
        this.demonstrationsRecorded++;
        signalTrainer();
        // consume the anchor so the next sample starts one tick later rather than re-emitting this one
        this.demonstrationFrames.removeFirst();
    }

    /**
     * Called by the aim shaper each tick with the window it built. Used to keep the bot's own behaviour in the replay
     * buffer at a low priority, which gives the trainer negative examples to contrast demonstrations against.
     */
    public void recordAimContext(Deque<float[]> window, float yawError, float pitchError,
                                 Rotation current, Rotation previous) {
        if (previous == null || window.isEmpty() || !Baritone.settings().mlRecordSelf.value) {
            return;
        }
        float yawDelta = Rotation.normalizeYaw(current.getYaw() - previous.getYaw());
        float pitchDelta = current.getPitch() - previous.getPitch();
        if (Math.abs(yawDelta) < 0.05f && Math.abs(pitchDelta) < 0.05f) {
            return;
        }
        float[][] rows = window.toArray(new float[0][]);
        AimSample sample = new AimSample(rows, yawDelta, pitchDelta, false);
        this.aimSamples.add(sample, sample.priority() * 0.25f);
    }

    @Override
    public void onWorldEvent(WorldEvent event) {
        // the world is gone; a trajectory that ends in a dimension change did not end because of anything the
        // policy did, and training on it would teach exactly the wrong lesson
        this.movementTrajectory = null;
        this.previousDistance = Double.NaN;
        this.demonstrationFrames.clear();
        this.trackedMovement = null;
        this.trackedSituation = null;
        this.trackedTicks = 0;
    }

    // --------------------------------------------------------------------------------------------------- training

    private void signalTrainer() {
        synchronized (this.trainerSignal) {
            this.trainerSignal.notifyAll();
        }
    }

    private void trainLoop() {
        while (this.running.get()) {
            try {
                int batch = Math.max(4, Baritone.settings().mlBatchSize.value);
                if (this.aimSamples.size() < batch * 2) {
                    // no aim data yet, but movement trajectories may still be waiting
                    if (!trainMovementPolicy()) {
                        synchronized (this.trainerSignal) {
                            this.trainerSignal.wait(2000);
                        }
                    }
                    continue;
                }
                trainAimStep(batch);
                trainMovementPolicy();
                if (this.trainingSteps % 200 == 0) {
                    publishModel();
                }
                if (this.trainingSteps % 2000 == 0) {
                    save();
                }
                // deliberately unhurried: this is a background thread on a machine that is also running a game
                Thread.sleep(Math.max(1, Baritone.settings().mlTrainingDelayMs.value));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            } catch (Throwable t) {
                this.lastError = t.toString();
                this.inferenceFailures++;
                try {
                    Thread.sleep(5000);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }

    /**
     * One supervised step: predict the view movement each sample actually made, under a Gaussian likelihood so the
     * model learns its own uncertainty alongside the mean.
     */
    private void trainAimStep(int batchSize) {
        List<ReplayBuffer.Sample<AimSample>> batch = this.aimSamples.sample(batchSize);
        if (batch.isEmpty()) {
            return;
        }
        this.trainingAimModel.train();
        this.aimSchedule.apply(this.aimOptimizer);
        this.aimOptimizer.zeroGrad();

        List<Tensor> encodedRows = new ArrayList<>(batch.size());
        Tensor targets = new Tensor(batch.size(), 2);
        Tensor weights = new Tensor(batch.size(), 1);
        for (int i = 0; i < batch.size(); i++) {
            ReplayBuffer.Sample<AimSample> sample = batch.get(i);
            encodedRows.add(this.trainingAimModel.encode(sample.value.windowTensor()));
            targets.data[i * 2] = sample.value.yawDelta;
            targets.data[i * 2 + 1] = sample.value.pitchDelta;
            // importance weight from prioritized replay, scaled down for the bot's own behaviour so that
            // demonstrations dominate the objective
            weights.data[i] = sample.weight * (sample.value.demonstration ? 1f : 0.3f);
        }
        Tensor encoded = Tensor.concatRows(encodedRows.toArray(new Tensor[0]));
        PolicyOutput output = this.trainingAimModel.forwardEncoded(encoded);
        Tensor perSample = Losses.gaussianLogProbability(output.mean, output.logStandardDeviation, targets).neg();
        Tensor loss = perSample.mul(weights).mean();
        loss.backward();
        boolean applied = this.aimOptimizer.step();
        this.trainingAimModel.eval();

        this.trainingSteps++;
        this.lastLoss = loss.item();
        this.aimSamples.annealBeta(1e-4f);

        // refresh priorities from the residual: samples the model now predicts well stop being replayed
        for (int i = 0; i < batch.size(); i++) {
            float residual = Math.abs(output.mean.data[i * 2] - targets.data[i * 2])
                    + Math.abs(output.mean.data[i * 2 + 1] - targets.data[i * 2 + 1]);
            this.aimSamples.updatePriority(batch.get(i).index, 0.1f + residual);
        }
        if (!applied || !this.trainingAimModel.isFinite()) {
            // the weights are gone; rebuilding costs whatever was learned since the last checkpoint, which is far
            // better than publishing a model that aims at NaN
            this.lastError = "training diverged, model reset from last checkpoint";
            this.trainingAimModel.loadStateFrom(this.liveAimModel);
            this.aimOptimizer.resetState();
        }
    }

    /**
     * Runs one PPO update if enough trajectories have finished.
     *
     * @return Whether an update was performed
     */
    private boolean trainMovementPolicy() {
        int required = Math.max(2, Baritone.settings().mlMovementBatch.value);
        if (this.finishedTrajectories.size() < required || this.movementTrainer == null) {
            return false;
        }
        List<Trajectory> batch = new ArrayList<>(required);
        Trajectory trajectory;
        while (batch.size() < required && (trajectory = this.finishedTrajectories.poll()) != null) {
            batch.add(trajectory);
        }
        if (batch.isEmpty()) {
            return false;
        }
        PpoTrainer.Report report = this.movementTrainer.update(batch);
        this.lastPolicyReport = report.policyLoss;
        this.lastPolicyReportText = report.toString();
        if (!this.trainingMovementPolicy.isFinite()) {
            this.lastError = "movement policy diverged, restored from the live snapshot";
            this.trainingMovementPolicy.loadStateFrom(this.liveMovementPolicy);
            this.movementOptimizer.resetState();
            return true;
        }
        MovementPolicyModel snapshot =
                new MovementPolicyModel(Math.max(16, Baritone.settings().mlMovementDimension.value), this.random);
        snapshot.loadStateFrom(this.trainingMovementPolicy);
        snapshot.eval();
        this.liveMovementPolicy = snapshot;
        return true;
    }

    /**
     * Publishes the trainer's weights to the model the game thread reads.
     */
    private void publishModel() {
        AimModel snapshot = new AimModel(this.modelWindow, this.modelDimension, this.modelHeads, this.modelDepth, this.random);
        snapshot.loadStateFrom(this.trainingAimModel);
        snapshot.eval();
        if (snapshot.isFinite()) {
            this.liveAimModel = snapshot;
        }
    }

    // ------------------------------------------------------------------------------------------------ persistence

    @Override
    public synchronized void save() {
        if (this.liveAimModel == null) {
            return;
        }
        try {
            Files.createDirectories(this.directory);
            Map<String, String> metadata = new LinkedHashMap<>();
            metadata.put("steps", Long.toString(this.trainingSteps));
            metadata.put("features", Integer.toString(AimModel.INPUT_FEATURES));
            metadata.put("window", Integer.toString(this.liveAimModel.getWindow()));
            metadata.put("demonstrations", Long.toString(this.demonstrationsRecorded));
            // the live snapshot, not the trainer's working copy: the trainer mutates its weights in place, so writing
            // those from another thread could put a half-applied optimizer step on disk
            ModelIO.save(this.liveAimModel, this.directory.resolve(AIM_MODEL_FILE), metadata);
            Map<String, String> policyMetadata = new LinkedHashMap<>();
            policyMetadata.put("features", Integer.toString(StateEncoder.FEATURES));
            policyMetadata.put("steps", Long.toString(this.policySteps));
            ModelIO.save(this.liveMovementPolicy, this.directory.resolve(MOVEMENT_MODEL_FILE), policyMetadata);
            this.memory.save(this.directory.resolve(MEMORY_FILE));
            try (DataOutputStream out = new DataOutputStream(new GZIPOutputStream(
                    new BufferedOutputStream(Files.newOutputStream(this.directory.resolve(STATISTICS_FILE)))))) {
                this.featureStatistics.write(out);
            }
            this.lastSaveMillis = System.currentTimeMillis();
        } catch (IOException | RuntimeException e) {
            this.lastError = "save failed: " + e;
        }
    }

    private void loadFromDisk() {
        Path modelFile = this.directory.resolve(AIM_MODEL_FILE);
        if (Files.exists(modelFile)) {
            try {
                Map<String, String> metadata = ModelIO.peekMetadata(modelFile);
                int features = Integer.parseInt(metadata.getOrDefault("features", "-1"));
                if (features != AimModel.INPUT_FEATURES) {
                    this.lastError = "aim checkpoint was trained on " + features
                            + " features, this build uses " + AimModel.INPUT_FEATURES + "; starting fresh";
                } else {
                    ModelIO.LoadReport report = ModelIO.load(this.trainingAimModel, modelFile);
                    this.liveAimModel.loadStateFrom(this.trainingAimModel);
                    this.trainingSteps = Long.parseLong(metadata.getOrDefault("steps", "0"));
                    this.demonstrationsRecorded = Long.parseLong(metadata.getOrDefault("demonstrations", "0"));
                    if (!report.isClean()) {
                        this.lastError = "aim checkpoint partially loaded: " + report;
                    }
                }
            } catch (IOException | RuntimeException e) {
                this.lastError = "could not load aim model: " + e;
            }
        }
        Path policyFile = this.directory.resolve(MOVEMENT_MODEL_FILE);
        if (Files.exists(policyFile)) {
            try {
                int features = Integer.parseInt(ModelIO.peekMetadata(policyFile).getOrDefault("features", "-1"));
                if (features != StateEncoder.FEATURES) {
                    this.lastError = "movement policy was trained on " + features + " features, this build uses "
                            + StateEncoder.FEATURES + "; starting fresh";
                } else {
                    ModelIO.load(this.trainingMovementPolicy, policyFile);
                    this.liveMovementPolicy.loadStateFrom(this.trainingMovementPolicy);
                }
            } catch (IOException | RuntimeException e) {
                this.lastError = "could not load movement policy: " + e;
            }
        }
        Path memoryFile = this.directory.resolve(MEMORY_FILE);
        if (Files.exists(memoryFile)) {
            try {
                this.memory.load(memoryFile);
            } catch (IOException | RuntimeException e) {
                this.lastError = "could not load memory: " + e;
            }
        }
        Path statisticsFile = this.directory.resolve(STATISTICS_FILE);
        if (Files.exists(statisticsFile)) {
            try (DataInputStream in = new DataInputStream(new GZIPInputStream(
                    new BufferedInputStream(Files.newInputStream(statisticsFile))))) {
                RunningStatistics loaded = RunningStatistics.read(in);
                // statistics are only meaningful together with the feature layout they were gathered under; a
                // mismatch means the features changed and the old ones would normalize into nonsense
                if (loaded.getFeatures() == StateEncoder.FEATURES) {
                    this.featureStatistics = loaded;
                }
            } catch (IOException | RuntimeException e) {
                this.lastError = "could not load feature statistics: " + e;
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------- pathing

    /**
     * Creates the cost adjuster for one path calculation, or {@code null} when learned costs are off or the memory is
     * too sparse to be worth consulting.
     */
    public LearnedCostAdjuster newCostAdjuster(CalculationContext context) {
        if (!Baritone.settings().mlEnabled.value || !Baritone.settings().mlLearnedCosts.value) {
            return null;
        }
        if (this.memory.size() < 200) {
            return null;
        }
        return new LearnedCostAdjuster(this.memory, context.bsi,
                Math.max(0, Math.min(1, Baritone.settings().mlCostInfluence.value)));
    }

    // ------------------------------------------------------------------------------------------------ diagnostics

    /**
     * Whether a model may run on this tick.
     * <p>
     * Inference happens inside the game loop, so its cost is not somebody else's problem: a model that takes four
     * milliseconds on a machine that has sixteen to spare is fine, and the same model on a machine already dropping
     * frames is the thing the player will blame. When the measured cost exceeds its budget the models are throttled
     * rather than silently allowed to eat the tick. One tick in eight is still let through, so if the machine frees
     * up the estimate recovers on its own instead of staying throttled until a restart.
     */
    public boolean canInfer() {
        if (!this.throttled) {
            return true;
        }
        return (this.tick & 7L) == 0L;
    }

    /**
     * Reports how long a model took this tick. The estimate is an exponential average, because what matters is the
     * sustained cost, not one slow tick during a chunk load.
     */
    public void recordInference(long nanos) {
        this.inferenceNanosEma = this.inferenceNanosEma == 0
                ? nanos
                : this.inferenceNanosEma * 0.98 + nanos * 0.02;
        double budget = Math.max(0.1, Baritone.settings().mlMaxInferenceMs.value) * 1_000_000.0;
        boolean over = this.inferenceNanosEma > budget;
        if (over != this.throttled) {
            this.throttled = over;
            this.throttleChanges++;
            logDirect(over
                    ? String.format("Barelentless ML: inference is costing %.2f ms a tick, above the %.2f ms budget; "
                            + "throttling the models. Raise mlMaxInferenceMs, or lower mlAimDimension.",
                            this.inferenceNanosEma / 1e6, budget / 1e6)
                    : "Barelentless ML: inference is back within budget.");
        }
    }

    /**
     * The measured cost of model inference, in milliseconds per tick.
     */
    public double getInferenceMillis() {
        return this.inferenceNanosEma / 1e6;
    }

    public boolean isThrottled() {
        return this.throttled;
    }

    public void onInferenceFailure(String where, RuntimeException e) {
        this.inferenceFailures++;
        this.lastError = where + ": " + e;
        if (this.inferenceFailures % 100 == 1) {
            logDirect("Barelentless ML: " + this.lastError);
        }
    }

    /**
     * A multi-line status report, for the {@code ml} command.
     */
    @Override
    public List<String> status() {
        List<String> lines = new ArrayList<>();
        lines.add("enabled: " + Baritone.settings().mlEnabled.value + ", trainer " + (this.running.get() ? "running" : "stopped"));
        AimModel model = this.liveAimModel;
        lines.add("aim model: " + (model == null ? "not built" :
                model.parameterCount() + " parameters, window " + model.getWindow()));
        lines.add("training: " + this.trainingSteps + " steps, loss "
                + (Float.isNaN(this.lastLoss) ? "n/a" : String.format("%.4f", this.lastLoss))
                + ", lr " + (this.aimOptimizer == null ? "n/a" : String.format("%.2e", this.aimOptimizer.getLearningRate())));
        lines.add("aim samples: " + this.aimSamples.size() + "/" + this.aimSamples.capacity()
                + " (" + this.demonstrationsRecorded + " demonstrations recorded)");
        lines.add("movement samples: " + this.movementSamples.size() + "/" + this.movementSamples.capacity());
        lines.add("memory: " + this.memory.size() + " situations, " + this.memory.totalVisits() + " visits, "
                + this.memory.getMerges() + " merges, " + this.memory.getEvictions() + " evictions");
        lines.add(String.format("inference: %.2f ms/tick%s", getInferenceMillis(),
                this.throttled ? " (THROTTLED - over mlMaxInferenceMs)" : ""));
        lines.add("movement policy: " + (this.liveMovementPolicy == null ? "not built"
                : this.liveMovementPolicy.parameterCount() + " parameters, " + this.policySteps + " steps acted, "
                + this.finishedTrajectories.size() + " trajectories queued"));
        if (!this.lastPolicyReportText.isEmpty()) {
            lines.add("last policy update: " + this.lastPolicyReportText);
        }
        if (this.cautionShaper != null) {
            lines.add("caution: " + this.cautionShaper.describe());
        }
        if (this.aimShaper != null && this.aimShaper.getLastPrediction() != null) {
            lines.add("last prediction: " + this.aimShaper.getLastPrediction());
        }
        lines.add("saved: " + (this.lastSaveMillis == 0 ? "never"
                : ((System.currentTimeMillis() - this.lastSaveMillis) / 1000) + "s ago")
                + ", failures: " + this.inferenceFailures);
        if (!this.lastError.isEmpty()) {
            lines.add("last error: " + this.lastError);
        }
        return lines;
    }

    @Override
    public long getTrainingSteps() {
        return this.trainingSteps;
    }

    public ReplayBuffer<AimSample> getAimSamples() {
        return this.aimSamples;
    }

    public ReplayBuffer<MovementExperience> getMovementSamples() {
        return this.movementSamples;
    }

    public RunningStatistics getFeatureStatistics() {
        return this.featureStatistics;
    }

    /**
     * Throws away everything learned and starts over. Files on disk are overwritten on the next save.
     */
    @Override
    public synchronized void reset() {
        boolean wasRunning = this.running.get();
        stop();
        this.memory.clear();
        this.aimSamples.clear();
        this.movementSamples.clear();
        this.featureStatistics.reset();
        this.finishedTrajectories.clear();
        this.movementTrajectory = null;
        this.trainingSteps = 0;
        this.policySteps = 0;
        this.lastPolicyReportText = "";
        this.demonstrationsRecorded = 0;
        this.lastLoss = Float.NaN;
        this.lastError = "";
        createModels();
        // overwrite the checkpoint now, so a crash before the next save does not resurrect what was just discarded
        save();
        if (wasRunning) {
            start();
        }
    }
}
