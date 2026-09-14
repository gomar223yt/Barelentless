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

package baritone.control.profile;

import baritone.Baritone;
import baritone.api.control.profile.AimKnob;
import baritone.api.control.profile.GaitKnob;
import baritone.api.control.profile.IProfileAPI;
import baritone.api.control.profile.MotionProfile;
import baritone.api.control.RotationTarget;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * Owns the active motion profile, the ones saved on disk, and the shaper that applies them.
 *
 * @author Barelentless
 */
public final class ProfileManager implements IProfileAPI {

    private static final String EXTENSION = ".profile";

    private final Baritone baritone;
    private final Path directory;

    private volatile MotionProfile activeProfile;

    /**
     * The profile the {@code humanAim} settings describe.
     * <p>
     * Those settings predate profiles and people have them in their configs, so they keep working: when no profile
     * is active and humanAim is on, this one is used. Having one implementation behind both is the point - two
     * shapers doing the same job at different priorities would fight, and the loser would be whichever the user was
     * actually trying to configure.
     */
    private final MotionProfile settingsProfile = new MotionProfile("humanAim");
    private ProfileShaper shaper;
    private final List<String> problems = new ArrayList<>();

    public ProfileManager(Baritone baritone) {
        this.baritone = baritone;
        this.directory = baritone.getDirectory().resolve("profiles");
        // starts disabled: installing the mod must not change how the bot walks until somebody asks it to
        this.activeProfile = new MotionProfile("default").setEnabled(false);
    }

    /**
     * Registers the shaper. Called once, after the control pipeline exists.
     */
    public void install() {
        if (this.shaper != null) {
            return;
        }
        // through active(), not the field, so the humanAim fallback reaches the shaper too
        this.shaper = new ProfileShaper(this::active);
        // below the learning shapers and below where an addon would normally register: a profile sets the character
        // of the movement, and anything more specific still gets the last word
        this.baritone.getControlAPI().registerInputShaper("profile", 300, this.shaper);
        this.baritone.getControlAPI().registerRotationShaper("profile", 300, this.shaper);
        writeExamplesIfMissing();
    }

    @Override
    public MotionProfile active() {
        MotionProfile chosen = this.activeProfile;
        if (chosen.isEnabled()) {
            return chosen;
        }
        if (Baritone.settings().humanAim.value) {
            syncSettingsProfile();
            return this.settingsProfile;
        }
        return chosen;
    }

    /**
     * Mirrors the humanAim settings into the profile that stands in for them. Done per read rather than on change
     * because settings have no change notification, and this is a handful of array writes.
     */
    private void syncSettingsProfile() {
        this.settingsProfile
                .set(AimKnob.MAX_YAW_SPEED, Baritone.settings().humanAimMaxYawSpeed.value)
                .set(AimKnob.MAX_PITCH_SPEED, Baritone.settings().humanAimMaxPitchSpeed.value)
                .set(AimKnob.RESPONSIVENESS, Baritone.settings().humanAimResponsiveness.value)
                .set(AimKnob.OVERSHOOT, Baritone.settings().humanAimOvershoot.value)
                .setEnabled(true);
    }

    @Override
    public void setActive(MotionProfile profile) {
        this.activeProfile = profile == null ? new MotionProfile("default").setEnabled(false) : profile;
    }

    @Override
    public boolean activate(String name) {
        Path file = fileFor(name);
        if (!Files.exists(file)) {
            return false;
        }
        try {
            this.problems.clear();
            String text = new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
            this.activeProfile = MotionProfile.load(text, name, this.problems);
            return true;
        } catch (IOException e) {
            this.problems.add("could not read " + file + ": " + e);
            return false;
        }
    }

    @Override
    public void save() {
        Path file = fileFor(this.activeProfile.getName());
        try {
            Files.createDirectories(this.directory);
            Files.write(file, this.activeProfile.save().getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            this.problems.add("could not write " + file + ": " + e);
        }
    }

    @Override
    public List<String> available() {
        List<String> names = new ArrayList<>();
        if (!Files.isDirectory(this.directory)) {
            return names;
        }
        try (Stream<Path> stream = Files.list(this.directory)) {
            stream.filter(path -> path.getFileName().toString().endsWith(EXTENSION))
                    .sorted()
                    .forEach(path -> {
                        String file = path.getFileName().toString();
                        names.add(file.substring(0, file.length() - EXTENSION.length()));
                    });
        } catch (IOException ignored) {
            // a listing that fails is reported as empty rather than thrown; the caller is usually a chat command
        }
        return names;
    }

    @Override
    public Path getDirectory() {
        return this.directory;
    }

    @Override
    public void reset() {
        String name = this.activeProfile.getName();
        boolean enabled = this.activeProfile.isEnabled();
        this.activeProfile = new MotionProfile(name).setEnabled(enabled);
    }

    public List<String> getProblems() {
        return new ArrayList<>(this.problems);
    }

    private Path fileFor(String name) {
        String safe = name.replaceAll("[^A-Za-z0-9._-]", "_");
        return this.directory.resolve(safe + EXTENSION);
    }

    /**
     * Writes a few starting points on first use. They are examples of what the dials do, not recommendations - the
     * point is that somebody opening the folder sees what a profile looks like and what is worth changing.
     */
    private void writeExamplesIfMissing() {
        try {
            Files.createDirectories(this.directory);
            for (Map.Entry<String, MotionProfile> example : examples().entrySet()) {
                Path file = fileFor(example.getKey());
                if (!Files.exists(file)) {
                    Files.write(file, example.getValue().save().getBytes(StandardCharsets.UTF_8));
                }
            }
        } catch (IOException e) {
            this.problems.add("could not write example profiles: " + e);
        }
    }

    private static Map<String, MotionProfile> examples() {
        Map<String, MotionProfile> examples = new LinkedHashMap<>();

        MotionProfile human = new MotionProfile("human");
        human.set(AimKnob.MAX_YAW_SPEED, 18)
                .set(AimKnob.RESPONSIVENESS, 0.3)
                .set(AimKnob.ACCELERATION, 0.4)
                .set(AimKnob.OVERSHOOT, 0.08)
                .set(AimKnob.JITTER, 0.35)
                .set(AimKnob.DRIFT, 0.8)
                .set(AimKnob.DRIFT_SPEED, 0.3)
                .set(AimKnob.REACTION_TICKS, 2)
                .set(GaitKnob.ACCELERATION_TICKS, 4)
                .set(GaitKnob.BRAKING_TICKS, 3)
                .set(GaitKnob.STRAFE_SMOOTHING, 0.3)
                .set(GaitKnob.JITTER, 0.02)
                .set(GaitKnob.CORNER_LEAN, 0.4)
                .set(GaitKnob.HESITATION, 0.004)
                // aim that has to land exactly is shaped too, but without the noise: the tolerance clamp would
                // absorb it anyway, and spending tolerance budget on tremor is how a block break gets missed
                .set(RotationTarget.Purpose.BLOCK_INTERACT, AimKnob.SHAPE_PRECISE, 1)
                .set(RotationTarget.Purpose.BLOCK_INTERACT, AimKnob.JITTER, 0)
                .set(RotationTarget.Purpose.BLOCK_INTERACT, AimKnob.DRIFT, 0)
                .set(RotationTarget.Purpose.BLOCK_INTERACT, AimKnob.MAX_YAW_SPEED, 40);
        examples.put("human", human);

        MotionProfile careful = new MotionProfile("careful");
        careful.set(GaitKnob.SPEED, 0.8)
                .set(GaitKnob.ACCELERATION_TICKS, 6)
                .set(GaitKnob.EDGE_CAUTION, 0.7)
                .set(GaitKnob.EDGE_LOOKAHEAD, 2)
                .set(GaitKnob.EDGE_SNEAK, 1)
                .set(GaitKnob.SPRINT_POLICY, 1)
                .set(AimKnob.MAX_YAW_SPEED, 25)
                .set(AimKnob.RESPONSIVENESS, 0.45);
        examples.put("careful", careful);

        MotionProfile machine = new MotionProfile("machine");
        machine.set(AimKnob.MAX_YAW_SPEED, 3600)
                .set(AimKnob.MAX_PITCH_SPEED, 3600)
                .set(AimKnob.RESPONSIVENESS, 1)
                .set(AimKnob.ACCELERATION, 1)
                .set(AimKnob.OVERSHOOT, 0)
                .set(GaitKnob.SPRINT_POLICY, 3);
        examples.put("machine", machine);

        return examples;
    }
}
