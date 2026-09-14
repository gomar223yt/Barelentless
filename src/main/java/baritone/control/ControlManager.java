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

package baritone.control;

import baritone.Baritone;
import baritone.api.control.ControlContext;
import baritone.api.control.IControlAPI;
import baritone.api.control.IInputShaper;
import baritone.api.control.IRotationShaper;
import baritone.api.control.MovementCommand;
import baritone.api.control.RotationTarget;
import baritone.api.event.events.TickEvent;
import baritone.api.event.events.WorldEvent;
import baritone.api.pathing.movement.IMovement;
import baritone.api.pathing.path.IPathExecutor;
import baritone.api.utils.Helper;
import baritone.api.utils.Rotation;
import baritone.api.utils.input.Input;
import baritone.behavior.Behavior;
import net.minecraft.world.phys.Vec3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * The implementation of the movement and aim pipeline.
 * <p>
 * Runs last among behaviours, so by the time it ticks, pathing has already decided which keys it wants. Those keys
 * are lifted back out of the input handler into a {@link MovementCommand}, every registered shaper gets a turn at it,
 * and the result is written back - which means a shaper can override a decision that movement code made a moment
 * earlier without that code needing to know anything about shapers.
 *
 * @author Barelentless
 */
public final class ControlManager extends Behavior implements IControlAPI, Helper {

    private static final Input[] MOVEMENT_INPUTS = {
            Input.MOVE_FORWARD, Input.MOVE_BACK, Input.MOVE_LEFT, Input.MOVE_RIGHT,
            Input.JUMP, Input.SNEAK, Input.SPRINT
    };

    private final List<RegisteredShaper<IInputShaper>> inputShapers = new CopyOnWriteArrayList<>();
    private final List<RegisteredShaper<IRotationShaper>> rotationShapers = new CopyOnWriteArrayList<>();

    private MovementCommand lastCommand = new MovementCommand();
    private MovementCommand requestedCommand;
    private int requestedCommandPriority = Integer.MIN_VALUE;
    private RotationTarget requestedRotation;
    private RotationTarget lastRotationTarget;
    private Rotation lastAppliedRotation;

    private long tick;
    private IMovement lastMovement;
    private int movementTicks;

    private boolean analogMovementEnabled = true;
    private boolean tracing;
    private List<String> trace = Collections.emptyList();

    public ControlManager(Baritone baritone) {
        super(baritone);
        // Built-in shaping now lives in the motion profile, registered by ProfileManager at priority 300, so that
        // gait and aim are one set of dials rather than two implementations that could disagree.
    }

    // ----------------------------------------------------------------------------------------------- registration

    private static final class RegisteredShaper<T> implements Registration {

        private final String name;
        private final int priority;
        private final T shaper;
        private final List<RegisteredShaper<T>> owner;
        private volatile boolean active = true;

        RegisteredShaper(String name, int priority, T shaper, List<RegisteredShaper<T>> owner) {
            this.name = name;
            this.priority = priority;
            this.shaper = shaper;
            this.owner = owner;
        }

        @Override
        public String name() {
            return this.name;
        }

        @Override
        public int priority() {
            return this.priority;
        }

        @Override
        public boolean isActive() {
            return this.active;
        }

        @Override
        public void close() {
            if (this.active) {
                this.active = false;
                this.owner.remove(this);
            }
        }
    }

    @Override
    public Registration registerInputShaper(String name, int priority, IInputShaper shaper) {
        RegisteredShaper<IInputShaper> registration = new RegisteredShaper<>(name, priority, shaper, this.inputShapers);
        this.inputShapers.add(registration);
        this.inputShapers.sort(Comparator.comparingInt(RegisteredShaper::priority));
        return registration;
    }

    @Override
    public Registration registerRotationShaper(String name, int priority, IRotationShaper shaper) {
        RegisteredShaper<IRotationShaper> registration =
                new RegisteredShaper<>(name, priority, shaper, this.rotationShapers);
        this.rotationShapers.add(registration);
        this.rotationShapers.sort(Comparator.comparingInt(RegisteredShaper::priority));
        return registration;
    }

    @Override
    public List<Registration> inputShapers() {
        return new ArrayList<>(this.inputShapers);
    }

    @Override
    public List<Registration> rotationShapers() {
        return new ArrayList<>(this.rotationShapers);
    }

    @Override
    public void clearShapers() {
        for (Registration registration : inputShapers()) {
            registration.close();
        }
        for (Registration registration : rotationShapers()) {
            registration.close();
        }
    }

    // ------------------------------------------------------------------------------------------------------ input

    @Override
    public void onTick(TickEvent event) {
        if (event.getType() == TickEvent.Type.OUT) {
            return;
        }
        this.tick++;
        updateMovementTracking();

        final boolean driving = this.baritone.getPathingBehavior().isPathing() || this.requestedCommand != null;
        List<String> trace = isTracing() ? new ArrayList<>() : null;
        MovementCommand command = currentCommandBase();
        if (trace != null) {
            trace.add("base: " + command);
        }

        ControlContext context = context();
        for (RegisteredShaper<IInputShaper> registration : this.inputShapers) {
            if (!registration.shaper.isEnabled()) {
                continue;
            }
            try {
                registration.shaper.shape(context, command);
            } catch (Exception e) {
                // one broken addon shaper must not take the bot down mid-movement
                logDirect("input shaper " + registration.name() + " threw " + e);
                registration.close();
                continue;
            }
            if (trace != null) {
                trace.add(registration.name() + " (" + registration.priority() + "): " + command);
            }
        }
        if (!isAnalogMovementEnabled()) {
            command.clearAnalog();
        }
        // While pathing, movement code rebuilds key state from scratch every tick, so writing our result back is
        // safe. While idle it does not, and writing back would make our own output the next tick's input - a shaper
        // that pressed sneak once would leave it pressed forever. So when nobody is driving, nothing is written.
        if (driving) {
            applyCommand(command);
        } else {
            this.baritone.getInputOverrideHandler().setActiveCommand(null);
        }
        this.lastCommand = command;
        this.requestedCommand = null;
        this.requestedCommandPriority = Integer.MIN_VALUE;
        if (trace != null) {
            this.trace = trace;
        }
    }

    /**
     * The command as movement code left it: whatever keys are currently forced down, plus any externally requested
     * command that outranks them.
     */
    private MovementCommand currentCommandBase() {
        MovementCommand command = new MovementCommand().setSource("pathing");
        for (Input input : Input.values()) {
            command.set(input, this.baritone.getInputOverrideHandler().isInputForcedDown(input));
        }
        if (this.requestedCommand != null && this.requestedCommandPriority >= 0) {
            return new MovementCommand(this.requestedCommand);
        }
        return command;
    }

    private void applyCommand(MovementCommand command) {
        for (Input input : MOVEMENT_INPUTS) {
            boolean pressed = command.isPressed(input);
            if (input == Input.SPRINT && !command.isSprintAllowed()) {
                pressed = false;
            }
            this.baritone.getInputOverrideHandler().setInputForceState(input, pressed);
        }
        this.baritone.getInputOverrideHandler().setActiveCommand(command);
    }

    private void updateMovementTracking() {
        IPathExecutor executor = this.baritone.getPathingBehavior().getCurrent();
        IMovement movement = null;
        if (executor != null && executor.getPosition() < executor.getPath().movements().size()) {
            movement = executor.getPath().movements().get(executor.getPosition());
        }
        if (movement != this.lastMovement) {
            this.lastMovement = movement;
            this.movementTicks = 0;
        } else if (movement != null) {
            this.movementTicks++;
        }
    }

    private ControlContext context() {
        double speed = 0;
        boolean onGround = false;
        if (ctx.player() != null) {
            Vec3 velocity = ctx.player().getDeltaMovement();
            speed = Math.sqrt(velocity.x * velocity.x + velocity.z * velocity.z);
            onGround = ctx.player().onGround();
        }
        return new ControlContext(this.baritone, this.tick, this.lastMovement, this.movementTicks,
                this.lastCommand, this.lastAppliedRotation, speed, onGround);
    }

    // --------------------------------------------------------------------------------------------------- rotation

    /**
     * Runs a rotation request through the shaper chain. Called by the look behaviour for every target it receives.
     *
     * @return The rotation to actually aim at
     */
    public Rotation shapeRotation(RotationTarget target) {
        RotationTarget effective = target;
        if (this.requestedRotation != null && this.requestedRotation.getPriority() > target.getPriority()) {
            effective = this.requestedRotation;
        }
        this.requestedRotation = null;
        this.lastRotationTarget = effective;

        ControlContext context = context();
        Rotation current = effective.getRotation();
        List<String> trace = isTracing() ? new ArrayList<>(this.trace) : null;
        for (RegisteredShaper<IRotationShaper> registration : this.rotationShapers) {
            IRotationShaper shaper = registration.shaper;
            if (!shaper.isEnabled() || !shaper.appliesTo(effective.getPurpose())) {
                continue;
            }
            Rotation shaped;
            try {
                shaped = shaper.shape(context, effective, current);
            } catch (Exception e) {
                logDirect("rotation shaper " + registration.name() + " threw " + e);
                registration.close();
                continue;
            }
            if (shaped == null || Float.isNaN(shaped.getYaw()) || Float.isNaN(shaped.getPitch())
                    || Float.isInfinite(shaped.getYaw()) || Float.isInfinite(shaped.getPitch())) {
                logDirect("rotation shaper " + registration.name() + " produced an invalid rotation, removing it");
                registration.close();
                continue;
            }
            // clamp after every stage rather than only at the end, so a shaper that reads `current` is reading
            // something that was already legal for this target
            current = effective.enforceTolerance(shaped.normalizeAndClamp());
            if (trace != null) {
                trace.add("aim " + registration.name() + ": " + current);
            }
        }
        if (trace != null) {
            this.trace = trace;
        }
        this.lastAppliedRotation = current;
        return current;
    }

    @Override
    public void onWorldEvent(WorldEvent event) {
        this.lastMovement = null;
        this.movementTicks = 0;
        this.lastAppliedRotation = null;
        this.lastRotationTarget = null;
        this.requestedCommand = null;
        this.requestedRotation = null;
    }

    // ------------------------------------------------------------------------------------------------------ state

    @Override
    public MovementCommand lastCommand() {
        return this.lastCommand;
    }

    @Override
    public RotationTarget lastRotationTarget() {
        return this.lastRotationTarget;
    }

    @Override
    public Rotation lastAppliedRotation() {
        return this.lastAppliedRotation;
    }

    @Override
    public void requestCommand(MovementCommand command, int priority) {
        if (priority >= this.requestedCommandPriority) {
            this.requestedCommand = command;
            this.requestedCommandPriority = priority;
        }
    }

    @Override
    public void requestRotation(RotationTarget target) {
        if (this.requestedRotation == null || target.getPriority() >= this.requestedRotation.getPriority()) {
            this.requestedRotation = target;
        }
        this.baritone.getLookBehavior().updateTarget(target.getRotation(), target.getPurpose().isPrecise());
    }

    @Override
    public boolean isAnalogMovementEnabled() {
        return this.analogMovementEnabled && Baritone.settings().analogMovement.value;
    }

    @Override
    public void setAnalogMovementEnabled(boolean enabled) {
        this.analogMovementEnabled = enabled;
    }

    @Override
    public List<String> lastTrace() {
        return this.trace;
    }

    @Override
    public void setTracing(boolean tracing) {
        this.tracing = tracing;
        if (!tracing) {
            this.trace = Collections.emptyList();
        }
    }

    @Override
    public boolean isTracing() {
        return this.tracing || Baritone.settings().controlTracing.value;
    }

    public long getTick() {
        return this.tick;
    }
}
