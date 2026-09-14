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

package baritone.control.script;

import baritone.Baritone;
import baritone.api.control.ControlContext;
import baritone.api.control.MovementCommand;
import baritone.api.control.RotationTarget;
import baritone.api.control.script.ScriptContext;
import baritone.api.control.script.ScriptFunctions;
import baritone.api.pathing.movement.IMovement;
import baritone.api.utils.BetterBlockPos;
import baritone.api.utils.Rotation;
import baritone.api.utils.input.Input;
import baritone.ml.MovementKinds;
import baritone.pathing.movement.MovementHelper;
import baritone.utils.BlockStateInterface;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.Vec3;

/**
 * The vocabulary a control script sees, and the code that fills it in each tick.
 * <p>
 * Two decisions shape the whole thing. First, everything is in the player's own frame: {@code dx} is how far forward
 * the destination is, not how far east, and {@code solid(1, 0, 0)} asks about the block ahead, not the block north.
 * Writing a movement in world coordinates means writing it four times, and getting the diagonals wrong.
 * <p>
 * Second, filling the frame is a fixed sequence of array writes into pre-resolved slots - no maps, no strings, no
 * allocation. This runs once per script per tick, and the whole point of the scripting layer is that it is cheap
 * enough that nobody has to think about whether they can afford one.
 *
 * @author Barelentless
 */
public final class ScriptBindings {

    private final ScriptContext context;

    // resolved slots, so filling is a straight line of array stores
    private final int tick;
    private final int movementTicks;
    private final int movementCost;
    private final int overrun;
    private final int speed;
    private final int forwardSpeed;
    private final int lateralSpeed;
    private final int verticalSpeed;
    private final int onGround;
    private final int inWater;
    private final int inLava;
    private final int climbing;
    private final int sprinting;
    private final int sneaking;
    private final int flying;
    private final int health;
    private final int hunger;
    private final int fallDistance;
    private final int lookYaw;
    private final int lookPitch;
    private final int targetYaw;
    private final int targetPitch;
    private final int posY;
    private final int fracX;
    private final int fracY;
    private final int fracZ;
    private final int dx;
    private final int dy;
    private final int dz;
    private final int dist;
    private final int distXZ;
    private final int angle;
    private final int yawError;
    private final int pitchError;
    private final int tolerance;
    private final int precise;
    private final int purpose;
    private final int kind;
    private final int kindUp;
    private final int kindReach;
    private final int pathing;
    private final int inForward;
    private final int inStrafe;
    private final int inJump;
    private final int inSprint;
    private final int inSneak;

    private final int outForward;
    private final int outStrafe;
    private final int outJump;
    private final int outSprint;
    private final int outSneak;
    private final int outSneakScale;
    private final int outYaw;
    private final int outPitch;
    private final int outYawDelta;
    private final int outPitchDelta;

    /**
     * The world as of the current evaluation. Set before a script runs and cleared after, so a world query function
     * can reach it without every function signature carrying the context.
     */
    private BlockStateInterface blocks;
    private BetterBlockPos feet;
    private int quarterTurn;

    /**
     * Where the last {@link #blockAt} call landed. The movement helpers need the position as well as the state -
     * whether a block can be walked through can depend on its neighbours - so the lookup records it.
     */
    private int queryX;
    private int queryY;
    private int queryZ;

    public ScriptBindings() {
        ScriptContext context = new ScriptContext();

        this.tick = declare(context, "tick", "ticks since this Baritone started");
        this.movementTicks = declare(context, "movementTicks", "ticks spent on the current movement");
        this.movementCost = declare(context, "movementCost", "what the current movement was estimated to cost, in ticks");
        this.overrun = declare(context, "overrun", "movementTicks / movementCost; above 1 means it is running late");
        this.speed = declare(context, "speed", "horizontal speed, blocks per tick");
        this.forwardSpeed = declare(context, "forwardSpeed", "speed along the direction you face, can be negative");
        this.lateralSpeed = declare(context, "lateralSpeed", "speed to your left, can be negative");
        this.verticalSpeed = declare(context, "verticalSpeed", "vertical speed, blocks per tick");
        this.onGround = declare(context, "onGround", "1 when standing on something");
        this.inWater = declare(context, "inWater", "1 when in water");
        this.inLava = declare(context, "inLava", "1 when in lava");
        this.climbing = declare(context, "climbing", "1 when on a ladder or vine");
        this.sprinting = declare(context, "sprinting", "1 when sprinting");
        this.sneaking = declare(context, "sneaking", "1 when sneaking");
        this.flying = declare(context, "flying", "1 when elytra flying");
        this.health = declare(context, "health", "health, 0 to 20");
        this.hunger = declare(context, "hunger", "food level, 0 to 20");
        this.fallDistance = declare(context, "fallDistance", "blocks fallen so far");
        this.lookYaw = declare(context, "lookYaw", "where the player is looking now, yaw in degrees");
        this.lookPitch = declare(context, "lookPitch", "where the player is looking now, pitch in degrees");
        this.targetYaw = declare(context, "targetYaw", "yaw the pipeline wants before this script, degrees");
        this.targetPitch = declare(context, "targetPitch", "pitch the pipeline wants before this script, degrees");
        this.posY = declare(context, "posY", "current height");
        this.fracX = declare(context, "fracX", "position within the block, 0 to 1, east");
        this.fracY = declare(context, "fracY", "position within the block, 0 to 1, up");
        this.fracZ = declare(context, "fracZ", "position within the block, 0 to 1, south");

        this.dx = declare(context, "dx", "how far ahead the destination is, in blocks");
        this.dy = declare(context, "dy", "how far above the destination is");
        this.dz = declare(context, "dz", "how far to the left the destination is");
        this.dist = declare(context, "dist", "distance to the destination");
        this.distXZ = declare(context, "distXZ", "horizontal distance to the destination");
        this.angle = declare(context, "angle", "degrees you would have to turn to face the destination");
        this.yawError = declare(context, "yawError", "degrees between where you look and where the target is");
        this.pitchError = declare(context, "pitchError", "same, vertically");
        this.tolerance = declare(context, "tolerance", "how far the aim may deviate and still work");
        this.precise = declare(context, "precise", "1 when the aim has to land exactly, such as breaking a block");
        this.purpose = declare(context, "purpose", "0 block, 1 entity, 2 movement, 3 flight, 4 cosmetic");

        this.kind = declare(context, "kind", "identifier of the movement's shape");
        this.kindUp = declare(context, "kindUp", "how many blocks the movement climbs, negative for down");
        this.kindReach = declare(context, "kindReach", "how far it reaches horizontally, 0 to 3");
        this.pathing = declare(context, "pathing", "1 when following a path");

        this.inForward = declare(context, "inForward", "forward impulse decided before this script, -1 to 1");
        this.inStrafe = declare(context, "inStrafe", "strafe impulse decided before this script, left positive");
        this.inJump = declare(context, "inJump", "1 if jump was already requested");
        this.inSprint = declare(context, "inSprint", "1 if sprint was already requested");
        this.inSneak = declare(context, "inSneak", "1 if sneak was already requested");

        this.outForward = output(context, "forward", "forward impulse to apply, -1 to 1");
        this.outStrafe = output(context, "strafe", "strafe impulse to apply, left positive");
        this.outJump = output(context, "jump", "hold jump when above 0.5");
        this.outSprint = output(context, "sprint", "allow sprinting when above 0.5");
        this.outSneak = output(context, "sneak", "hold sneak when above 0.5");
        this.outSneakScale = output(context, "sneakScale", "how much sneaking slows movement, default 0.3");
        this.outYaw = output(context, "yaw", "absolute yaw to look at, degrees; defaults to targetYaw");
        this.outPitch = output(context, "pitch", "absolute pitch to look at, degrees; defaults to targetPitch");
        this.outYawDelta = output(context, "yawDelta", "degrees to turn horizontally this tick");
        this.outPitchDelta = output(context, "pitchDelta", "degrees to turn vertically this tick");

        installWorldFunctions(context);
        ScriptFunctions.install(context);
        this.context = context;
    }

    private static int declare(ScriptContext context, String name, String description) {
        context.variable(name, description);
        return context.slotOf(name);
    }

    private static int output(ScriptContext context, String name, String description) {
        context.output(name, description);
        return context.slotOf(name);
    }

    /**
     * The world queries. Coordinates are forward / up / left in blocks, relative to the block the player is standing
     * in, snapped to the nearest quarter turn - the terrain lattice is axis aligned, so a continuous rotation would
     * alias, while a quarter turn is exact.
     */
    private void installWorldFunctions(ScriptContext context) {
        context.function("solid", 3, 3, "solid(forward, up, left)",
                "1 when that block blocks movement", (frame, a) -> {
                    BlockState state = blockAt(a);
                    if (state == null) {
                        return 1; // unknown terrain counts as blocked: walking into an unloaded chunk is how bots fall
                    }
                    return MovementHelper.canWalkThrough(this.blocks, this.queryX, this.queryY, this.queryZ, state)
                            ? 0 : 1;
                });
        context.function("standable", 3, 3, "standable(forward, up, left)",
                "1 when you could stand on that block", (frame, a) -> {
                    BlockState state = blockAt(a);
                    return state != null
                            && MovementHelper.canWalkOn(this.blocks, this.queryX, this.queryY, this.queryZ, state)
                            ? 1 : 0;
                });
        context.function("liquid", 3, 3, "liquid(forward, up, left)",
                "1 when that block contains a fluid", (frame, a) -> {
                    BlockState state = blockAt(a);
                    return state != null && !state.getFluidState().isEmpty() ? 1 : 0;
                });
        context.function("air", 3, 3, "air(forward, up, left)",
                "1 when that block is air", (frame, a) -> {
                    BlockState state = blockAt(a);
                    return state != null && state.isAir() ? 1 : 0;
                });
        context.function("friction", 3, 3, "friction(forward, up, left)",
                "slipperiness of that block; ordinary ground is 0.6, ice is higher", (frame, a) -> {
                    BlockState state = blockAt(a);
                    return state == null ? 0.6 : state.getBlock().getFriction();
                });
        context.function("known", 3, 3, "known(forward, up, left)",
                "1 when that block is in a loaded chunk", (frame, a) -> blockAt(a) == null ? 0 : 1);
    }

    private BlockState blockAt(double[] arguments) {
        if (this.blocks == null || this.feet == null) {
            return null;
        }
        int forward = (int) Math.round(arguments[0]);
        int up = (int) Math.round(arguments[1]);
        int left = (int) Math.round(arguments[2]);
        int east;
        int south;
        switch (this.quarterTurn) {
            case 0:  // facing south (+z)
                east = -left;
                south = forward;
                break;
            case 1:  // facing west (-x)
                east = -forward;
                south = -left;
                break;
            case 2:  // facing north (-z)
                east = left;
                south = -forward;
                break;
            default: // facing east (+x)
                east = forward;
                south = left;
                break;
        }
        this.queryX = this.feet.x + east;
        this.queryY = this.feet.y + up;
        this.queryZ = this.feet.z + south;
        if (!this.blocks.worldContainsLoadedChunk(this.queryX, this.queryZ)) {
            return null;
        }
        return this.blocks.get0(this.queryX, this.queryY, this.queryZ);
    }

    public ScriptContext getContext() {
        return this.context;
    }

    /**
     * Fills the movement half of the frame. Called before a script that shapes input runs.
     */
    public void fill(double[] frame, ControlContext control, MovementCommand command) {
        if (control.player().player() == null) {
            return;
        }
        this.blocks = ((Baritone) control.baritone()).bsi;
        this.feet = control.player().playerFeet();
        float facing = control.player().player().getYRot();
        this.quarterTurn = Math.floorMod(Math.round(facing / 90f), 4);

        Vec3 velocity = control.player().playerMotion();
        double radians = Math.toRadians(facing);
        double sin = Math.sin(radians);
        double cos = Math.cos(radians);

        frame[this.tick] = control.tick();
        frame[this.movementTicks] = control.movementTicks();
        frame[this.speed] = control.speed();
        frame[this.forwardSpeed] = -velocity.x * sin + velocity.z * cos;
        frame[this.lateralSpeed] = -velocity.x * cos - velocity.z * sin;
        frame[this.verticalSpeed] = velocity.y;
        frame[this.onGround] = control.onGround() ? 1 : 0;
        frame[this.inWater] = control.player().player().isInWater() ? 1 : 0;
        frame[this.inLava] = control.player().player().isInLava() ? 1 : 0;
        frame[this.climbing] = control.player().player().onClimbable() ? 1 : 0;
        frame[this.sprinting] = control.player().player().isSprinting() ? 1 : 0;
        frame[this.sneaking] = control.player().player().isShiftKeyDown() ? 1 : 0;
        frame[this.flying] = control.player().player().isFallFlying() ? 1 : 0;
        frame[this.health] = control.player().player().getHealth();
        frame[this.hunger] = control.player().player().getFoodData().getFoodLevel();
        frame[this.fallDistance] = control.player().player().fallDistance;
        frame[this.lookYaw] = facing;
        frame[this.lookPitch] = control.player().player().getXRot();

        Vec3 position = control.player().player().position();
        frame[this.posY] = position.y;
        frame[this.fracX] = position.x - Math.floor(position.x);
        frame[this.fracY] = position.y - Math.floor(position.y);
        frame[this.fracZ] = position.z - Math.floor(position.z);
        frame[this.pathing] = control.isPathing() ? 1 : 0;

        IMovement movement = control.movement();
        if (movement != null) {
            BetterBlockPos destination = movement.getDest();
            double toEast = destination.x + 0.5 - position.x;
            double toUp = destination.y - position.y;
            double toSouth = destination.z + 0.5 - position.z;
            double ahead = -toEast * sin + toSouth * cos;
            double left = -toEast * cos - toSouth * sin;
            frame[this.dx] = ahead;
            frame[this.dy] = toUp;
            frame[this.dz] = left;
            double horizontal = Math.sqrt(toEast * toEast + toSouth * toSouth);
            frame[this.dist] = Math.sqrt(horizontal * horizontal + toUp * toUp);
            frame[this.distXZ] = horizontal;
            frame[this.angle] = Math.toDegrees(Math.atan2(left, ahead));
            double cost = movement.getCost();
            frame[this.movementCost] = cost;
            frame[this.overrun] = cost > 0 ? control.movementTicks() / cost : 0;
            int identifier = MovementKinds.of(movement);
            frame[this.kind] = identifier;
            frame[this.kindUp] = destination.y - movement.getSrc().y;
            frame[this.kindReach] = Math.max(Math.abs(destination.x - movement.getSrc().x),
                    Math.abs(destination.z - movement.getSrc().z));
        } else {
            frame[this.dx] = 0;
            frame[this.dy] = 0;
            frame[this.dz] = 0;
            frame[this.dist] = 0;
            frame[this.distXZ] = 0;
            frame[this.angle] = 0;
            frame[this.movementCost] = 0;
            frame[this.overrun] = 0;
            frame[this.kind] = -1;
            frame[this.kindUp] = 0;
            frame[this.kindReach] = 0;
        }

        frame[this.inForward] = command.getForwardImpulse();
        frame[this.inStrafe] = command.getStrafeImpulse();
        frame[this.inJump] = command.isPressed(Input.JUMP) ? 1 : 0;
        frame[this.inSprint] = command.isSprintAllowed() && command.isPressed(Input.SPRINT) ? 1 : 0;
        frame[this.inSneak] = command.isPressed(Input.SNEAK) ? 1 : 0;

        // outputs start as what the previous stage decided, so a script that only sets some of them leaves the rest
        // exactly as they were rather than zeroing them
        frame[this.outForward] = frame[this.inForward];
        frame[this.outStrafe] = frame[this.inStrafe];
        frame[this.outJump] = frame[this.inJump];
        frame[this.outSprint] = frame[this.inSprint];
        frame[this.outSneak] = frame[this.inSneak];
        frame[this.outSneakScale] = command.getSneakScale();
    }

    /**
     * Fills the aim half of the frame. Called before a script that shapes rotation runs.
     */
    public void fillAim(double[] frame, ControlContext control, RotationTarget target, Rotation current) {
        if (control.player().player() == null) {
            return;
        }
        Rotation from = control.player().playerRotations();
        frame[this.lookYaw] = from.getYaw();
        frame[this.lookPitch] = from.getPitch();
        frame[this.targetYaw] = current.getYaw();
        frame[this.targetPitch] = current.getPitch();
        frame[this.yawError] = Rotation.normalizeYaw(current.getYaw() - from.getYaw());
        frame[this.pitchError] = current.getPitch() - from.getPitch();
        frame[this.tolerance] = target.getTolerance();
        frame[this.precise] = target.getPurpose().isPrecise() ? 1 : 0;
        switch (target.getPurpose()) {
            case BLOCK_INTERACT:
                frame[this.purpose] = 0;
                break;
            case ENTITY_INTERACT:
                frame[this.purpose] = 1;
                break;
            case MOVEMENT:
                frame[this.purpose] = 2;
                break;
            case FLIGHT:
                frame[this.purpose] = 3;
                break;
            default:
                frame[this.purpose] = 4;
                break;
        }
        frame[this.outYaw] = current.getYaw();
        frame[this.outPitch] = current.getPitch();
        frame[this.outYawDelta] = frame[this.yawError];
        frame[this.outPitchDelta] = frame[this.pitchError];
    }

    /**
     * Releases the world reference once evaluation is done, so a script cannot hold a block accessor alive past the
     * tick it belongs to.
     */
    public void release() {
        this.blocks = null;
        this.feet = null;
    }

    // slot accessors for the shaper, which reads the outputs back out of the frame

    public int slotForward() {
        return this.outForward;
    }

    public int slotStrafe() {
        return this.outStrafe;
    }

    public int slotJump() {
        return this.outJump;
    }

    public int slotSprint() {
        return this.outSprint;
    }

    public int slotSneak() {
        return this.outSneak;
    }

    public int slotSneakScale() {
        return this.outSneakScale;
    }

    public int slotYaw() {
        return this.outYaw;
    }

    public int slotPitch() {
        return this.outPitch;
    }

    public int slotYawDelta() {
        return this.outYawDelta;
    }

    public int slotPitchDelta() {
        return this.outPitchDelta;
    }
}
