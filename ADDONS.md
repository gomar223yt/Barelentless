# Using this from your own mod

Everything the control and learning subsystems do is reachable through `baritone.api`, and the jar people install
keeps real class names, so an addon compiled against the API can call it at runtime.

## Names are not obfuscated

Upstream Baritone runs its release jars through ProGuard with renaming on, and strips the API from the standalone
build entirely — an addon can compile against `baritone-api` but not call anything at runtime, and a stack trace
from a user reads `at baritone.a.a(SourceFile)`.

This fork turns renaming off (`-dontobfuscate`) and keeps `baritone.api.**` in *every* build, including standalone.
ProGuard still shrinks and optimizes, so the jar stays about the same size; only the renaming is gone.

Verified on the shipped Fabric jar: 254 API classes present, 0 obfuscated class names.

## Depending on it

Use the API jar produced by the build (`fabric/build/libs/baritone-api-fabric-<version>.jar`) as a `compileOnly`
dependency, and let the installed mod provide it at runtime:

```gradle
dependencies {
    compileOnly files("libs/baritone-api-fabric-1.17.0.jar")
}
```

Your `fabric.mod.json` should declare the dependency so the loader orders you after it:

```json
"depends": { "baritone": "*" }
```

## Getting in

```java
import baritone.api.BaritoneAPI;
import baritone.api.IBaritone;

IBaritone baritone = BaritoneAPI.getProvider().getPrimaryBaritone();
```

From there:

| Call | What it gives you |
|---|---|
| `baritone.getControlAPI()` | the movement and aim pipeline |
| `baritone.getProfileAPI()` | how it walks and turns, as dials: speeds, acceleration, overshoot, spread |
| `baritone.getScriptAPI()` | movement and aim written as formulas |
| `baritone.getCostRegistry()` | what the pathfinder is told things cost, i.e. which way it goes |
| `baritone.getLearningAPI()` | what the bot has learned from what it has done |
| `baritone.getPathingControlManager().registerProcess(…)` | your own process, competing with mine/follow/build |
| `baritone.getCommandManager().getRegistry().register(…)` | your own chat command |
| `baritone.getGameEventHandler().registerEventListener(…)` | every tick, packet, chunk and path event |
| `baritone.getCustomGoalProcess().setGoalAndPath(…)` | send it somewhere, with your own `Goal` |

## Tuning gait and aim without a shaper

Most of what people want to change is a number, not a behaviour. A profile is that set of numbers, applied by a
built-in shaper at priority 300:

```java
MotionProfile profile = baritone.getProfileAPI().active();

profile.set(AimKnob.MAX_YAW_SPEED, 18)
       .set(AimKnob.RESPONSIVENESS, 0.3)
       .set(AimKnob.OVERSHOOT, 0.08)
       .set(AimKnob.JITTER, 0.4)          // per-tick spread
       .set(AimKnob.DRIFT, 0.8)           // slow smooth wander
       .set(AimKnob.REACTION_TICKS, 2)
       .set(GaitKnob.ACCELERATION_TICKS, 4)
       .set(GaitKnob.CORNER_LEAN, 0.4)
       .set(GaitKnob.EDGE_CAUTION, 0.6)
       .setEnabled(true);

// spread everywhere except where a block break depends on the angle
profile.set(RotationTarget.Purpose.BLOCK_INTERACT, AimKnob.JITTER, 0);
```

Dials are also addressable by name, which is what a config screen wants:

```java
profile.setByName("gait.speed", 0.8);
profile.setByName("movement.jitter", 3);
MotionProfile.vocabulary().forEach((key, description) -> ...);   // every dial, with descriptions
```

Profiles serialise to and from plain `key = value` text (`profile.save()` / `MotionProfile.load(...)`), writing only
what differs from the default, so you can ship one with your mod or let users swap them.

## Shaping movement

A shaper is a stage of the pipeline. It sees the command as the previous stage left it, and may change any part of
it. Priorities are ascending — a higher number runs later and wins. Built-ins occupy 0–1000.

```java
IControlAPI control = baritone.getControlAPI();

IControlAPI.Registration handle = control.registerInputShaper("myAddon:cautious", 800, (ctx, command) -> {
    if (ctx.movementTicks() > 40) {          // this movement is badly overdue
        command.scaleAnalog(0.5f);            // same heading, half speed
        command.setSprintAllowed(false);
    }
});

// when your mod unloads
handle.close();
```

`MovementCommand` carries both the discrete keys and a continuous vector, so "forty percent forward" is expressible:

```java
command.setAnalog(0.4f, -0.2f);   // 40% forward, 20% right, in the player's own frame
command.setSneakScale(1.0f);      // sneak without the vanilla 30% slowdown
```

## Shaping aim

A rotation request carries *why* it exists. Whatever your shaper returns is clamped back inside the target's
tolerance, so you cannot break a block interaction however aggressive you are.

```java
control.registerRotationShaper("myAddon:lazyPitch", 400, (ctx, target, current) -> {
    if (target.getPurpose().isPrecise()) {
        return current;                       // leave breaking and placing alone
    }
    return new Rotation(current.getYaw(), current.getPitch() * 0.8f);
});
```

## Driving the player directly

To move the player yourself — a combat routine, a minigame bot — request a command. It goes through the same shaping
pipeline as one produced by pathing, so it inherits every registered shaper.

```java
MovementCommand command = new MovementCommand().setSource("myAddon");
command.setAnalog(1.0f, 0f).press(Input.JUMP);
control.requestCommand(command, 100);         // higher priority wins within a tick

control.requestRotation(new RotationTarget(rotation, RotationTarget.Purpose.ENTITY_INTERACT, "myAddon"));
```

## Registering a script from memory

You can ship movement logic as a string instead of Java, and register it without touching the player's script
folder:

```java
IScriptAPI scripts = baritone.getScriptAPI();

IControlAPI.Registration handle = scripts.register("myAddon:edges", String.join("\n",
        "priority 700",
        "when onGround && !flying",
        "let floorAhead = standable(1, -1, 0)",
        "forward = inForward * (floorAhead > 0.5 ? 1 : 0.4)",
        "sprint  = inSprint && floorAhead > 0.5"));
```

It compiles at registration and throws `ScriptException` — with the line, column and a caret — if it does not. The
registered script is an ordinary pipeline stage: it shows up in `control list` and is removed by closing the handle.

`scripts.newContext()` returns the full vocabulary (every variable and function, with descriptions) if you want to
show it in a config screen or validate what a user typed:

```java
scripts.newContext().describeVariables().forEach(System.out::println);
scripts.check(userTypedExpression);           // throws ScriptException if it is wrong
```

## Changing which way it goes

Shapers change how the bot walks. To change *where* it decides to walk, adjust what the pathfinder is told things
cost. Avoid someone's farmland, prefer lit corridors at night, route around the area your own mod is building in:

```java
ICostRegistry costs = baritone.getCostRegistry();

ICostRegistry.Registration handle = costs.register("myAddon:avoidFarmland", (instance, blocks) -> {
    // one adjuster per path calculation, so this cache is thread-confined and needs no synchronization
    Long2DoubleMap cache = new Long2DoubleOpenHashMap();

    return (move, srcX, srcY, srcZ, destX, destY, destZ, cost) -> {
        if (!blocks.isLoaded(destX, destZ)) {
            return cost;
        }
        return blocks.get(destX, destY - 1, destZ).getBlock() == Blocks.FARMLAND ? cost * 5 : cost;
    };
});
```

Every registered adjuster runs in turn, each seeing the cost the previous one left. The result is validated before
use — anything not finite and positive is discarded and the original kept — so a broken adjuster degrades into
having no opinion rather than into paths through lava.

**This is the hottest code in the mod.** A* asks about hundreds of thousands of candidates per calculation; an
adjuster that takes a microsecond adds a fifth of a second to every path. Cache per *situation*, not per candidate —
the same question repeats thousands of times within one calculation, which is why the factory hands you a fresh
instance to cache in. `control costs` shows what your adjuster actually did.

The learning subsystem's own cost adjustment goes through this same public registry, not through a private hook.

## Reading what the bot has learned

```java
ILearningAPI learning = baritone.getLearningAPI();
EpisodicMemory memory = learning.getMemory();

EpisodicMemory.Estimate estimate = memory.estimate(situationKey, movementKind, 8);
if (!estimate.isEmpty() && estimate.successRate < 0.8) {
    // this kind of movement, in this kind of place, has a history of going wrong
}
```

An addon that adds its own kind of movement can contribute to the same memory rather than building a parallel one —
file outcomes under a context id of your own so your situations never answer somebody else's question:

```java
memory.remember(key, MY_CONTEXT_ID, null, ticksTaken / estimatedTicks, succeeded, System.currentTimeMillis());
```

## Using the learning engine on its own

`baritone.api.ml` has no Minecraft types in it anywhere. Tensors with reverse-mode autodiff, transformer blocks,
Adam, prioritized replay, PPO, checkpoints — usable for anything, not just this mod:

```java
Mlp model = new Mlp(inputs, new int[]{64, 64}, outputs, Activation.GELU, Activation.IDENTITY, true, random);
Adam optimizer = new Adam(model.parameters(), 3e-4f);

optimizer.zeroGrad();
Tensor loss = Losses.huber(model.forward(batch), targets, 1f);
loss.backward();
optimizer.step();

ModelIO.save(model, path, Map.of("trainedOn", "12345 samples"));
```

`MlDiagnostics.runAll()` gradient-checks every operation against central differences, which is also what
`ml selftest` runs in game.

## What you cannot do through the API

Being straight about the edges, because finding them yourself at 2am is worse:

- **Add a new movement type to the pathfinder.** `Moves` is an enum in the internals, and A* iterates its values, so
  an addon cannot add "grapple hook" as a thing the search considers. You can make existing movements cheaper or
  more expensive (above), and you can take over execution entirely once a path is being followed (shapers, or
  `requestCommand`), but the search space itself is fixed.
- **Add a setting to `Settings`.** It is a class of fields, read reflectively by the `set` command. Keep your own
  config; you can validate expressions against the script vocabulary with `IScriptAPI#check` if you want the same
  language in it.
- **Change how a specific movement class executes.** `MovementParkour` and friends are internal. In practice a
  shaper at a high priority gets you there, since it sees every tick of that movement and can overwrite the whole
  command - but it is not a subclass hook.
- **Run a script that keeps state across ticks.** The scripting language has no variables that survive a tick and no
  loops, on purpose. Anything stateful is a Java shaper.

Everything else the bot does to move - the keys, the movement vector, the aim, route preference, what it learns,
when it paths and where to - is reachable from `baritone.api`.

## Stability

`baritone.api` is the contract: it keeps its names, and the whole package survives into every build. Everything
outside it is internal — it keeps its names too, which makes stack traces and debugging sane, but it is not a
promise.
