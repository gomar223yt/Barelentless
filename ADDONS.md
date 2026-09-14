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

From there, four subsystems:

| Call | What it gives you |
|---|---|
| `baritone.getControlAPI()` | the movement and aim pipeline |
| `baritone.getScriptAPI()` | movement and aim written as formulas |
| `baritone.getLearningAPI()` | what the bot has learned from what it has done |
| `baritone.getPathingBehavior()`, `getCustomGoalProcess()`, … | the pathing API, unchanged from upstream |

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

## Stability

`baritone.api` is the contract: it keeps its names, and the whole package survives into every build. Everything
outside it is internal — it keeps its names too, which makes stack traces and debugging sane, but it is not a
promise.
