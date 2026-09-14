# Control pipeline and learning

Two subsystems, added on top of Baritone 1.21.11. They are independent: the control pipeline is useful with no
learning at all, and learning is expressed entirely through the control pipeline rather than by patching movement
code.

Everything described here defaults to off or to behaviour identical to unmodified Baritone.

---

## 1. The control pipeline

### What it replaces

Baritone decided movement in forty different movement classes, each setting boolean keys directly, and decided aim by
computing an angle and assigning it. There was no single place where "how the bot moves" lived, so changing it meant
editing every movement, and a third-party addon could not change it at all.

Now every tick's keys, movement vector and aim pass through one pipeline, and anything can be inserted into it.

```
movement code / process / addon
        |
        v
  MovementCommand  ──►  input shapers (ascending priority)  ──►  applied to the game
        |
  RotationTarget   ──►  aim shapers (ascending priority)  ──►  tolerance clamp  ──►  aim processor (GCD)  ──►  applied
```

### Analog movement

`MovementCommand` carries both the discrete keys that existing code thinks in and an optional continuous vector:

```java
command.setAnalog(0.4f, -0.2f);   // 40% forward, 20% right, in the player's own frame
command.scaleAnalog(0.5f);        // same heading, half speed
command.setSneakScale(1.0f);      // sneak without the vanilla 30% slowdown
```

Vanilla input is eight booleans, so Baritone could only ever express full speed in one of eight directions. A ledge
approach that wants forty percent forward had to be a stutter of the forward key. With `analogMovement` on (default),
the vector reaches the game verbatim while key state stays honest for anything reading it.

Turn `analogMovement` off on a server whose movement checks expect the input vector to only ever be one of the
vanilla nine.

### Shaping movement

```java
IControlAPI control = baritone.getControlAPI();

IControlAPI.Registration handle = control.registerInputShaper("carefulNearLava", 700, (ctx, command) -> {
    if (ctx.movementTicks() > 40) {          // this movement is badly overdue
        command.scaleAnalog(0.5f);
        command.setSprintAllowed(false);
    }
});

handle.close();   // removed; the behaviour is exactly gone
```

Shapers run in ascending priority, each seeing what the previous stage decided. Built-in shapers occupy 0–1000. A
shaper that throws is removed and reported rather than being allowed to take the bot down mid-movement.

### Shaping aim

A rotation request carries *why* it exists, which decides how far it may be shaped:

| Purpose | Precise | Meaning |
|---|---|---|
| `BLOCK_INTERACT` | yes | the raytrace has to hit, or breaking/placing fails |
| `ENTITY_INTERACT` | yes | same, but the target moves |
| `FLIGHT` | yes | elytra, where pitch is the throttle |
| `MOVEMENT` | no | steering; only needs to roughly point the right way |
| `COSMETIC` | no | looking around for its own sake |

After every shaper the result is clamped back inside the target's tolerance, so **no shaper can break a precise
interaction**, however aggressively it is configured. This is what makes human-looking aim safe to enable.

```java
control.registerRotationShaper("lazyPitch", 400, (ctx, target, current) ->
        new Rotation(current.getYaw(), current.getPitch() * 0.8f));
```

### Built in: `humanAim`

```
set humanAim true
set humanAimMaxYawSpeed 22
set humanAimResponsiveness 0.35
set humanAimOvershoot 0.06
```

Aim gains a velocity that accelerates into a turn, decelerates out of it, and overshoots slightly on large
corrections before settling — instead of teleporting to the target angle in one tick. Applies only to non-precise
targets.

### Inspecting it

```
control             what was applied on the last tick
control list        every registered shaper, in run order
control trace       what each stage changed (enables tracing on first use)
control remove <n>  remove a shaper by name
```

---

## 2. Learning

### What is learned, and from what

| Model | Learns from | Affects |
|---|---|---|
| Aim model | your own mouse movement, plus the bot's | how the view moves, via an aim shaper |
| Episodic memory | every executed movement's real outcome | pathfinding cost estimates, and how carefully a movement is walked |
| Movement policy | reinforcement: progress made, time spent, damage taken | approach speed, strafe, sprint and jump timing within a movement |

Baritone estimated the cost of every movement and then never checked whether the estimate was right. The memory
closes that loop: the situation as it was, the estimate that was made, and the ticks it really took, plus damage
taken.

### Why a memory and not just a network

A network generalizes, which is what makes it useful and also what makes it wrong exactly where it matters: the one
specific ledge in your base that needs a late jump looks, to a network, like a thousand ordinary ledges.

`EpisodicMemory` keeps that ledge as its own record, with its own measured mean, its own variance and its own
confidence, indexed by LSH so lookup stays fast at hundreds of thousands of records — measured at ~14 µs per query at
200 000 records, with full recall of the near-identical situations that matter. Meeting the same situation again
*updates* the record rather than appending a duplicate — so after a hundred jumps off the same kind of ledge, the
record knows both the average cost and how reliable that average is. The bot can tell "this works" from "this worked
once".

The network handles situations never seen before; the memory handles the ones that have been.

**How it stays fast.** Keys live in one contiguous float array rather than scattered per-record arrays, buckets are
chains through primitive int arrays rather than lists of boxed integers, candidate de-duplication uses a query stamp
instead of a per-query array the size of the memory, and the hash bit count is derived from capacity so buckets hold
about one record each. Together those took a query at 200 000 records from ~160 µs to ~14 µs. The parameters were
picked from measurements, not taste: more bits is faster and loses recall on marginal matches, and the twelve tables
are what buys that recall back.

### The aim model

A causal transformer over a window of ticks (default 8), predicting a **distribution** over the next view delta, not
a number:

- the same 30° error means "flick" at the start of a turn and "you have overshot, settle" at the end of one, and the
  difference is only visible in the preceding ticks — hence a window, and hence attention rather than a decay
  constant;
- it emits a mean *and* a spread, so the controller can hand back to the deterministic path when the model itself
  reports that it does not know this situation, instead of confidently steering into a wall.

Its influence is `mlAimStrength × its own confidence`, and it is clamped to the target's tolerance afterwards.
The output layer is initialized near zero, so a freshly created model aims exactly like unmodified Baritone.

**Demonstrations.** While you are controlling the view yourself, each tick becomes a training sample, labelled with
where you actually ended up looking a few ticks later — not where you were at the time, which would teach nothing.
That is what makes the bot's aim resemble the person it belongs to.

### Where training runs

In-process, in Java, on one background daemon thread at minimum priority. No native library, no Python, no IPC. The
game thread only appends to buffers and reads a `volatile` model reference; the trainer works on its own copy and
publishes a snapshot when it is done, so there is no point at which half-updated weights can be read mid-tick.

Replay is prioritized: a tick where the view barely moved is the overwhelming majority of ticks and teaches almost
nothing, so it is sampled far less often than a large correction, with importance weights correcting the bias.

### Cost adjustment without slowing down A*

A* evaluates hundreds of thousands of candidates per calculation, so a memory lookup per candidate is out of the
question. But situations repeat enormously within one calculation — "traverse north from stone onto stone with air
above" comes up thousands of times and is the same question every time. The situation descriptor is hashed and the
answer cached for the rest of the calculation, so the lookup happens once per *distinct* situation.

The multiplier is clamped to [0.6, 2.5]: the memory can bias the search substantially, but can never convince the
pathfinder that a route through lava is free.

### Learned caution

Where `mlLearnedCosts` changes *which* route is chosen, `mlCaution` changes *how* the chosen route is walked. If a
kind of movement in a kind of place has a history of failing or of being unpredictable, the bot approaches it slower,
without sprinting, and sneaks when a fall looks likely — the same thing a player does at a jump they have missed
before.

There is no rule about jumps or ledges or ice anywhere in it. The behaviour comes entirely from recorded outcomes, so
a situation nobody anticipated — a modded block that is slippery in an unusual way, a contraption that intermittently
blocks a doorway — produces caution for exactly the same reason a vanilla one does. This is only expressible at all
because analog movement exists: "slower" is not something eight booleans can say.

### The movement policy

The one part that learns by trial rather than by imitation. Every tick of an executing movement, the policy proposes
a bounded adjustment to the command pathing already produced, and is rewarded by what follows: progress towards the
destination, minus a small per-tick cost so dithering is not free, minus damage. Finishing the movement pays; failing
it costs.

What it deliberately cannot do:

- it never chooses a direction — it scales the movement vector pathing chose and nudges the strafe;
- it can decline a jump the path planned, but never invent one — a jump the path did not plan for is how a bot ends
  up in a hole, while declining one merely wastes a tick;
- it does not run at all below six hearts, in lava, or on a movement that has to place a block.

Trained with PPO, whose clipped objective bounds how far one update can move the policy away from the one that
generated the data. That matters more here than in an offline setting: the data is collected by the same agent, while
someone is watching, and a single over-large update turns a bot that walks into a bot that vibrates.

Data collected under an older snapshot is still valid — each step stores the log probability of the policy that
produced it, which is exactly what the importance ratio corrects for.

`mlMovementExplore` samples actions rather than taking the mean. Exploration is what makes improvement possible at
all, and also what makes movement visibly less consistent; turn it off to run a trained policy without further
learning.

### Staying out of the frame time

At the default size (window 8, dimension 48, depth 2, ~47 000 parameters) one aim inference measures ~0.35 ms, against
a 50 ms tick. The knobs cost about what you would expect: depth 1 halves it, dimension 32 halves it, window 16 with
dimension 64 costs ~0.9 ms.

Inference runs inside the game loop, so its cost is the player's frame time. The measured cost per tick is tracked as
an exponential average, and above `mlMaxInferenceMs` (default 3 ms) the models step aside instead of eating the tick,
with one tick in eight still let through so the measurement recovers on its own if the machine frees up. `ml status`
reports the current figure and whether throttling is active.

### Settings

```
set mlEnabled true          record experience and run the trainer (nothing else changes on its own)
set mlAim true              let the aim model shape aim
set mlAimStrength 0.6       upper bound on its influence
set mlLearnedCosts true     let remembered outcomes adjust path costs
set mlCostInfluence 0.5     how strongly
set mlCaution true          move carefully where experience says care is warranted
set mlCautionStrength 0.7   how strongly
set mlLearnFromPlayer true  learn from your own mouse movement (default on)
set mlAimWindow 8           how many past ticks the model reads
set mlAimDimension 48       model width
set mlAimDepth 2            transformer blocks
set mlMovement true         let the policy adjust how movements are walked
set mlMovementStrength 0.5  how far it may adjust
set mlMovementExplore true  sample actions (needed to keep learning) or take the mean
set mlMovementBatch 8       finished movements per reinforcement update
set mlBatchSize 32          samples per training step
set mlTrainingDelayMs 25    pause between steps
set mlMaxInferenceMs 3.0    per-tick inference budget before the models throttle themselves
```

`mlEnabled` on its own only records and trains. Behaviour changes only when `mlAim` or `mlLearnedCosts` is also on.

### Commands

```
ml                  what the models are and what they have seen
ml start / stop     enable learning / stop and save
ml save             write to disk now
ml selftest         verify every gradient and layer on this machine
ml memory [n]       the situations remembered most often, with visits, mean, deviation, success rate
ml consolidate      merge duplicate memories, drop ones that never mattered
ml reset confirm    discard everything learned
```

### Files

`.minecraft/baritone/ml/`

| File | Contents |
|---|---|
| `aim.brlm` | aim model weights, named by parameter path |
| `memory.brlm` | remembered situations |
| `movement.brlm` | movement policy weights |
| `features.bin` | feature normalization statistics |

Checkpoints store parameters by qualified name, so they keep loading after layers are reordered in code; anything
that no longer matches is reported rather than silently corrupting the model. Writes are atomic. A checkpoint
recorded against a different feature layout is refused rather than loaded into weights that would misinterpret it.

---

## 3. The learning engine

`baritone.api.ml` — usable on its own, no Minecraft types anywhere in it.

| Package | Contents |
|---|---|
| `baritone.api.ml` | `Tensor` (2D float matrix with reverse-mode autodiff), `Losses`, `MlDiagnostics` |
| `baritone.api.ml.nn` | `Module`, `Linear`, `LayerNorm`, `Dropout`, `Embedding`, `GruCell`, `MultiHeadAttention`, `TransformerBlock`, `SequenceEncoder`, `Mlp`, `Sequential` |
| `baritone.api.ml.optim` | `Adam`/AdamW, `Sgd`, `LearningRateSchedule` |
| `baritone.api.ml.data` | `ReplayBuffer` (prioritized), `RunningStatistics` (Welford) |
| `baritone.api.ml.memory` | `EpisodicMemory`, `MemoryRecord` |
| `baritone.api.ml.rl` | `Policy`, `Trajectory` (GAE), `PpoTrainer` |
| `baritone.api.ml.io` | `ModelIO` checkpoints |

`Tensor.operation` is the public escape hatch for hand-written differentiable kernels — layer norm, attention
gathers and most losses are written that way, and participate in the tape exactly like the built-in operations.

Optimizers clip the global gradient norm and **refuse to apply a non-finite update at all**, leaving the model at its
last good state rather than poisoning weights that took hours to train.

### Self-checks

A learning system that silently computes the wrong gradient does not crash — it just gets slowly, plausibly worse.
So every differentiable operation is checked against central differences, and the checks ship with the mod:

```
ml selftest
```

They also run as part of the ordinary build (`gradlew test`), alongside unit tests for the autodiff tape and the
episodic memory that a numerical gradient check cannot see - shared subexpressions accumulating both contributions,
constants never allocating gradients, softmax staying finite on large logits, memory merging rather than duplicating,
and a saved memory meaning the same thing when loaded back.

Standalone:

```
java -cp <jar> baritone.api.ml.MlDiagnostics
```

16 checks: gradients for elementwise ops, activations, matmul, softmax, layer norm, attention, GRU and every loss;
end-to-end learning of XOR and of a look-back task that is impossible without working attention; a PPO run that has
to raise average reward on a task whose answer the policy is never told; gradient clipping
and non-finite rejection; running statistics; prioritized replay; episodic recall, merging and context isolation;
checkpoint round-trip.

The gradient check skips differences below a float noise floor derived from the loss magnitude and epsilon —
otherwise a *correct* backward pass fails on rounding inside its own finite difference.
