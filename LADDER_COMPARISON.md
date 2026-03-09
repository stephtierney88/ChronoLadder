# ChronoLadder Variant Comparison

This file compares the three ladder directions currently present in the local workspace.

## Where These Files Are

These files are local to:

`C:\Users\thebeast\OneDrive\Desktop\ChronoLadder`

This workspace is not currently a git checkout, so these additions are local files, not changes on a GitHub branch.

Current local variants:

- `ChronoLadder.py`: rough AE-centric ladder sketch
- `chronoladder_v2.py`: 3-rung surprise-gated latent ladder
- `chronoladder_v3_slots.py`: slot-based ladder with anchor reuse
- `TRAINING_SPEC_V2.md`: v2 equations, losses, and curriculum

## Short Version

If you want:

- simplest rough scaffold: use the original sketch
- cleanest test of semantic horizons: use `v2`
- strongest long-run research bet: use `v3`

## Variant 1: AE-Centric Ladder

File:

- `ChronoLadder.py`

Core idea:

- each rung stores a compressed latent summary
- higher rungs update more slowly
- memory is fused back into the model as conditioning

What it is good at:

- very fast iteration
- easy to understand
- useful as a first proof that persistent latent side-state can help at all

Likely learned behavior:

- rungs become graded compressions of recent hidden state
- slower rungs may drift toward generic context summaries
- state can help short-term persistence without learning clean temporal roles

Main failure modes:

- rung collapse, where adjacent rungs encode the same information
- compression of nuisance detail rather than future-relevant invariants
- poor handling of multiple simultaneous latent objects or schemas

Best use:

- baseline
- ablation target
- quick toy experiments

## Variant 2: Surprise-Gated Latent Ladder

File:

- `chronoladder_v2.py`

Core idea:

- `r0` is workspace
- `r1-r3` are persistent latent bands
- cadence is a prior
- surprise is the write trigger
- bubble-up carries persistent lower-rung mismatch upward

What it is good at:

- testing the actual ChronoLadder thesis
- clean write logic
- explicit inertia
- straightforward ablations for cadence, surprise, and bubble-up

Likely learned behavior:

- `r1` tracks local mode and micro-episode state
- `r2` tracks task phase / subgoal
- `r3` tracks slower schema or session strategy
- higher rungs stay still until enough mismatch accumulates

Main failure modes:

- adjacent rungs learn similar bands if horizons are not well separated
- slow rungs become stale if thresholds are too conservative
- all rungs can become generic “summary vectors” if residual utility is weak

Best use:

- primary research baseline
- benchmark-oriented development
- testing whether semantic half-life separation is real

## Variant 3: Slot Ladder With Anchor Reuse

File:

- `chronoladder_v3_slots.py`

Core idea:

- persistent memory is a bank of slots, not one latent per rung
- slots have identity-like persistence
- writes are action-like: `copy`, `refresh`, `spawn`, `promote`
- upward flow is mostly linear
- outward read is a sparse memory-token bank

What it is good at:

- preserving distinct latent entities or schemas
- reusing old anchors instead of overwriting state
- handling “same schema, different surface form”
- continual-memory-style behavior

Likely learned behavior:

- `r1` stores local micro-event anchors
- `r2` stores episode chunks or mode anchors
- `r3` stores reusable schema anchors
- repeated similar situations refresh existing slots instead of creating new ones
- genuinely novel conditions spawn new slots and may later get promoted upward

Main failure modes:

- dead slots
- duplicate slots
- slot explosion if spawn becomes too cheap
- unstable promotion if surprise is noisy

Best use:

- strongest long-term architecture bet
- agent memory and persistent task state
- environments with reusable structures across different scenes

## Side-By-Side Expectations

### If the task is simple delayed recall

- Variant 1 may already look decent
- Variant 2 should be more stable under distractors
- Variant 3 may be overkill

### If the task requires interruption recovery

- Variant 1 will likely degrade first
- Variant 2 should recover better if the right rung held the task phase
- Variant 3 should recover best if the interruption does not destroy slot-anchor identity

### If the task has repeated schemas with changing local detail

- Variant 1 may memorize averages
- Variant 2 may learn a slower schema latent if training is good
- Variant 3 is most naturally aligned because slot reuse can preserve schema anchors

### If the task has many simultaneous latent objects

- Variant 1 is weakest
- Variant 2 is limited by one-vector-per-rung pressure
- Variant 3 is the natural fit

## Expected Ablation Results

### Remove surprise

- Variant 2 becomes cadence-driven clock memory
- Variant 3 spawns or refreshes on weak heuristics instead of semantics

Expected symptom:

- stale upper state or meaningless periodic churn

### Remove cadence

- Variant 2 writes become too reactive
- Variant 3 may over-refresh frequently matched slots

Expected symptom:

- slow state loses half-life discipline

### Remove bubble-up

- Variant 2 misses slow changes between write opportunities
- Variant 3 under-promotes persistent novelty

Expected symptom:

- important changes stay trapped in lower bands

### Remove redundancy / diversity penalties

- Variant 1 and Variant 2 develop band overlap
- Variant 3 develops duplicate anchors

## What To Benchmark

Best benchmark families:

- interruption and resumption
- persistent goal carry
- local scene variation with stable higher-level affordances
- repeated schemas across different visual or textual surfaces
- multi-object latent state tracking

Pokemon-like tasks are useful because they naturally separate:

- local movement constraints
- current episode objective
- reusable traversal schemas

## Recommended Development Order

1. Use Variant 1 only as a rough sanity check.
2. Build proper evaluations around Variant 2.
3. Move to Variant 3 if evaluations show you need distinct persistent anchors, schema reuse, or multi-object memory.

## What To Watch During Training

For all variants:

- write rates by rung
- latent movement by rung
- ablation drop when masking each rung
- recovery after interruption

For Variant 2 specifically:

- surprise magnitude by rung
- bubble magnitude by rung
- correlation between adjacent rung states

For Variant 3 specifically:

- active slot count per rung
- slot reuse rate
- slot replacement rate
- promotion rate
- duplicate-slot similarity

## My Current Bet

If the question is “what will most quickly show whether ChronoLadder’s thesis has signal,” the answer is Variant 2.

If the question is “what architecture would I rather keep pushing if the thesis is right,” the answer is Variant 3.
