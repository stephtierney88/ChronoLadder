# ChronoLadder Architecture Map

This note maps the ChronoLadder variants currently in the project and clarifies how they relate to the ideas of:

- linear ladder
- full hierarchy ladder
- latent compression
- surprise-gated persistence
- slot-based anchor reuse

## First: What The Names Mean

The names `v1`, `v2`, and `v2-b` are local project labels for the three current directions in this repo.

They are not formal published versions.

Current mapping:

- `v1` = `ChronoLadder.py`
- `v2` = `chronoladder_v2.py`
- `v2-b` = `chronoladder_v2b_slots.py`

## High-Level View

### v1

Question:

> can multi-rate latent side-memory help at all?

Style:

- rung latents
- mostly compression-centric
- higher rungs read all lower rungs

Closest label:

- rough full-hierarchy latent ladder

### v2

Question:

> can explicit semantic horizons persist by copying by default and updating only when surprise plus cadence says they should?

Style:

- rung latents
- explicit workspace plus persistent bands
- surprise-gated writes
- bubble-up evidence
- cadence as age prior

Closest label:

- semantic-horizon latent ladder
- can be linear or full hierarchy

### v2-b

Question:

> should persistence be organized as reusable anchor slots rather than one vector per rung?

Style:

- slot banks instead of single rung vectors
- anchor matching
- `copy`, `refresh`, `spawn`, `promote`
- linear upward promotion
- broad memory-token readout

Closest label:

- slot-based anchor ladder

## What A Linear Ladder Means

A linear ladder means write flow is mostly local and upward:

- `r1` is built mainly from `r0`
- `r2` is built mainly from `r1`
- `r3` is built mainly from `r2`

The key property is:

- higher rungs do not directly absorb everything below them by default

Benefits:

- cleaner specialization
- less cross-scale contamination
- easier credit assignment

Risks:

- slower propagation of large changes unless bubble-up or promotion is strong

## What A Full Hierarchy Ladder Means

A full hierarchy means a higher rung can directly read all lower rungs during update:

- `r2` can inspect `r0` and `r1`
- `r3` can inspect `r0`, `r1`, and `r2`

The key property is:

- higher rungs get a full lower-scale view at write time

Benefits:

- abrupt cross-scale changes can be integrated faster
- higher rungs can react directly to lower-level patterns

Risks:

- role collapse
- slow-state contamination by fast details
- more entanglement unless sparsity and gating are strong

## Variant Map

## v1: Legacy AE Ladder

File:

- `ChronoLadder.py`

State form:

- one latent vector per rung

Write path:

- current hidden state plus lower-rung latents are compressed into a rung latent

Read path:

- all rung latents are concatenated and projected back into model conditioning

Theory:

- build a multi-rate latent summary stack
- slower rungs keep broader or longer-lived context

Why it is a useful baseline:

- it tests whether persistent latent side-state helps at all

What it usually wants to learn:

- compressed summaries of context at different cadences

Main limitation:

- compression pressure is not the same thing as invariance pressure
- it can easily learn blurry summary vectors instead of clean horizon-specific state

Topology label:

- closest to full hierarchy

Why:

- higher rungs consume all lower rung outputs during update

## v2: Semantic-Horizon Latent Ladder

File:

- `chronoladder_v2.py`

State form:

- one persistent latent vector per rung
- plus explicit workspace `r0`

Write path:

- each rung forms a proposal for new state
- each rung predicts what should already persist
- proposal/prediction mismatch becomes surprise
- cadence prior depends on age
- lower-rung mismatch bubbles upward
- a hysteretic write gate decides whether to copy or update

Read path:

- each rung emits a memory token
- these tokens are meant to be fed back into the main model

Theory:

- the main job of memory is not to compress history
- it is to preserve the right latent state at the right half-life

This is the cleanest expression of ChronoLadder's main thesis.

### v2 Linear Mode

Config:

- `ChronoLadderV2Config(topology="linear")`

Write structure:

- `r1` reads workspace
- `r2` reads workspace plus `r1`
- `r3` reads workspace plus `r2`

Behavior:

- cleaner band separation
- more disciplined hierarchy
- usually better first research baseline

### v2 Hierarchical Mode

Config:

- `ChronoLadderV2Config(topology="hierarchical")`

Write structure:

- each rung can read all lower rungs

Behavior:

- potentially faster reaction to regime shifts
- higher risk of rung overlap and state pollution

Best use:

- only after linear mode works and you want to test richer cross-scale interaction

## v2-b: Slot-Based Anchor Ladder

File:

- `chronoladder_v2b_slots.py`

State form:

- each rung is a bank of slots
- each slot has:
  - key
  - value
  - confidence
  - age
  - alive state

Write path:

- build a candidate key/value from the current context
- match candidate against existing slots
- decide whether to:
  - copy nothing
  - refresh matched slot
  - spawn a new slot
  - promote summary upward

Read path:

- each rung emits a summary token plus slot tokens
- these form a richer memory bank than one vector per rung

Theory:

- persistence is better modeled as anchor reuse than as repeated recompression

The guiding idea is:

> if the current situation matches an existing stable anchor, refresh it  
> if it does not, spawn a new one  
> if it stays important, promote it upward

### Why v2-b Exists

The main concern with `v1` and `v2` is that one vector per rung can still smear together multiple latent factors:

- two different schemas
- two tracked objects
- two competing goals
- two distinct episode fragments

`v2-b` tries to preserve identity structure.

### What v2-b Is Best At

- reusable schemas across different local scenes
- multiple concurrent latent objects or patterns
- interruption recovery where an anchor should be resumed rather than re-derived
- continual-memory-like behavior without immediate weight changes

### What v2-b Is Not

- not pure full hierarchy
- not pure dense recurrence
- not pure retrieval memory

It is best described as:

- linear promotion ladder for writes
- slot bank for persistence
- broad memory-token bank for reads

## Side-By-Side Summary

| Variant | State unit | Write style | Read style | Closest topology | Main strength | Main weakness |
| --- | --- | --- | --- | --- | --- | --- |
| `v1` | one latent per rung | compress current context upward | concat all latents | rough full hierarchy | simplest baseline | blurry summary soup |
| `v2-linear` | one latent per rung | surprise-gated local upward writes | memory tokens | linear ladder | clean horizon separation | limited multi-entity capacity |
| `v2-hierarchical` | one latent per rung | surprise-gated all-lower writes | memory tokens | full hierarchy | richer cross-scale access | role collapse risk |
| `v2-b` | slot bank per rung | anchor match + refresh/spawn/promote | summary + slot tokens | linear write ladder | identity reuse and schema persistence | hardest to train |

## How To Think About v2-b Specifically

`v2-b` is the most different of the three.

The best mental model is:

- `v1`: compress
- `v2`: persist by selective update
- `v2-b`: persist by reusing anchors

That means `v2-b` is trying to answer a different question:

- not only "what should persist?"
- but also "which persistent thing is this another instance of?"

This matters for cases like:

- cave and dungeon differ locally
- but traversal schema is similar

In a slot ladder:

- lower rungs can track local cave-versus-dungeon details
- higher rungs can refresh the same schema slot if the structure is functionally similar

That is a stronger and more reusable notion of memory than just compressing both into nearby vectors.

## Practical Guidance

Use:

- `v1` if you want the roughest baseline
- `v2-linear` if you want the cleanest test of ChronoLadder's core thesis
- `v2-hierarchical` if you want to test whether richer cross-scale writes help more than they hurt
- `v2-b` if you believe identity, reuse, and multi-object persistence are central

## My Current View

If the goal is scientific clarity:

- start with `v2-linear`

If the goal is strongest long-run architecture bet:

- push `v2-b`

If the goal is just to confirm that persistent latent side-state helps at all:

- keep `v1` around as the simplest baseline
