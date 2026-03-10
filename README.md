# ChronoLadder

ChronoLadder is a research sketch for semantic continuity across inferences.

The core claim is simple:

- not all information should update at the same rate
- not all meaning has the same half-life
- persistence should be structured by temporal horizon, not flattened into one recurrent soup or re-inferred from chat history every turn

Instead of treating memory as one blob, ChronoLadder splits persistence into semantic horizons that update at different cadences and ideally only move when something actually changes at that scale.

## What Is In This Folder

This folder currently has three ladder directions:

- `ChronoLadder.py`
  - legacy rough sketch
  - AE-centric ladder over a language-model backbone
- `chronoladder_v2.py`
  - 3-rung surprise-gated latent ladder
  - explicit workspace, cadence prior, bubble-up evidence, and hysteresis
- `chronoladder_v2b_slots.py`
  - slot-based sibling branch of `v2`
  - anchor reuse with `copy`, `refresh`, `spawn`, and `promote` behavior

Supporting docs:

- `TRAINING_SPEC_V2.md`
  - equations, losses, update rules, and training curriculum for the `v2` ladder
- `LADDER_COMPARISON.md`
  - likely behaviors, strengths, weaknesses, and benchmark advice across the three variants

## Variant Summary

### 1. Legacy AE Ladder

Useful as a rough baseline.

Pros:

- simple
- cheap
- easy to modify

Cons:

- tends to learn compressed summaries rather than clean invariants
- weak pressure for role separation across timescales
- likely to smear multiple latent factors together

### 2. v2 Surprise-Gated Latent Ladder

This is the clearest implementation of the semantic-horizon thesis.

Main ideas:

- `r0` is fast workspace
- `r1-r3` are persistent bands
- cadence acts as an age prior
- surprise drives writes
- bubble-up propagates persistent lower-rung mismatch upward

This is the best starting point if the question is:

> does explicit multi-rate latent persistence actually help?

### 3. v2-b Slot Ladder

This is the slot-based sibling branch of `v2`, not a direct successor.

Main ideas:

- use slot banks instead of single rung vectors
- reuse anchors instead of overwriting state
- promote persistent novelty upward
- preserve multiple distinct latent entities or schemas at once

This is a better fit than plain compression if the real target is:

- schema reuse
- multi-object memory
- interruption recovery
- continual state without total entanglement

## Design Thesis

ChronoLadder is trying to solve a specific failure mode in current systems:

- larger context windows improve retrieval
- they do not automatically improve what should persist
- current models often recover intent by re-reading tokens, not by carrying explicit structured state between inferences

The central question is:

> which parts of meaning should remain stable across time, and which should be allowed to move?

## Current Status

This repository is a research scaffold, not a production implementation.

The files here are intended to:

- make the ideas concrete
- support ablations
- provide a path from rough sketch to testable architecture variants

## Suggested Reading Order

1. `README.md`
2. `LADDER_COMPARISON.md`
3. `TRAINING_SPEC_V2.md`
4. `chronoladder_v2.py`
5. `chronoladder_v2b_slots.py`

## Best Near-Term Experiments

- interruption recovery
- persistent goal carry
- local scene variation with stable higher-level affordances
- schema reuse across different surfaces
- tasks where local details churn but episode or strategy state should remain stable

Pokemon-like navigation is a good fit because it naturally separates:

- local motion and obstacle state
- current subgoal / episode
- reusable traversal schemas

## Notes

- The original sketch is preserved because it is still useful as a baseline.
- `v2` is the best direct test of the semantic-horizon idea.
- `v2-b` is the slot-based branch I would personally push further if the core thesis holds.
