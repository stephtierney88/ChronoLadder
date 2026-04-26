# ChronoLadder Hybrid Variants To Try

This note sketches the next branch after `v2` and `v2-b`.

The motivating idea is:

> latent memory carries pressure; ledger memory carries warrants.

Latent memory is good at preserving implicit task state:

- subgoal pressure
- uncertainty
- stance / belief state
- local expectations
- where the task "feels" like it is

Explicit ledger memory is good at preserving auditable state:

- provenance
- confidence
- expiry
- contradictions
- user corrections
- stable facts and decisions

The strongest practical system is probably hybrid.

## Variant Name

This branch is called `v2-c`.

Why not `v3`:

- it is not a cleaner successor to `v2`
- it is another sibling attack on the same problem
- it keeps the ChronoLadder principle but adds explicit epistemic bookkeeping

Current skeleton file:

- `chronoladder_v2c_hybrid.py`

## Architecture Sketch

```text
raw transcript / frames / tool logs
             |
             v
evidence encoder / event parser
             |
   +---------+----------+----------------+
   |         |          |                |
   v         v          v                v
r0           r1 slots   r2 slots         ledger
workspace    events     task state       provenance /
referents    occlusion  goals            confidence /
local detail failures   constraints      expiry /
                                      contradiction
             |          |                |
             +----+-----+--------+-------+
                  |              |
                  v              v
             r3 schema      audit / correction
             slots          interface
             invariants
             routines
                  |
                  v
        memory-token / cross-attn readout
                  |
                  v
            core model / policy
```

## Bets To Test

1. Slots beat single vectors when identity matters.
2. Surprise + cadence beats cadence alone.
3. Predictive/control losses beat reconstruction losses.
4. Linear upward write flow is safer than dense full hierarchy.
5. Slow memory needs explicit anti-drift pressure.
6. Readout should start as memory tokens or cross-attention.
7. Text/symbolic memory remains useful for audit and slow facts.
8. Latent memory carries what text summaries lose: belief state, uncertainty, subgoal pressure, and implicit context.
9. The best version is hybrid, not purist.
10. Cross-rung identity links matter.

## Variant A: Latent Slots + Ledger

This is the main `v2-c` idea.

State:

- `r0`: workspace
- `r1`: event slots
- `r2`: task-state slots
- `r3`: schema slots
- ledger: explicit auditable entries

Expected win:

- latent slots preserve behavioral continuity
- ledger preserves correction, confidence, provenance, and expiry

Expected failure:

- if ledger and latent slots disagree, the model needs a resolver
- if ledger is too dominant, it becomes a brittle text memory system
- if latent slots are too dominant, auditability is lost

## Variant B: Cross-Rung Identity Threads

Question:

> does a local slot know which slower schema slot it is evidence for?

Mechanism:

- fast slots can attach evidence to slower slots
- promotion creates an identity thread across rungs
- contradiction can flow down from ledger to prevent bad refreshes

Expected win:

- better schema reuse
- less duplicate-slot growth
- cleaner interruption recovery

Expected failure:

- wrong attachment poisons a slow schema
- too much attachment pressure causes premature abstraction

## Variant C: Ledger-Gated Writes

Question:

> should explicit facts and corrections control which latent writes are allowed?

Mechanism:

- ledger entries emit trust / contradiction / expiry features
- latent write gates receive those features
- contradictions can suppress refresh and encourage spawn or retire

Expected win:

- user corrections become durable
- contradictory evidence does not silently smear into old slots

Expected failure:

- over-trusting ledger text can freeze mistaken state

## Variant D: kNN Over Slots And Ledger Entries

Question:

> once ladder state exists, does retrieval work better when keyed by rung or slot state?

Mechanism:

- index slow slot summaries
- index ledger embeddings
- retrieve prior similar schema states or explicit decisions

Expected win:

- better recurring-task recovery
- less first-day-on-the-job syndrome for related tasks

Expected failure:

- stale retrieved slots may bias the current task
- retrieval can bypass write discipline if not gated

## Variant E: Hybrid Readout

Question:

> should the core read latent memory and ledger memory differently?

Mechanism:

- latent slots are read through cross-attention / memory tokens
- ledger entries are read through structured metadata tokens
- correction entries get a stronger or separate channel

Expected win:

- latent continuity without losing auditability
- clearer behavior under contradiction or user correction

Expected failure:

- too many memory tokens can become another long-context problem

## Training Pressures

Useful losses:

- horizon prediction for latent slots
- write sparsity by rung
- anti-drift for slow slots
- slot diversity / anti-duplicate pressure
- ledger consistency with explicit facts
- contradiction resolution loss
- expiry calibration loss
- retrieval usefulness loss
- rung / ledger masking ablations

Bad default:

- training everything to reconstruct the transcript

Why:

- it encourages feature transport rather than predictive state

## Key Ablations

- latent slots only
- ledger only
- slots + ledger
- slots + ledger + cross-rung identity links
- slots + ledger + kNN retrieval
- ledger-gated writes on / off
- explicit contradiction channel on / off
- text facts as memory tokens vs separate structured tokens

## What To Measure

Behavior:

- interruption recovery
- goal survival
- schema transfer
- correction persistence
- contradiction handling
- reduced need for full chat history

State:

- active slot count
- duplicate slot similarity
- slot refresh / spawn / retire rate
- ledger expiry calibration
- rung masking failures
- cross-rung identity attachment accuracy

## Current Position

`v2` is the clean test of semantic-horizon decomposition.

`v2-b` is the slot/anchor test for identity and multi-instance persistence.

`v2-c` is the practical hybrid:

- latent slots for implicit continuity
- explicit ledger for warrants and correction
- memory-token / cross-attention readout into the core

