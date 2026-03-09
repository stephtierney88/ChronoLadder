# ChronoLadder v2 Training Spec

This file turns the rough ChronoLadder idea into a concrete 3-rung training target.

## Why plain recurrence is a bad fit

Traditional forward recurrence keeps rewriting one shared hidden state:

`h_t = f(x_t, h_{t-1})`

That is useful, but it creates three pressures that conflict with semantic half-lives:

- One state has to carry fast scratch work and slow invariants at the same time.
- Every new token perturbs the same state subspace, so local noise can leak into slow state.
- Long-lived information survives only by repeated multiplication through the recurrent dynamics, which encourages entanglement, aliasing, and saturation.

ChronoLadder instead treats persistence as an explicit state-routing problem:

- `r0` is fast workspace.
- `r1-r3` are persistence holders.
- slow rungs get a strong copy bias and only move when surprise plus cadence say they should.

## 3-rung target

Use four bands total:

- `r0`: instant workspace, no long-term persistence objective
- `r1`: local trail, seconds-scale
- `r2`: episode, minute-scale
- `r3`: schema/session, tens of minutes to hours

Start here before adding `r4`.

## State variables

For each persistent rung `k`:

- `z_k(t)`: latent state
- `e_k(t)`: accumulated evidence / bubble state
- `a_k(t)`: age since last write
- `o_k(t)`: whether the rung is currently in an open write window

## Update equations

### r0 workspace

`r0(t) = phi_0(H_t)`

`H_t` is the current segment hidden state from the AR core. `phi_0` is a pooler or local workspace encoder.

`r0` is allowed to change every step and should not carry the same inertia penalty as the persistent ladder.

### Context collection

Linear ladder:

- `ctx_1 = proj_1(r0)`
- `ctx_2 = proj_2(r0, z_1)`
- `ctx_3 = proj_3(r0, z_2)`

Full hierarchy:

- `ctx_k = sparse_read_k(r0, z_1, ..., z_{k-1})`

Use sparse reads or edge penalties in the full hierarchy. Do not use dense concatenation without pressure.

### Proposal and prediction

For rung `k`:

- `p_k(t) = proposal_k([z_k(t-1), ctx_k(t)])`
- `q_k(t) = predict_k([z_k(t-1), ctx_k(t)])`

`p_k` is the candidate new state.
`q_k` is the model's expectation of what should already persist.

### Surprise

Use proposal/prediction mismatch instead of raw token novelty:

`s_k(t) = alpha * ||sg(p_k(t)) - q_k(t)||_2^2 + beta * (1 - cos(sg(p_k(t)), q_k(t)))`

`sg` is stop-gradient.

This makes surprise horizon-aware: the rung is surprised only when its own stable summary no longer predicts itself.

### Bubble-up evidence

`e_k(t) = lambda_k * e_k(t-1) + (1 - lambda_k) * s_k(t) + bubble_{k-1}(t)`

`bubble_0(t) = 0`

`bubble_k(t) = gamma_k * relu(e_k(t) - theta_k)`

Bubble-up lets persistent lower-rung disruption accumulate into higher-rung evidence instead of waiting for a hard periodic boundary.

### Cadence prior

Cadence is a prior, not a hard clock:

`c_k(t) = sigmoid((a_k(t) - tau_k) / temp_k)`

This says "the older this state is relative to its cadence, the more plausible an update becomes."

### Write probability

`w_k(t) = sigmoid(g_k([p_k(t), z_k(t-1), ctx_k(t), s_k(t), e_k(t), c_k(t), a_k(t)]))`

Recommended interpretation:

- `s_k`: immediate mismatch
- `e_k`: persistent accumulated mismatch
- `c_k`: cadence prior
- `a_k`: hysteresis support

### Hysteresis

Use two thresholds:

- open if `w_k > T_open`
- stay open while `w_k > T_close`

with `T_open > T_close`

This avoids single-step flicker.

### Latent update

Soft training update:

`z_k(t) = (1 - m_k(t)) * z_k(t-1) + m_k(t) * p_k(t)`

where `m_k(t)` is the soft write mask.

Hard inference update:

- if open, write candidate
- if closed, copy old state

Age update:

- if write: `a_k(t) = 0`
- else: `a_k(t) = a_k(t-1) + 1`

## Losses

Total loss:

`L = L_task + sum_k L_pred(k) + sum_k L_inv(k) + sum_k L_inertia(k) + sum_k L_write(k) + L_cov + L_resid`

### 1. Horizon prediction loss

Each rung predicts a future target matched to its horizon:

- `r1`: next local chunk / next action band
- `r2`: next subgoal / next episode summary
- `r3`: next schema or strategy band

`L_pred(k) = SmoothL1(readout_k(z_k(t)), target_k(t + H_k))`

This is the main sufficiency pressure.

### 2. Invariance loss

Construct positives that preserve the same horizon-relevant future but differ in irrelevant detail.

Examples:

- same cave-navigation micro-episode with different local phrasing for `r2`
- same traversal schema across cave and dungeon for `r3`

Recommended form:

`L_inv(k) = InfoNCE(proj_k(z_k(anchor)), proj_k(z_k(pos)), negs_k)`

Positives and negatives must be horizon-conditioned.

### 3. Inertia / movement loss

Penalize moving when there is no reason:

`L_inertia(k) = E[(1 - sg(m_k(t))) * ||z_k(t) - z_k(t-1)||_2^2]`

Slow rungs get stronger weight.

### 4. Write sparsity loss

Encourage sparse writes around a target rate:

`L_write(k) = (mean(m_k) - rho_k)^2`

`rho_k` should decrease with rung depth.

### 5. Covariance / decorrelation loss

Use mild redundancy reduction, not hard orthogonality everywhere:

- VICReg covariance penalty
- Barlow-style off-diagonal penalty

Apply within rung projections and between adjacent rung projections.

### 6. Residual utility loss

Rung `k` should explain something not already handled by lower rungs.

`L_resid(k) = || readout_k(z_k) - stopgrad(readout_{k-1}(z_{k-1})) - residual_target_k ||`

Practical version:

- train readout from `r1`
- train readout from `r1 + r2`
- train readout from `r1 + r2 + r3`
- reward incremental gain from adding the next rung

If a higher rung adds nothing, it is dead or redundant.

## Training curriculum

### Phase 1: band pretraining

Train rung encoders and proposal/predictor heads on offline trajectories.

- use horizon-conditioned positives and negatives
- no end-to-end AR loss yet
- stronger invariance and covariance weights

### Phase 2: gate training

Train surprise, evidence, cadence, and write policy.

- target low write rate on slow rungs
- use interruption/recovery data
- use write utility measured over horizon `H_k`

### Phase 3: AR integration

Feed memory tokens back into the AR model.

- use memory-token dropout
- mask individual rungs during training
- check whether each rung provides measurable gain

### Phase 4: partial end-to-end tuning

Unfreeze selected ladder pieces and the memory read path.

- keep lower LR on slow rungs
- keep high inertia on slow rungs
- keep rung dropout on so the AR core does not overfit one band

## Linear vs full hierarchy

### Linear ladder

Use this first.

Pros:

- cleaner timescale separation
- simpler credit assignment
- easier ablations

Rules:

- `r1` reads `r0`
- `r2` reads `r0 + r1`
- `r3` reads `r0 + r2`

Do not let `r3` directly ingest every low-level detail without pressure.

### Full hierarchy

Use after the linear ladder works.

Pros:

- can capture cross-scale interactions sooner
- may improve recovery from abrupt regime changes

Requirements:

- sparse read attention
- edge dropout
- cross-rung read penalties
- bubble-up evidence

Without those, role collapse is likely.

## Failure modes to test

- No surprise, cadence only: clock-driven stale or thrashy updates
- No cadence, surprise only: unstable write rates and no half-life prior
- No bubble-up: important slow changes get missed between cadence windows
- Too many rungs: adjacent bands collapse
- Cadences too close: redundant features
- Cadences too far: slow rungs stay stale

## Recommended first benchmark

Do not start with generic language modeling.

Start with tasks that require:

- interruption recovery
- persistent goal carry
- map/schema reuse across visually different scenes
- local scene variation with stable higher-level affordances

Pokemon-like navigation is actually a good fit because it naturally separates:

- local tile/motion state
- episode objective
- reusable traversal schema

## Papers that informed this spec

- HM-RNN: https://arxiv.org/abs/1609.01704
- Clockwork RNN: https://proceedings.mlr.press/v32/koutnik14.html
- Compressive Transformer: https://arxiv.org/abs/1911.05507
- Titans: https://arxiv.org/abs/2501.00663
- VICReg: https://arxiv.org/abs/2105.04906
- Barlow Twins: https://arxiv.org/abs/2103.03230
- Slot Attention: https://arxiv.org/abs/2006.15055
- Incremental SFA: https://arxiv.org/abs/1112.2113
