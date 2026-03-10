from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    total = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return total / denom


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("off_diagonal expects a square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def covariance_penalty(z: torch.Tensor) -> torch.Tensor:
    if z.size(0) <= 1:
        return z.new_zeros(())
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (z.size(0) - 1)
    return off_diagonal(cov).pow(2).mean()


def pairwise_slot_similarity(keys: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    if keys.size(1) < 2:
        return keys.new_zeros(())
    normed = F.normalize(keys, dim=-1, eps=1e-6)
    sim = torch.einsum("bsd,btd->bst", normed, normed)
    alive_mask = alive @ alive.transpose(1, 2)
    eye = torch.eye(keys.size(1), device=keys.device, dtype=keys.dtype).unsqueeze(0)
    masked = sim * alive_mask * (1.0 - eye)
    denom = alive_mask.mul(1.0 - eye).sum().clamp_min(1.0)
    return masked.pow(2).sum() / denom


def gather_slot(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    expand = index.unsqueeze(-1).expand(-1, 1, tensor.size(-1))
    return tensor.gather(1, expand).squeeze(1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SlotRungSpec:
    name: str
    num_slots: int
    key_dim: int
    value_dim: int
    cadence: int
    horizon: int
    refresh_threshold: float = 0.55
    spawn_threshold: float = 0.60
    promote_threshold: float = 0.50
    bubble_decay: float = 0.90
    bubble_gain: float = 1.0
    bubble_threshold: float = 0.20
    target_refresh_rate: float = 0.25
    target_spawn_rate: float = 0.05
    target_promote_rate: float = 0.15
    alive_decay: float = 0.995


@dataclass
class ChronoSlotLadderConfig:
    hidden_dim: int
    rung_specs: Tuple[SlotRungSpec, ...] = field(
        default_factory=lambda: (
            SlotRungSpec("r1", num_slots=8, key_dim=96, value_dim=192, cadence=2, horizon=8, target_refresh_rate=0.35, target_spawn_rate=0.15, target_promote_rate=0.30),
            SlotRungSpec("r2", num_slots=6, key_dim=128, value_dim=256, cadence=8, horizon=32, target_refresh_rate=0.20, target_spawn_rate=0.08, target_promote_rate=0.18),
            SlotRungSpec("r3", num_slots=4, key_dim=160, value_dim=320, cadence=32, horizon=128, target_refresh_rate=0.10, target_spawn_rate=0.03, target_promote_rate=0.10),
        )
    )
    workspace_dim: int = 256
    context_hidden_dim: int = 512
    gate_hidden_dim: int = 384
    memory_token_dim: Optional[int] = None
    temperature: float = 0.25

    def __post_init__(self) -> None:
        if self.memory_token_dim is None:
            self.memory_token_dim = self.hidden_dim


@dataclass
class SlotRungState:
    keys: torch.Tensor
    values: torch.Tensor
    confidence: torch.Tensor
    age: torch.Tensor
    alive: torch.Tensor
    evidence: torch.Tensor


@dataclass
class ChronoSlotLadderState:
    rungs: Dict[str, SlotRungState]


@dataclass
class SlotRungStats:
    candidate_key: torch.Tensor
    candidate_value: torch.Tensor
    match_index: torch.Tensor
    spawn_index: torch.Tensor
    match_score: torch.Tensor
    surprise: torch.Tensor
    cadence_prior: torch.Tensor
    refresh_prob: torch.Tensor
    spawn_prob: torch.Tensor
    promote_prob: torch.Tensor
    refresh_mask: torch.Tensor
    spawn_mask: torch.Tensor
    promote_mask: torch.Tensor
    bubble: torch.Tensor
    summary: torch.Tensor
    promoted_summary: torch.Tensor
    updated_state: SlotRungState


@dataclass
class ChronoSlotLadderOutput:
    workspace: torch.Tensor
    memory_tokens: torch.Tensor
    state: ChronoSlotLadderState
    stats: Dict[str, SlotRungStats]


class TemporalWorkspace(nn.Module):
    def __init__(self, hidden_dim: int, workspace_dim: int):
        super().__init__()
        self.proj = MLP(hidden_dim * 2, hidden_dim, workspace_dim)

    def forward(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        pooled = masked_mean(hidden, attention_mask)
        last_token = hidden[:, -1]
        return self.proj(torch.cat([pooled, last_token], dim=-1))


class SlotRung(nn.Module):
    def __init__(self, cfg: ChronoSlotLadderConfig, spec: SlotRungSpec, ctx_dim: int):
        super().__init__()
        self.spec = spec
        self.candidate_key = MLP(ctx_dim, cfg.context_hidden_dim, spec.key_dim)
        self.candidate_value = MLP(ctx_dim, cfg.context_hidden_dim, spec.value_dim)
        self.value_predictor = MLP(ctx_dim + spec.value_dim, cfg.context_hidden_dim, spec.value_dim)

        gate_in = ctx_dim + spec.key_dim + (2 * spec.value_dim) + 5
        self.refresh_gate = MLP(gate_in, cfg.gate_hidden_dim, 1)
        self.spawn_gate = MLP(gate_in, cfg.gate_hidden_dim, 1)
        self.promote_gate = MLP(gate_in, cfg.gate_hidden_dim, 1)

        self.summary_proj = nn.Sequential(
            nn.Linear(spec.value_dim, cfg.memory_token_dim),
            nn.LayerNorm(cfg.memory_token_dim),
        )
        self.slot_token_proj = nn.Sequential(
            nn.Linear(spec.value_dim, cfg.memory_token_dim),
            nn.LayerNorm(cfg.memory_token_dim),
        )
        self.readout = MLP(spec.value_dim, cfg.context_hidden_dim, cfg.memory_token_dim)

    def forward(
        self,
        prev_state: SlotRungState,
        context: torch.Tensor,
        incoming_bubble: torch.Tensor,
        hard: bool = False,
        temperature: float = 0.25,
    ) -> SlotRungStats:
        candidate_key = F.normalize(self.candidate_key(context), dim=-1, eps=1e-6)
        candidate_value = self.candidate_value(context)

        match_index, match_score, matched_key, matched_value, matched_confidence, matched_age = self._match(prev_state, candidate_key)
        predicted_value = self.value_predictor(torch.cat([context, matched_value], dim=-1))
        prediction_error = (candidate_value.detach() - predicted_value).pow(2).mean(dim=-1, keepdim=True)

        cadence_prior = torch.sigmoid((matched_age - float(self.spec.cadence)) / max(float(self.spec.cadence), 1.0))
        surprise = (1.0 - match_score) + 0.25 * prediction_error + incoming_bubble
        gate_features = torch.cat(
            [
                context,
                candidate_key,
                candidate_value,
                matched_value,
                predicted_value,
                match_score,
                matched_confidence,
                cadence_prior,
                surprise,
                incoming_bubble,
            ],
            dim=-1,
        )

        refresh_prob = torch.sigmoid(self.refresh_gate(gate_features))
        spawn_prob = torch.sigmoid(self.spawn_gate(gate_features))
        promote_prob = torch.sigmoid(self.promote_gate(gate_features))
        refresh_mask, spawn_mask, promote_mask = self._action_masks(
            refresh_prob=refresh_prob,
            spawn_prob=spawn_prob,
            promote_prob=promote_prob,
            has_match=prev_state.alive.max(dim=1).values,
            hard=hard,
            temperature=temperature,
        )

        spawn_index = self._select_spawn_slot(prev_state, match_index)
        next_state = self._update_state(
            prev_state=prev_state,
            candidate_key=candidate_key,
            candidate_value=candidate_value,
            refresh_mask=refresh_mask,
            spawn_mask=spawn_mask,
            match_index=match_index,
            spawn_index=spawn_index,
        )
        summary = self._summary(next_state)
        bubble = self.spec.bubble_gain * torch.relu(next_state.evidence - self.spec.bubble_threshold)
        promoted_summary = promote_mask * self.summary_proj(summary)

        return SlotRungStats(
            candidate_key=candidate_key,
            candidate_value=candidate_value,
            match_index=match_index,
            spawn_index=spawn_index,
            match_score=match_score,
            surprise=surprise,
            cadence_prior=cadence_prior,
            refresh_prob=refresh_prob,
            spawn_prob=spawn_prob,
            promote_prob=promote_prob,
            refresh_mask=refresh_mask,
            spawn_mask=spawn_mask,
            promote_mask=promote_mask,
            bubble=bubble,
            summary=summary,
            promoted_summary=promoted_summary,
            updated_state=next_state,
        )

    def memory_tokens(self, state: SlotRungState, summary: torch.Tensor) -> torch.Tensor:
        slot_tokens = self.slot_token_proj(state.values) * state.confidence * state.alive
        summary_token = self.summary_proj(summary).unsqueeze(1)
        return torch.cat([summary_token, slot_tokens], dim=1)

    def _match(
        self,
        prev_state: SlotRungState,
        candidate_key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        live = prev_state.alive.squeeze(-1) > 0
        safe_keys = F.normalize(prev_state.keys + (1.0 - prev_state.alive) * 1e-4, dim=-1, eps=1e-6)
        scores = torch.einsum("bd,bsd->bs", candidate_key, safe_keys)
        scores = scores.masked_fill(~live, -1e4)
        has_live = live.any(dim=1, keepdim=True)
        best_score, best_index = scores.max(dim=1, keepdim=True)
        best_score = torch.where(has_live, best_score, torch.zeros_like(best_score))
        best_index = torch.where(has_live, best_index, torch.zeros_like(best_index))

        matched_key = gather_slot(prev_state.keys, best_index)
        matched_value = gather_slot(prev_state.values, best_index)
        matched_conf = gather_slot(prev_state.confidence, best_index)
        matched_age = gather_slot(prev_state.age, best_index)
        return best_index, best_score, matched_key, matched_value, matched_conf, matched_age

    def _action_masks(
        self,
        refresh_prob: torch.Tensor,
        spawn_prob: torch.Tensor,
        promote_prob: torch.Tensor,
        has_match: torch.Tensor,
        hard: bool,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hard:
            refresh_mask = ((refresh_prob > self.spec.refresh_threshold) & (has_match > 0.5)).to(refresh_prob.dtype)
            spawn_mask = ((spawn_prob > self.spec.spawn_threshold) & (refresh_mask < 0.5)).to(spawn_prob.dtype)
            promote_mask = (promote_prob > self.spec.promote_threshold).to(promote_prob.dtype)
            return refresh_mask, spawn_mask, promote_mask

        refresh_raw = torch.sigmoid((refresh_prob - self.spec.refresh_threshold) / max(temperature, 1e-4)) * has_match
        spawn_raw = torch.sigmoid((spawn_prob - self.spec.spawn_threshold) / max(temperature, 1e-4))
        denom = 1.0 + refresh_raw + spawn_raw
        refresh_mask = refresh_raw / denom
        spawn_mask = spawn_raw / denom
        promote_mask = torch.sigmoid((promote_prob - self.spec.promote_threshold) / max(temperature, 1e-4))
        return refresh_mask, spawn_mask, promote_mask

    def _select_spawn_slot(self, prev_state: SlotRungState, match_index: torch.Tensor) -> torch.Tensor:
        inactive = (prev_state.alive.squeeze(-1) < 0.5).float()
        if inactive.any():
            candidate = torch.argmax(inactive, dim=1, keepdim=True)
            has_inactive = inactive.max(dim=1, keepdim=True).values > 0.5
        else:
            candidate = torch.zeros_like(match_index)
            has_inactive = torch.zeros_like(match_index, dtype=torch.bool)

        utility = prev_state.confidence.squeeze(-1) - 0.01 * prev_state.age.squeeze(-1)
        utility = utility.masked_fill(match_index == torch.arange(prev_state.keys.size(1), device=prev_state.keys.device).unsqueeze(0), 1e4)
        replacement = torch.argmin(utility, dim=1, keepdim=True)
        return torch.where(has_inactive, candidate, replacement)

    def _update_state(
        self,
        prev_state: SlotRungState,
        candidate_key: torch.Tensor,
        candidate_value: torch.Tensor,
        refresh_mask: torch.Tensor,
        spawn_mask: torch.Tensor,
        match_index: torch.Tensor,
        spawn_index: torch.Tensor,
    ) -> SlotRungState:
        batch_size, num_slots, _ = prev_state.keys.shape
        match_hot = F.one_hot(match_index.squeeze(-1), num_classes=num_slots).to(prev_state.keys.dtype).unsqueeze(-1)
        spawn_hot = F.one_hot(spawn_index.squeeze(-1), num_classes=num_slots).to(prev_state.keys.dtype).unsqueeze(-1)

        refresh_slot_mask = match_hot * refresh_mask.unsqueeze(1)
        spawn_slot_mask = spawn_hot * spawn_mask.unsqueeze(1)
        keep_mask = (1.0 - refresh_slot_mask - spawn_slot_mask).clamp_min(0.0)

        candidate_key_exp = candidate_key.unsqueeze(1)
        candidate_value_exp = candidate_value.unsqueeze(1)

        refreshed_keys = F.normalize(
            keep_mask * prev_state.keys + refresh_slot_mask * F.normalize(prev_state.keys + candidate_key_exp, dim=-1, eps=1e-6) + spawn_slot_mask * candidate_key_exp,
            dim=-1,
            eps=1e-6,
        )
        refreshed_values = keep_mask * prev_state.values + refresh_slot_mask * (0.5 * prev_state.values + 0.5 * candidate_value_exp) + spawn_slot_mask * candidate_value_exp

        decayed_conf = (prev_state.confidence * self.spec.alive_decay).clamp(0.0, 1.0)
        refreshed_conf = keep_mask * decayed_conf + refresh_slot_mask * torch.maximum(decayed_conf, refresh_mask.unsqueeze(1)) + spawn_slot_mask * spawn_mask.unsqueeze(1)
        refreshed_alive = torch.clamp(keep_mask * prev_state.alive + refresh_slot_mask + spawn_slot_mask, 0.0, 1.0)
        refreshed_age = keep_mask * (prev_state.age + 1.0) + refresh_slot_mask * 0.0 + spawn_slot_mask * 0.0

        evidence = self.spec.bubble_decay * prev_state.evidence + (1.0 - self.spec.bubble_decay) * torch.maximum(refresh_mask, spawn_mask)
        return SlotRungState(
            keys=refreshed_keys,
            values=refreshed_values,
            confidence=refreshed_conf,
            age=refreshed_age,
            alive=refreshed_alive,
            evidence=evidence,
        )

    def _summary(self, state: SlotRungState) -> torch.Tensor:
        weights = (state.confidence * state.alive).squeeze(-1)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        normalized = weights / denom
        return torch.einsum("bs,bsd->bd", normalized, state.values)


class ChronoSlotLadderV2B(nn.Module):
    def __init__(self, cfg: ChronoSlotLadderConfig):
        super().__init__()
        self.cfg = cfg
        self.workspace = TemporalWorkspace(cfg.hidden_dim, cfg.workspace_dim)
        self.rung_order = [spec.name for spec in cfg.rung_specs]
        self.context_proj = nn.ModuleDict()
        self.rungs = nn.ModuleDict()

        for idx, spec in enumerate(cfg.rung_specs):
            ctx_dim = self._context_dim_for(idx)
            self.context_proj[spec.name] = MLP(ctx_dim, cfg.context_hidden_dim, ctx_dim)
            self.rungs[spec.name] = SlotRung(cfg, spec, ctx_dim=ctx_dim)

    def init_state(self, batch_size: int, device: torch.device) -> ChronoSlotLadderState:
        states: Dict[str, SlotRungState] = {}
        for spec in self.cfg.rung_specs:
            states[spec.name] = SlotRungState(
                keys=torch.zeros(batch_size, spec.num_slots, spec.key_dim, device=device),
                values=torch.zeros(batch_size, spec.num_slots, spec.value_dim, device=device),
                confidence=torch.zeros(batch_size, spec.num_slots, 1, device=device),
                age=torch.zeros(batch_size, spec.num_slots, 1, device=device),
                alive=torch.zeros(batch_size, spec.num_slots, 1, device=device),
                evidence=torch.zeros(batch_size, 1, device=device),
            )
        return ChronoSlotLadderState(rungs=states)

    def forward(
        self,
        hidden: torch.Tensor,
        state: ChronoSlotLadderState,
        attention_mask: Optional[torch.Tensor] = None,
        hard: bool = False,
    ) -> ChronoSlotLadderOutput:
        workspace = self.workspace(hidden, attention_mask)
        next_states: Dict[str, SlotRungState] = {}
        stats: Dict[str, SlotRungStats] = {}
        memory_tokens: List[torch.Tensor] = []

        promoted = torch.zeros(workspace.size(0), self.cfg.memory_token_dim, device=workspace.device, dtype=workspace.dtype)
        bubble = torch.zeros(workspace.size(0), 1, device=workspace.device, dtype=workspace.dtype)

        for idx, spec in enumerate(self.cfg.rung_specs):
            prev_state = state.rungs[spec.name]
            context = self._build_context(idx, workspace, promoted, bubble)
            context = self.context_proj[spec.name](context)
            rung_stats = self.rungs[spec.name](
                prev_state=prev_state,
                context=context,
                incoming_bubble=bubble,
                hard=hard,
                temperature=self.cfg.temperature,
            )

            next_states[spec.name] = rung_stats.updated_state
            stats[spec.name] = rung_stats
            bubble = rung_stats.bubble
            promoted = rung_stats.promoted_summary
            memory_tokens.append(self.rungs[spec.name].memory_tokens(rung_stats.updated_state, rung_stats.summary))

        memory = torch.cat(memory_tokens, dim=1)
        return ChronoSlotLadderOutput(
            workspace=workspace,
            memory_tokens=memory,
            state=ChronoSlotLadderState(rungs=next_states),
            stats=stats,
        )

    def _context_dim_for(self, rung_idx: int) -> int:
        if rung_idx == 0:
            return self.cfg.workspace_dim + 1
        return self.cfg.workspace_dim + self.cfg.memory_token_dim + 1

    def _build_context(
        self,
        rung_idx: int,
        workspace: torch.Tensor,
        promoted: torch.Tensor,
        bubble: torch.Tensor,
    ) -> torch.Tensor:
        if rung_idx == 0:
            return torch.cat([workspace, bubble], dim=-1)
        return torch.cat([workspace, promoted, bubble], dim=-1)


def rung_prediction_loss(readout: nn.Module, summary: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(readout(summary), target)


def anchor_reuse_loss(stats: Mapping[str, SlotRungStats]) -> torch.Tensor:
    terms = []
    for rung in stats.values():
        terms.append((rung.spawn_mask * rung.match_score).mean())
        terms.append((rung.refresh_mask * (1.0 - rung.match_score)).mean())
    return torch.stack(terms).mean()


def action_rate_loss(stats: Mapping[str, SlotRungStats], specs: Iterable[SlotRungSpec]) -> torch.Tensor:
    terms = []
    for spec in specs:
        rung = stats[spec.name]
        terms.append((rung.refresh_mask.mean() - spec.target_refresh_rate).pow(2))
        terms.append((rung.spawn_mask.mean() - spec.target_spawn_rate).pow(2))
        terms.append((rung.promote_mask.mean() - spec.target_promote_rate).pow(2))
    return torch.stack(terms).mean()


def slot_diversity_loss(state: ChronoSlotLadderState) -> torch.Tensor:
    losses = []
    for rung in state.rungs.values():
        losses.append(pairwise_slot_similarity(rung.keys, rung.alive))
    return torch.stack(losses).mean()


def alive_sparsity_loss(state: ChronoSlotLadderState) -> torch.Tensor:
    losses = [rung.alive.mean() for rung in state.rungs.values()]
    return torch.stack(losses).mean()


def summary_covariance_loss(stats: Mapping[str, SlotRungStats]) -> torch.Tensor:
    losses = [covariance_penalty(rung.summary) for rung in stats.values()]
    return torch.stack(losses).mean()


def promotion_flow_loss(stats: Mapping[str, SlotRungStats]) -> torch.Tensor:
    losses = [(rung.promote_mask * rung.surprise).mean() for rung in stats.values()]
    return torch.stack(losses).mean()


def slot_ladder_auxiliary_loss(
    model: ChronoSlotLadderV2B,
    output: ChronoSlotLadderOutput,
    horizon_targets: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    pred_terms = []
    for spec in model.cfg.rung_specs:
        rung_stats = output.stats[spec.name]
        pred_terms.append(rung_prediction_loss(model.rungs[spec.name].readout, rung_stats.summary, horizon_targets[spec.name]))

    pred = torch.stack(pred_terms).mean()
    reuse = anchor_reuse_loss(output.stats)
    rates = action_rate_loss(output.stats, model.cfg.rung_specs)
    diversity = slot_diversity_loss(output.state)
    alive = alive_sparsity_loss(output.state)
    cov = summary_covariance_loss(output.stats)
    flow = promotion_flow_loss(output.stats)

    total = pred + 0.10 * reuse + 0.05 * rates + 0.05 * diversity + 0.02 * alive + 0.05 * cov + 0.02 * flow
    return {
        "total": total,
        "pred": pred,
        "reuse": reuse,
        "rates": rates,
        "diversity": diversity,
        "alive": alive,
        "cov": cov,
        "flow": flow,
    }
