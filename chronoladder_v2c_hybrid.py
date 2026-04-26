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
class HybridRungSpec:
    name: str
    role: str
    num_slots: int
    key_dim: int
    value_dim: int
    cadence: int
    horizon: int
    refresh_threshold: float = 0.55
    spawn_threshold: float = 0.60
    promote_threshold: float = 0.50
    retire_threshold: float = 0.15
    target_refresh_rate: float = 0.20
    target_spawn_rate: float = 0.05
    target_promote_rate: float = 0.10
    confidence_decay: float = 0.995
    drift_weight: float = 0.10


@dataclass
class LedgerSpec:
    num_entries: int = 12
    value_dim: int = 256
    metadata_dim: int = 6
    write_threshold: float = 0.55
    contradiction_threshold: float = 0.60
    target_write_rate: float = 0.05
    expiry_decay: float = 0.995


@dataclass
class ChronoHybridConfig:
    hidden_dim: int
    rung_specs: Tuple[HybridRungSpec, ...] = field(
        default_factory=lambda: (
            HybridRungSpec("r1", "event", 8, 96, 192, cadence=2, horizon=8, target_refresh_rate=0.35, target_spawn_rate=0.15),
            HybridRungSpec("r2", "task_state", 6, 128, 256, cadence=8, horizon=32, target_refresh_rate=0.20, target_spawn_rate=0.08),
            HybridRungSpec("r3", "schema", 4, 160, 320, cadence=32, horizon=128, target_refresh_rate=0.10, target_spawn_rate=0.03),
        )
    )
    ledger: LedgerSpec = field(default_factory=LedgerSpec)
    workspace_dim: int = 256
    hidden_mlp_dim: int = 512
    gate_hidden_dim: int = 384
    memory_token_dim: Optional[int] = None
    temperature: float = 0.25

    def __post_init__(self) -> None:
        if self.memory_token_dim is None:
            self.memory_token_dim = self.hidden_dim


@dataclass
class LedgerEntry:
    text: str
    source: str
    confidence: float
    expiry_step: Optional[int] = None
    contradiction_of: Optional[str] = None


@dataclass
class HybridSlotState:
    keys: torch.Tensor
    values: torch.Tensor
    confidence: torch.Tensor
    age: torch.Tensor
    alive: torch.Tensor
    thread: torch.Tensor


@dataclass
class LedgerState:
    values: torch.Tensor
    confidence: torch.Tensor
    expiry: torch.Tensor
    contradiction: torch.Tensor
    alive: torch.Tensor
    metadata: torch.Tensor
    entries: Optional[List[List[LedgerEntry]]] = None


@dataclass
class ChronoHybridState:
    rungs: Dict[str, HybridSlotState]
    ledger: LedgerState


@dataclass
class HybridRungStats:
    candidate_key: torch.Tensor
    candidate_value: torch.Tensor
    match_index: torch.Tensor
    spawn_index: torch.Tensor
    match_score: torch.Tensor
    refresh_mask: torch.Tensor
    spawn_mask: torch.Tensor
    promote_mask: torch.Tensor
    retire_mask: torch.Tensor
    ledger_bias: torch.Tensor
    summary: torch.Tensor
    promoted: torch.Tensor
    updated_state: HybridSlotState


@dataclass
class LedgerStats:
    candidate_value: torch.Tensor
    write_prob: torch.Tensor
    contradiction_prob: torch.Tensor
    write_mask: torch.Tensor
    contradiction_mask: torch.Tensor
    updated_state: LedgerState


@dataclass
class ChronoHybridOutput:
    workspace: torch.Tensor
    memory_tokens: torch.Tensor
    state: ChronoHybridState
    rung_stats: Dict[str, HybridRungStats]
    ledger_stats: LedgerStats


class EvidenceEncoder(nn.Module):
    def __init__(self, hidden_dim: int, workspace_dim: int):
        super().__init__()
        self.workspace = MLP(hidden_dim * 2, hidden_dim, workspace_dim)

    def forward(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        pooled = masked_mean(hidden, attention_mask)
        last = hidden[:, -1]
        return self.workspace(torch.cat([pooled, last], dim=-1))


class LedgerBank(nn.Module):
    def __init__(self, cfg: ChronoHybridConfig):
        super().__init__()
        self.spec = cfg.ledger
        in_dim = cfg.workspace_dim + cfg.memory_token_dim
        self.candidate = MLP(in_dim, cfg.hidden_mlp_dim, self.spec.value_dim)
        gate_dim = in_dim + self.spec.value_dim + self.spec.metadata_dim
        self.write_gate = MLP(gate_dim, cfg.gate_hidden_dim, 1)
        self.contradiction_gate = MLP(gate_dim, cfg.gate_hidden_dim, 1)
        self.token_proj = nn.Sequential(
            nn.Linear(self.spec.value_dim + self.spec.metadata_dim, cfg.memory_token_dim),
            nn.LayerNorm(cfg.memory_token_dim),
        )

    def init_state(self, batch_size: int, device: torch.device) -> LedgerState:
        return LedgerState(
            values=torch.zeros(batch_size, self.spec.num_entries, self.spec.value_dim, device=device),
            confidence=torch.zeros(batch_size, self.spec.num_entries, 1, device=device),
            expiry=torch.ones(batch_size, self.spec.num_entries, 1, device=device),
            contradiction=torch.zeros(batch_size, self.spec.num_entries, 1, device=device),
            alive=torch.zeros(batch_size, self.spec.num_entries, 1, device=device),
            metadata=torch.zeros(batch_size, self.spec.num_entries, self.spec.metadata_dim, device=device),
            entries=None,
        )

    def forward(
        self,
        state: LedgerState,
        workspace: torch.Tensor,
        slow_summary_token: torch.Tensor,
        hard: bool,
        temperature: float,
    ) -> LedgerStats:
        context = torch.cat([workspace, slow_summary_token], dim=-1)
        candidate_value = self.candidate(context)
        metadata_summary = self._metadata_summary(state)
        gate_features = torch.cat([context, candidate_value, metadata_summary], dim=-1)
        write_prob = torch.sigmoid(self.write_gate(gate_features))
        contradiction_prob = torch.sigmoid(self.contradiction_gate(gate_features))

        if hard:
            write_mask = (write_prob > self.spec.write_threshold).to(write_prob.dtype)
            contradiction_mask = (contradiction_prob > self.spec.contradiction_threshold).to(write_prob.dtype)
        else:
            write_mask = torch.sigmoid((write_prob - self.spec.write_threshold) / max(temperature, 1e-4))
            contradiction_mask = torch.sigmoid((contradiction_prob - self.spec.contradiction_threshold) / max(temperature, 1e-4))

        slot_index = self._select_slot(state)
        updated = self._write_slot(state, candidate_value, write_mask, contradiction_mask, slot_index)
        return LedgerStats(
            candidate_value=candidate_value,
            write_prob=write_prob,
            contradiction_prob=contradiction_prob,
            write_mask=write_mask,
            contradiction_mask=contradiction_mask,
            updated_state=updated,
        )

    def memory_tokens(self, state: LedgerState) -> torch.Tensor:
        payload = torch.cat([state.values, state.metadata], dim=-1)
        return self.token_proj(payload) * state.alive * state.confidence * state.expiry

    def ledger_bias(self, state: LedgerState) -> torch.Tensor:
        trust = state.alive * state.confidence * state.expiry * (1.0 - state.contradiction)
        contradiction = state.alive * state.contradiction
        trust_score = trust.mean(dim=1)
        contradiction_score = contradiction.mean(dim=1)
        return torch.cat([trust_score, contradiction_score], dim=-1)

    def _metadata_summary(self, state: LedgerState) -> torch.Tensor:
        weights = state.alive * state.confidence
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (state.metadata * weights).sum(dim=1) / denom

    def _select_slot(self, state: LedgerState) -> torch.Tensor:
        inactive = (state.alive.squeeze(-1) < 0.5).float()
        has_inactive = inactive.max(dim=1, keepdim=True).values > 0.5
        inactive_choice = torch.argmax(inactive, dim=1, keepdim=True)
        utility = (state.confidence * state.expiry * (1.0 - state.contradiction)).squeeze(-1)
        replacement = torch.argmin(utility, dim=1, keepdim=True)
        return torch.where(has_inactive, inactive_choice, replacement)

    def _write_slot(
        self,
        state: LedgerState,
        candidate_value: torch.Tensor,
        write_mask: torch.Tensor,
        contradiction_mask: torch.Tensor,
        slot_index: torch.Tensor,
    ) -> LedgerState:
        batch_size, num_entries, _ = state.values.shape
        hot = F.one_hot(slot_index.squeeze(-1), num_classes=num_entries).to(state.values.dtype).unsqueeze(-1)
        write_slot = hot * write_mask.unsqueeze(1)
        keep = (1.0 - write_slot).clamp_min(0.0)

        values = keep * state.values + write_slot * candidate_value.unsqueeze(1)
        confidence = keep * (state.confidence * self.spec.expiry_decay) + write_slot * write_mask.unsqueeze(1)
        expiry = keep * (state.expiry * self.spec.expiry_decay) + write_slot
        contradiction = keep * state.contradiction + write_slot * contradiction_mask.unsqueeze(1)
        alive = torch.clamp(keep * state.alive + write_slot, 0.0, 1.0)

        metadata = state.metadata.clone()
        metadata = keep * metadata
        metadata[..., 0:1] = confidence
        metadata[..., 1:2] = expiry
        metadata[..., 2:3] = contradiction
        metadata[..., 3:4] = alive

        return LedgerState(
            values=values,
            confidence=confidence,
            expiry=expiry,
            contradiction=contradiction,
            alive=alive,
            metadata=metadata,
            entries=state.entries,
        )


class HybridSlotRung(nn.Module):
    def __init__(self, cfg: ChronoHybridConfig, spec: HybridRungSpec, context_dim: int):
        super().__init__()
        self.spec = spec
        self.key = MLP(context_dim, cfg.hidden_mlp_dim, spec.key_dim)
        self.value = MLP(context_dim, cfg.hidden_mlp_dim, spec.value_dim)
        gate_dim = context_dim + spec.key_dim + (2 * spec.value_dim) + 5
        self.refresh_gate = MLP(gate_dim, cfg.gate_hidden_dim, 1)
        self.spawn_gate = MLP(gate_dim, cfg.gate_hidden_dim, 1)
        self.promote_gate = MLP(gate_dim, cfg.gate_hidden_dim, 1)
        self.retire_gate = MLP(gate_dim, cfg.gate_hidden_dim, 1)
        self.summary_proj = nn.Sequential(
            nn.Linear(spec.value_dim, cfg.memory_token_dim),
            nn.LayerNorm(cfg.memory_token_dim),
        )
        self.slot_token_proj = nn.Sequential(
            nn.Linear(spec.value_dim, cfg.memory_token_dim),
            nn.LayerNorm(cfg.memory_token_dim),
        )
        self.readout = MLP(spec.value_dim, cfg.hidden_mlp_dim, cfg.memory_token_dim)

    def init_state(self, batch_size: int, device: torch.device) -> HybridSlotState:
        return HybridSlotState(
            keys=torch.zeros(batch_size, self.spec.num_slots, self.spec.key_dim, device=device),
            values=torch.zeros(batch_size, self.spec.num_slots, self.spec.value_dim, device=device),
            confidence=torch.zeros(batch_size, self.spec.num_slots, 1, device=device),
            age=torch.zeros(batch_size, self.spec.num_slots, 1, device=device),
            alive=torch.zeros(batch_size, self.spec.num_slots, 1, device=device),
            thread=torch.zeros(batch_size, self.spec.num_slots, 1, device=device),
        )

    def forward(
        self,
        state: HybridSlotState,
        context: torch.Tensor,
        ledger_bias: torch.Tensor,
        hard: bool,
        temperature: float,
    ) -> HybridRungStats:
        candidate_key = F.normalize(self.key(context), dim=-1, eps=1e-6)
        candidate_value = self.value(context)
        match_index, match_score, matched_value, matched_age = self._match(state, candidate_key)
        cadence_prior = torch.sigmoid((matched_age - float(self.spec.cadence)) / max(float(self.spec.cadence), 1.0))
        surprise = 1.0 - match_score

        gate_features = torch.cat(
            [
                context,
                candidate_key,
                candidate_value,
                matched_value,
                match_score,
                cadence_prior,
                surprise,
                ledger_bias,
            ],
            dim=-1,
        )
        refresh_prob = torch.sigmoid(self.refresh_gate(gate_features))
        spawn_prob = torch.sigmoid(self.spawn_gate(gate_features))
        promote_prob = torch.sigmoid(self.promote_gate(gate_features))
        retire_prob = torch.sigmoid(self.retire_gate(gate_features))

        refresh_mask, spawn_mask, promote_mask, retire_mask = self._masks(
            refresh_prob, spawn_prob, promote_prob, retire_prob, state, hard, temperature
        )
        spawn_index = self._select_spawn_slot(state, match_index)
        updated = self._update_state(state, candidate_key, candidate_value, refresh_mask, spawn_mask, retire_mask, match_index, spawn_index)
        summary = self._summary(updated)
        promoted = promote_mask * self.summary_proj(summary)

        return HybridRungStats(
            candidate_key=candidate_key,
            candidate_value=candidate_value,
            match_index=match_index,
            spawn_index=spawn_index,
            match_score=match_score,
            refresh_mask=refresh_mask,
            spawn_mask=spawn_mask,
            promote_mask=promote_mask,
            retire_mask=retire_mask,
            ledger_bias=ledger_bias,
            summary=summary,
            promoted=promoted,
            updated_state=updated,
        )

    def memory_tokens(self, state: HybridSlotState, summary: torch.Tensor) -> torch.Tensor:
        slot_tokens = self.slot_token_proj(state.values) * state.alive * state.confidence
        return torch.cat([self.summary_proj(summary).unsqueeze(1), slot_tokens], dim=1)

    def _match(
        self,
        state: HybridSlotState,
        candidate_key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        live = state.alive.squeeze(-1) > 0
        safe_keys = F.normalize(state.keys + (1.0 - state.alive) * 1e-4, dim=-1, eps=1e-6)
        scores = torch.einsum("bd,bsd->bs", candidate_key, safe_keys)
        scores = scores.masked_fill(~live, -1e4)
        has_live = live.any(dim=1, keepdim=True)
        best_score, best_index = scores.max(dim=1, keepdim=True)
        best_score = torch.where(has_live, best_score, torch.zeros_like(best_score))
        best_index = torch.where(has_live, best_index, torch.zeros_like(best_index))
        return best_index, best_score, gather_slot(state.values, best_index), gather_slot(state.age, best_index)

    def _masks(
        self,
        refresh_prob: torch.Tensor,
        spawn_prob: torch.Tensor,
        promote_prob: torch.Tensor,
        retire_prob: torch.Tensor,
        state: HybridSlotState,
        hard: bool,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        has_live = state.alive.max(dim=1).values
        if hard:
            refresh = ((refresh_prob > self.spec.refresh_threshold) & (has_live > 0.5)).to(refresh_prob.dtype)
            spawn = ((spawn_prob > self.spec.spawn_threshold) & (refresh < 0.5)).to(spawn_prob.dtype)
            promote = (promote_prob > self.spec.promote_threshold).to(promote_prob.dtype)
            retire = (retire_prob > self.spec.retire_threshold).to(retire_prob.dtype)
            return refresh, spawn, promote, retire

        refresh_raw = torch.sigmoid((refresh_prob - self.spec.refresh_threshold) / max(temperature, 1e-4)) * has_live
        spawn_raw = torch.sigmoid((spawn_prob - self.spec.spawn_threshold) / max(temperature, 1e-4))
        denom = 1.0 + refresh_raw + spawn_raw
        refresh = refresh_raw / denom
        spawn = spawn_raw / denom
        promote = torch.sigmoid((promote_prob - self.spec.promote_threshold) / max(temperature, 1e-4))
        retire = torch.sigmoid((retire_prob - self.spec.retire_threshold) / max(temperature, 1e-4))
        return refresh, spawn, promote, retire

    def _select_spawn_slot(self, state: HybridSlotState, match_index: torch.Tensor) -> torch.Tensor:
        inactive = (state.alive.squeeze(-1) < 0.5).float()
        has_inactive = inactive.max(dim=1, keepdim=True).values > 0.5
        inactive_choice = torch.argmax(inactive, dim=1, keepdim=True)
        utility = (state.confidence - 0.01 * state.age).squeeze(-1)
        slot_ids = torch.arange(state.keys.size(1), device=state.keys.device).unsqueeze(0)
        utility = utility.masked_fill(match_index == slot_ids, 1e4)
        replacement = torch.argmin(utility, dim=1, keepdim=True)
        return torch.where(has_inactive, inactive_choice, replacement)

    def _update_state(
        self,
        state: HybridSlotState,
        candidate_key: torch.Tensor,
        candidate_value: torch.Tensor,
        refresh_mask: torch.Tensor,
        spawn_mask: torch.Tensor,
        retire_mask: torch.Tensor,
        match_index: torch.Tensor,
        spawn_index: torch.Tensor,
    ) -> HybridSlotState:
        _, num_slots, _ = state.keys.shape
        match_hot = F.one_hot(match_index.squeeze(-1), num_classes=num_slots).to(state.keys.dtype).unsqueeze(-1)
        spawn_hot = F.one_hot(spawn_index.squeeze(-1), num_classes=num_slots).to(state.keys.dtype).unsqueeze(-1)
        refresh_slot = match_hot * refresh_mask.unsqueeze(1)
        spawn_slot = spawn_hot * spawn_mask.unsqueeze(1)
        retire_slot = match_hot * retire_mask.unsqueeze(1)
        keep = (1.0 - refresh_slot - spawn_slot - retire_slot).clamp_min(0.0)

        key_exp = candidate_key.unsqueeze(1)
        value_exp = candidate_value.unsqueeze(1)
        keys = F.normalize(
            keep * state.keys
            + refresh_slot * F.normalize(state.keys + key_exp, dim=-1, eps=1e-6)
            + spawn_slot * key_exp,
            dim=-1,
            eps=1e-6,
        )
        values = keep * state.values + refresh_slot * (0.5 * state.values + 0.5 * value_exp) + spawn_slot * value_exp
        confidence = keep * (state.confidence * self.spec.confidence_decay) + refresh_slot * refresh_mask.unsqueeze(1) + spawn_slot * spawn_mask.unsqueeze(1)
        age = keep * (state.age + 1.0)
        alive = torch.clamp(keep * state.alive + refresh_slot + spawn_slot, 0.0, 1.0)
        thread = keep * state.thread + spawn_slot * spawn_index.to(state.thread.dtype).unsqueeze(1)
        return HybridSlotState(keys=keys, values=values, confidence=confidence, age=age, alive=alive, thread=thread)

    def _summary(self, state: HybridSlotState) -> torch.Tensor:
        weights = (state.alive * state.confidence).squeeze(-1)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.einsum("bs,bsd->bd", weights / denom, state.values)


class ChronoHybridLadderV2C(nn.Module):
    def __init__(self, cfg: ChronoHybridConfig):
        super().__init__()
        self.cfg = cfg
        self.evidence = EvidenceEncoder(cfg.hidden_dim, cfg.workspace_dim)
        self.ledger = LedgerBank(cfg)
        self.rung_order = [spec.name for spec in cfg.rung_specs]
        self.rungs = nn.ModuleDict()
        self.context_proj = nn.ModuleDict()
        for idx, spec in enumerate(cfg.rung_specs):
            context_dim = self._context_dim(idx)
            self.context_proj[spec.name] = MLP(context_dim, cfg.hidden_mlp_dim, context_dim)
            self.rungs[spec.name] = HybridSlotRung(cfg, spec, context_dim)

    def init_state(self, batch_size: int, device: torch.device) -> ChronoHybridState:
        rungs = {spec.name: self.rungs[spec.name].init_state(batch_size, device) for spec in self.cfg.rung_specs}
        return ChronoHybridState(rungs=rungs, ledger=self.ledger.init_state(batch_size, device))

    def forward(
        self,
        hidden: torch.Tensor,
        state: ChronoHybridState,
        attention_mask: Optional[torch.Tensor] = None,
        hard: bool = False,
    ) -> ChronoHybridOutput:
        workspace = self.evidence(hidden, attention_mask)
        promoted = torch.zeros(workspace.size(0), self.cfg.memory_token_dim, device=workspace.device, dtype=workspace.dtype)
        ledger_bias = self.ledger.ledger_bias(state.ledger)
        next_rungs: Dict[str, HybridSlotState] = {}
        stats: Dict[str, HybridRungStats] = {}
        memory_tokens: List[torch.Tensor] = []

        for idx, spec in enumerate(self.cfg.rung_specs):
            context = self._build_context(idx, workspace, promoted, ledger_bias)
            context = self.context_proj[spec.name](context)
            rung_stats = self.rungs[spec.name](
                state=state.rungs[spec.name],
                context=context,
                ledger_bias=ledger_bias,
                hard=hard,
                temperature=self.cfg.temperature,
            )
            next_rungs[spec.name] = rung_stats.updated_state
            stats[spec.name] = rung_stats
            promoted = rung_stats.promoted
            memory_tokens.append(self.rungs[spec.name].memory_tokens(rung_stats.updated_state, rung_stats.summary))

        ledger_stats = self.ledger(state.ledger, workspace, promoted, hard=hard, temperature=self.cfg.temperature)
        ledger_tokens = self.ledger.memory_tokens(ledger_stats.updated_state)
        memory = torch.cat([*memory_tokens, ledger_tokens], dim=1)
        return ChronoHybridOutput(
            workspace=workspace,
            memory_tokens=memory,
            state=ChronoHybridState(rungs=next_rungs, ledger=ledger_stats.updated_state),
            rung_stats=stats,
            ledger_stats=ledger_stats,
        )

    def _context_dim(self, rung_idx: int) -> int:
        if rung_idx == 0:
            return self.cfg.workspace_dim + 2
        return self.cfg.workspace_dim + self.cfg.memory_token_dim + 2

    def _build_context(
        self,
        rung_idx: int,
        workspace: torch.Tensor,
        promoted: torch.Tensor,
        ledger_bias: torch.Tensor,
    ) -> torch.Tensor:
        if rung_idx == 0:
            return torch.cat([workspace, ledger_bias], dim=-1)
        return torch.cat([workspace, promoted, ledger_bias], dim=-1)


def hybrid_prediction_loss(readout: nn.Module, summary: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(readout(summary), target)


def hybrid_action_rate_loss(stats: Mapping[str, HybridRungStats], specs: Iterable[HybridRungSpec]) -> torch.Tensor:
    terms = []
    for spec in specs:
        rung = stats[spec.name]
        terms.append((rung.refresh_mask.mean() - spec.target_refresh_rate).pow(2))
        terms.append((rung.spawn_mask.mean() - spec.target_spawn_rate).pow(2))
        terms.append((rung.promote_mask.mean() - spec.target_promote_rate).pow(2))
    return torch.stack(terms).mean()


def ledger_rate_loss(stats: LedgerStats, spec: LedgerSpec) -> torch.Tensor:
    return (stats.write_mask.mean() - spec.target_write_rate).pow(2)


def slow_drift_loss(output: ChronoHybridOutput, specs: Iterable[HybridRungSpec]) -> torch.Tensor:
    terms = []
    for spec in specs:
        state = output.state.rungs[spec.name]
        movement = state.values.pow(2).mean()
        terms.append(movement * spec.drift_weight * (1.0 / max(float(spec.cadence), 1.0)))
    return torch.stack(terms).mean()


def ledger_consistency_loss(state: LedgerState) -> torch.Tensor:
    contradiction_with_high_trust = state.contradiction * state.confidence * state.alive
    expired_alive = (1.0 - state.expiry) * state.alive
    return contradiction_with_high_trust.mean() + expired_alive.mean()


def hybrid_auxiliary_loss(
    model: ChronoHybridLadderV2C,
    output: ChronoHybridOutput,
    horizon_targets: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    pred_terms = []
    cov_terms = []
    for spec in model.cfg.rung_specs:
        rung = output.rung_stats[spec.name]
        pred_terms.append(hybrid_prediction_loss(model.rungs[spec.name].readout, rung.summary, horizon_targets[spec.name]))
        cov_terms.append(covariance_penalty(rung.summary))

    pred = torch.stack(pred_terms).mean()
    cov = torch.stack(cov_terms).mean()
    rates = hybrid_action_rate_loss(output.rung_stats, model.cfg.rung_specs)
    ledger_rates = ledger_rate_loss(output.ledger_stats, model.cfg.ledger)
    drift = slow_drift_loss(output, model.cfg.rung_specs)
    ledger_consistency = ledger_consistency_loss(output.state.ledger)
    total = pred + 0.05 * cov + 0.05 * rates + 0.05 * ledger_rates + 0.05 * drift + 0.10 * ledger_consistency
    return {
        "total": total,
        "pred": pred,
        "cov": cov,
        "rates": rates,
        "ledger_rates": ledger_rates,
        "drift": drift,
        "ledger_consistency": ledger_consistency,
    }
