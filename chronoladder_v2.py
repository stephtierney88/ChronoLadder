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


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a, b, dim=-1, eps=1e-6)


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("off_diagonal expects a square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_covariance_loss(z: torch.Tensor) -> torch.Tensor:
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(z.size(0) - 1, 1)
    return off_diagonal(cov).pow(2).mean()


def cross_rung_redundancy_loss(latents: Iterable[torch.Tensor]) -> torch.Tensor:
    tensors = [F.normalize(z, dim=-1) for z in latents]
    if len(tensors) < 2:
        first = tensors[0]
        return first.new_zeros(())
    losses = []
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            corr = (tensors[i].T @ tensors[j]) / max(tensors[i].size(0), 1)
            losses.append(corr.pow(2).mean())
    return torch.stack(losses).mean()


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
class RungSpec:
    name: str
    latent_dim: int
    cadence: int
    horizon: int
    bubble_decay: float = 0.90
    bubble_gain: float = 1.0
    bubble_threshold: float = 0.25
    open_threshold: float = 0.65
    close_threshold: float = 0.45
    target_write_rate: float = 0.10
    inertia_weight: float = 1.0
    surprise_l2_weight: float = 1.0
    surprise_cos_weight: float = 0.25


@dataclass
class ChronoLadderV2Config:
    hidden_dim: int
    topology: str = "linear"
    rung_specs: Tuple[RungSpec, ...] = field(
        default_factory=lambda: (
            RungSpec("r1", latent_dim=256, cadence=2, horizon=8, target_write_rate=0.35, inertia_weight=0.05),
            RungSpec("r2", latent_dim=384, cadence=8, horizon=32, target_write_rate=0.15, inertia_weight=0.20),
            RungSpec("r3", latent_dim=512, cadence=32, horizon=128, target_write_rate=0.05, inertia_weight=0.50),
        )
    )
    workspace_dim: int = 256
    write_hidden_dim: int = 256
    proposal_hidden_dim: int = 512
    temperature: float = 0.25
    memory_token_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.topology not in {"linear", "hierarchical"}:
            raise ValueError("topology must be 'linear' or 'hierarchical'")
        if self.memory_token_dim is None:
            self.memory_token_dim = self.hidden_dim


@dataclass
class RungState:
    latent: torch.Tensor
    evidence: torch.Tensor
    age: torch.Tensor
    open_mask: torch.Tensor


@dataclass
class ChronoLadderState:
    rungs: Dict[str, RungState]


@dataclass
class RungStats:
    proposal: torch.Tensor
    prediction: torch.Tensor
    surprise: torch.Tensor
    cadence_prior: torch.Tensor
    write_prob: torch.Tensor
    write_mask: torch.Tensor
    bubble: torch.Tensor
    evidence: torch.Tensor
    previous_latent: torch.Tensor
    updated_latent: torch.Tensor
    age: torch.Tensor


@dataclass
class ChronoLadderOutput:
    workspace: torch.Tensor
    memory_tokens: torch.Tensor
    state: ChronoLadderState
    stats: Dict[str, RungStats]


class TemporalWorkspace(nn.Module):
    def __init__(self, hidden_dim: int, workspace_dim: int):
        super().__init__()
        self.proj = MLP(hidden_dim * 2, hidden_dim, workspace_dim)

    def forward(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        mean_pool = masked_mean(hidden, attention_mask)
        last_token = hidden[:, -1]
        return self.proj(torch.cat([mean_pool, last_token], dim=-1))


class ChronoRung(nn.Module):
    def __init__(self, cfg: ChronoLadderV2Config, spec: RungSpec, ctx_dim: int):
        super().__init__()
        self.spec = spec
        self.proposal = MLP(spec.latent_dim + ctx_dim, cfg.proposal_hidden_dim, spec.latent_dim)
        self.predictor = MLP(spec.latent_dim + ctx_dim, cfg.proposal_hidden_dim, spec.latent_dim)
        gate_in = spec.latent_dim * 2 + ctx_dim + 4
        self.write_head = MLP(gate_in, cfg.write_hidden_dim, 1)
        self.memory_proj = nn.Sequential(
            nn.Linear(spec.latent_dim, cfg.memory_token_dim),
            nn.LayerNorm(cfg.memory_token_dim),
        )
        self.readout = MLP(spec.latent_dim, cfg.proposal_hidden_dim, cfg.memory_token_dim)

    def forward(
        self,
        prev_state: RungState,
        context: torch.Tensor,
        incoming_bubble: torch.Tensor,
        hard: bool = False,
        temperature: float = 0.25,
    ) -> Tuple[RungState, RungStats]:
        proposal = self.proposal(torch.cat([prev_state.latent, context], dim=-1))
        prediction = self.predictor(torch.cat([prev_state.latent, context], dim=-1))

        l2_term = (proposal.detach() - prediction).pow(2).mean(dim=-1, keepdim=True)
        cos_term = cosine_distance(proposal.detach(), prediction).unsqueeze(-1)
        surprise = self.spec.surprise_l2_weight * l2_term + self.spec.surprise_cos_weight * cos_term

        evidence = self.spec.bubble_decay * prev_state.evidence + (1.0 - self.spec.bubble_decay) * surprise + incoming_bubble

        cadence_prior = torch.sigmoid((prev_state.age - float(self.spec.cadence)) / max(float(self.spec.cadence), 1.0))
        gate_features = torch.cat(
            [
                proposal,
                prev_state.latent,
                context,
                surprise,
                evidence,
                cadence_prior,
                prev_state.age / max(float(self.spec.cadence), 1.0),
            ],
            dim=-1,
        )
        write_prob = torch.sigmoid(self.write_head(gate_features))
        write_mask = self._write_mask(write_prob, prev_state.open_mask, hard=hard, temperature=temperature)

        updated_latent = (1.0 - write_mask) * prev_state.latent + write_mask * proposal
        updated_age = (1.0 - write_mask) * (prev_state.age + 1.0)
        bubble = self.spec.bubble_gain * torch.relu(evidence - self.spec.bubble_threshold)
        next_state = RungState(
            latent=updated_latent,
            evidence=evidence,
            age=updated_age,
            open_mask=write_mask,
        )
        stats = RungStats(
            proposal=proposal,
            prediction=prediction,
            surprise=surprise,
            cadence_prior=cadence_prior,
            write_prob=write_prob,
            write_mask=write_mask,
            bubble=bubble,
            evidence=evidence,
            previous_latent=prev_state.latent,
            updated_latent=updated_latent,
            age=prev_state.age,
        )
        return next_state, stats

    def _write_mask(
        self,
        write_prob: torch.Tensor,
        prev_open: torch.Tensor,
        hard: bool,
        temperature: float,
    ) -> torch.Tensor:
        if hard:
            opened = write_prob > self.spec.open_threshold
            held = (prev_open > 0.5) & (write_prob > self.spec.close_threshold)
            return (opened | held).to(write_prob.dtype)
        open_logit = (write_prob - self.spec.open_threshold) / max(temperature, 1e-4)
        hold_logit = (write_prob - self.spec.close_threshold) / max(temperature, 1e-4)
        smooth_open = torch.sigmoid(open_logit)
        smooth_hold = torch.sigmoid(hold_logit) * prev_open
        return torch.clamp(torch.maximum(smooth_open, smooth_hold), 0.0, 1.0)


class ChronoLadderV2(nn.Module):
    def __init__(self, cfg: ChronoLadderV2Config):
        super().__init__()
        self.cfg = cfg
        self.workspace = TemporalWorkspace(cfg.hidden_dim, cfg.workspace_dim)
        self.rung_order = [spec.name for spec in cfg.rung_specs]
        self.context_proj = nn.ModuleDict()
        self.rungs = nn.ModuleDict()

        for idx, spec in enumerate(cfg.rung_specs):
            ctx_dim = self._context_dim_for(idx)
            self.context_proj[spec.name] = MLP(ctx_dim, cfg.proposal_hidden_dim, ctx_dim)
            self.rungs[spec.name] = ChronoRung(cfg, spec, ctx_dim=ctx_dim)

    def init_state(self, batch_size: int, device: torch.device) -> ChronoLadderState:
        states: Dict[str, RungState] = {}
        for spec in self.cfg.rung_specs:
            states[spec.name] = RungState(
                latent=torch.zeros(batch_size, spec.latent_dim, device=device),
                evidence=torch.zeros(batch_size, 1, device=device),
                age=torch.zeros(batch_size, 1, device=device),
                open_mask=torch.zeros(batch_size, 1, device=device),
            )
        return ChronoLadderState(rungs=states)

    def forward(
        self,
        hidden: torch.Tensor,
        state: ChronoLadderState,
        attention_mask: Optional[torch.Tensor] = None,
        hard: bool = False,
    ) -> ChronoLadderOutput:
        workspace = self.workspace(hidden, attention_mask)
        next_states: Dict[str, RungState] = {}
        stats: Dict[str, RungStats] = {}
        memory_tokens: List[torch.Tensor] = []

        bubble = torch.zeros(workspace.size(0), 1, device=workspace.device, dtype=workspace.dtype)
        lower_latents: List[torch.Tensor] = []

        for idx, spec in enumerate(self.cfg.rung_specs):
            name = spec.name
            prev_state = state.rungs[name]
            context = self._build_context(idx, workspace, state, next_states, lower_latents, bubble)
            context = self.context_proj[name](context)
            rung_state, rung_stats = self.rungs[name](
                prev_state=prev_state,
                context=context,
                incoming_bubble=bubble,
                hard=hard,
                temperature=self.cfg.temperature,
            )

            next_states[name] = rung_state
            stats[name] = rung_stats
            bubble = rung_stats.bubble
            lower_latents.append(rung_state.latent)
            memory_tokens.append(self.rungs[name].memory_proj(rung_state.latent))

        memory = torch.stack(memory_tokens, dim=1)
        return ChronoLadderOutput(
            workspace=workspace,
            memory_tokens=memory,
            state=ChronoLadderState(rungs=next_states),
            stats=stats,
        )

    def _context_dim_for(self, rung_idx: int) -> int:
        if self.cfg.topology == "linear":
            if rung_idx == 0:
                return self.cfg.workspace_dim
            prev_dim = self.cfg.rung_specs[rung_idx - 1].latent_dim
            return self.cfg.workspace_dim + prev_dim + 1
        total = self.cfg.workspace_dim + 1
        for spec in self.cfg.rung_specs[:rung_idx]:
            total += spec.latent_dim
        return total

    def _build_context(
        self,
        rung_idx: int,
        workspace: torch.Tensor,
        state: ChronoLadderState,
        next_states: Mapping[str, RungState],
        lower_latents: List[torch.Tensor],
        bubble: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.topology == "linear":
            if rung_idx == 0:
                return workspace
            prev_name = self.cfg.rung_specs[rung_idx - 1].name
            prev_latent = next_states[prev_name].latent
            return torch.cat([workspace, prev_latent, bubble], dim=-1)

        if not lower_latents:
            return torch.cat([workspace, bubble], dim=-1)
        return torch.cat([workspace, *lower_latents, bubble], dim=-1)


def horizon_prediction_loss(readout: nn.Module, latent: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(readout(latent), target)


def invariance_info_nce(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)
    pos_logits = (anchor * positive).sum(dim=-1, keepdim=True)
    neg_logits = anchor @ negatives.T
    logits = torch.cat([pos_logits, neg_logits], dim=-1) / temperature
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)


def inertia_loss(stats: Mapping[str, RungStats], specs: Iterable[RungSpec]) -> torch.Tensor:
    terms = []
    for spec in specs:
        rung = stats[spec.name]
        movement = (rung.updated_latent - rung.previous_latent).pow(2).mean(dim=-1, keepdim=True)
        terms.append(((1.0 - rung.write_mask.detach()) * movement).mean() * spec.inertia_weight)
    return torch.stack(terms).mean()


def write_rate_loss(stats: Mapping[str, RungStats], specs: Iterable[RungSpec]) -> torch.Tensor:
    terms = []
    for spec in specs:
        observed = stats[spec.name].write_mask.mean()
        terms.append((observed - spec.target_write_rate).pow(2))
    return torch.stack(terms).mean()


def bubble_stability_loss(stats: Mapping[str, RungStats]) -> torch.Tensor:
    losses = [rung.bubble.mean() for rung in stats.values()]
    return torch.stack(losses).mean()


def ladder_auxiliary_loss(
    model: ChronoLadderV2,
    output: ChronoLadderOutput,
    horizon_targets: Mapping[str, torch.Tensor],
    positives: Optional[Mapping[str, torch.Tensor]] = None,
    negatives: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    pred_terms = []
    cov_terms = []
    inv_terms = []

    latents = []
    for spec in model.cfg.rung_specs:
        name = spec.name
        latent = output.state.rungs[name].latent
        latents.append(latent)
        pred_terms.append(horizon_prediction_loss(model.rungs[name].readout, latent, horizon_targets[name]))
        cov_terms.append(vicreg_covariance_loss(latent))
        if positives is not None and negatives is not None and name in positives and name in negatives:
            inv_terms.append(invariance_info_nce(latent, positives[name], negatives[name]))

    total_pred = torch.stack(pred_terms).mean()
    total_cov = torch.stack(cov_terms).mean()
    total_red = cross_rung_redundancy_loss(latents)
    total_inertia = inertia_loss(output.stats, model.cfg.rung_specs)
    total_write = write_rate_loss(output.stats, model.cfg.rung_specs)
    total_bubble = bubble_stability_loss(output.stats)
    total_inv = torch.stack(inv_terms).mean() if inv_terms else total_pred.new_zeros(())

    total = total_pred + 0.25 * total_inv + 0.05 * total_cov + 0.05 * total_red + 0.10 * total_inertia + 0.05 * total_write + 0.01 * total_bubble
    return {
        "total": total,
        "pred": total_pred,
        "inv": total_inv,
        "cov": total_cov,
        "redundancy": total_red,
        "inertia": total_inertia,
        "write": total_write,
        "bubble": total_bubble,
    }
