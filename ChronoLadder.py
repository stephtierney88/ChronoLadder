from __future__ import annotations

"""
ChronoLadder – Toggle‑able reference implementation (PyTorch ≥ 2.1)
-------------------------------------------------------------------
Five‑rung cadence memory ladder over a GPT‑2 Medium backbone.

⚙️  **Config flags**
    use_tags          – frozen horizon‑ID tags (anti‑soup)              [default=True]
    use_contrastive   – InfoNCE loss between rungs                      [default=True]
    use_critic        – slow‑tier latent critics                        [default=False]
    dropout_p         – slot‑dropout probability                        [default=0.1]
    tag_dim           – dimensionality of horizon‑ID vectors            [default=32]

The defaults match a "production‑leaning" setup (continuous memory rungs
with tag disentanglement and advanced write‑gate) while leaving the more
speculative critic off unless explicitly enabled.
"""

import math
import random
import string
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# -----------------------------------------------------------------------------
#  Configuration
# -----------------------------------------------------------------------------

class CLConfig:
    def __init__(
        self,
        tag_dim: int = 32,
        use_tags: bool = True,
        use_contrastive: bool = True,
        use_critic: bool = False,
        dropout_p: float = 0.1,
    ):
        self.tag_dim = tag_dim
        self.use_tags = use_tags
        self.use_contrastive = use_contrastive
        self.use_critic = use_critic
        self.dropout_p = dropout_p

# -----------------------------------------------------------------------------
#  Utility losses
# -----------------------------------------------------------------------------

def orthogonality_loss(latents: List[torch.Tensor]) -> torch.Tensor:
    if len(latents) < 2:
        return latents[0].new_zeros([])
    loss, pairs = 0.0, 0
    for i in range(len(latents)):
        for j in range(i + 1, len(latents)):
            loss += F.cosine_similarity(latents[i], latents[j], dim=-1).mean()
            pairs += 1
    return loss / pairs


def horizon_contrastive(latents_by_rung: List[torch.Tensor]) -> torch.Tensor:
    """InfoNCE contrast across rungs (each rung = class)."""
    anchors = torch.cat(latents_by_rung, 0)
    logits = (anchors @ anchors.T) * 0.1  # temperature scaling
    labels = torch.arange(len(latents_by_rung), device=anchors.device)
    return F.cross_entropy(logits, labels)

# -----------------------------------------------------------------------------
#  Critic for slow tiers (optional)
# -----------------------------------------------------------------------------

class SlowTierCritic(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)

# -----------------------------------------------------------------------------
#  AutoEncoder (tiny MLP stub)
# -----------------------------------------------------------------------------

class AutoEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, in_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

# -----------------------------------------------------------------------------
#  Memory rung
# -----------------------------------------------------------------------------

class MemoryRung(nn.Module):
    def __init__(
        self,
        name: str,
        cadence: int,
        in_dim: int,
        latent_dim: int,
        horizon_id: int,
        cfg: CLConfig,
        slots: int = 1,
    ):
        super().__init__()
        self.name, self.tau, self.slots = name, cadence, slots
        self.cfg = cfg
        self.ae = AutoEncoder(in_dim, latent_dim)

        # horizon tag (frozen one‑hot) or zero vector
        if cfg.use_tags:
            tag = F.one_hot(torch.tensor(horizon_id), cfg.tag_dim).float()
        else:
            tag = torch.zeros(cfg.tag_dim)
        self.register_buffer("h_tag", tag, persistent=False)

        # advanced two‑layer write gate conditioned on [x, prev, tag]
        self.write_gate = nn.Sequential(
            nn.Linear(in_dim + latent_dim + cfg.tag_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.register_buffer("latents", torch.zeros(slots, latent_dim))
        self.step_counter = 0

    # ---------------------------------------------------------
    def _stateful_write(self, x: torch.Tensor):
        prev = self.latents[0]
        enc = self.ae.encode(x)
        feat = torch.cat([x, prev, self.h_tag], -1)
        p = torch.sigmoid(self.write_gate(feat))
        new_latent = p * enc + (1 - p) * prev
        # slot‑dropout
        if self.training and torch.rand(1, device=x.device) < self.cfg.dropout_p:
            new_latent *= 0.0
        self.latents[0] = new_latent.detach()

    def _kl_refresh(self):
        with torch.no_grad():
            decoded = self.ae.decode(self.latents)
            self.latents.copy_(self.ae.encode(decoded))

    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor):
        self.step_counter += 1
        if self.step_counter % self.tau == 0:
            self._stateful_write(x.detach())
        if self.tau >= 64 and self.step_counter % (self.tau * 4) == 0:
            self._kl_refresh()
        return torch.cat([self.h_tag, self.latents.view(-1)], -1)

    # ----------------- Aux losses ----------------------------
    def aux_losses(self, target: torch.Tensor) -> torch.Tensor:
        recon = self.ae.decode(self.latents)
        return F.mse_loss(recon, target.expand_as(recon))

# -----------------------------------------------------------------------------
#  ChronoLadder LM wrapper
# -----------------------------------------------------------------------------

class ChronoLadderLM(nn.Module):
    def __init__(self, backbone_name: str = "gpt2-medium", cfg: CLConfig | None = None):
        super().__init__()
        self.cfg = cfg or CLConfig()
        self.backbone = GPT2LMHeadModel.from_pretrained(backbone_name)
        h = self.backbone.config.n_embd
        # build rungs
        self.rungs = nn.ModuleList([
            MemoryRung("AE1", 1, h, 256, 0, self.cfg),
            MemoryRung("AE4", 4, h, 512, 1, self.cfg),
            MemoryRung("AE16", 16, h, 768, 2, self.cfg, slots=2),
            MemoryRung("AE64", 64, h, 1024, 3, self.cfg, slots=2),
            MemoryRung("AE256", 256, h, 2048, 4, self.cfg),
        ])
        fused_dim = sum(r.latents.numel() // r.slots + self.cfg.tag_dim for r in self.rungs)
        self.mem_proj = nn.Sequential(nn.Linear(fused_dim, h), nn.LayerNorm(h))

        # critics (optional)
        if self.cfg.use_critic:
            self.critics = nn.ModuleDict({
                r.name: SlowTierCritic(r.latents.size(-1)) for r in self.rungs if r.tau >= 64
            })
        else:
            self.critics = nn.ModuleDict()

    # ---------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, hidden_state: torch.Tensor):
        rung_lat = [r(hidden_state.detach()) for r in self.rungs]
        mem = self.mem_proj(torch.cat(rung_lat, -1))
        conditioned = hidden_state + mem
        out = self.backbone(inputs_embeds=conditioned, labels=input_ids)
        return out.loss, rung_lat

# -----------------------------------------------------------------------------
#  Trainer
# -----------------------------------------------------------------------------

class ChronoTrainer:
    def __init__(self, model: ChronoLadderLM, tok: GPT2Tokenizer, device: str | None = None):
        self.m = model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = tok
        self.opt = torch.optim.AdamW(self.m.parameters(), lr=3e-5)
        self.device = self.m.device

    # -----------------------------------------------------
    def step(self, prompts: List[str]):
        ids = self.tok(prompts, return_tensors="pt", padding=True).input_ids.to(self.device)
        with torch.no_grad():
            h = self.m.backbone.transformer.wte(ids)
        lm_loss, rung_lat = self.m(ids, h)

        # Aux losses
        recon = sum(r.aux_losses(h.mean(1)) for r in self.m.rungs) * 0.1
        contrast = 0.0
        if self.m.cfg.use_contrastive:
            contrast = horizon_contrastive([l.view(1, -1) for l in rung_lat]) * 0.05

        critic_loss = 0.0
        if self.m.cfg.use_critic:
            for r, lat in zip(self.m.rungs, rung_lat):
                if r.name in self.m.critics:
                    critic = self.m.critics[r.name]
                    critic_loss += F.mse_loss(critic(lat.detach()), torch.zeros_like(critic(lat))) * 0.02

        total = lm_loss + recon + contrast + critic_loss
        total.backward()
        nn.utils.clip_grad_norm_(self.m.parameters(), 1.0)
        self.opt.step(); self.opt.zero_grad()
        return {
            "total": total.item(),
            "task": lm_loss.item(),
            "recon": recon.item(),
            "contrast": contrast if isinstance(contrast, float) else contrast.item(),
            "critic": critic_loss if isinstance(critic_loss, float) else critic_loss.item(),
        }

# -----------------------------------------------------------------------------
#  Synthetic dataset helper
# -----------------------------------------------------------------------------

def make_copy_dataset(delay: int = 64, size: int = 4000):
    data, abc = [], list(string.ascii_lowercase)
    for _ in range(size):
        t = random.choice(abc)
        prompt = f"remember {t} then wait {delay} steps " + "x " * delay + "now what?"
        data.append(prompt)
    return data

# -----------------------------------------------------------------------------
#  Quick demo run
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    tok = GPT2Tokenizer.from_pretrained("gpt2-medium")
    cfg = CLConfig()  # defaults: tags+contrastive on, critic off
    model = ChronoLadderLM(cfg=cfg)
    trainer = ChronoTrainer(model, tok)

    dataset = make_copy_dataset()
    for step in range(1000):
        batch = random.sample(dataset, k=4)
        metrics = trainer.step(batch)
        if step % 100 == 0:
            print(
                f"step {step} | total {metrics['total']:.3f} "
                f"task {metrics['task']:.3f} recon {metrics['recon']:.3f} "
                f"ctr {metrics['contrast']:.3f} critic {metrics['critic']:.3f}"
            )
