from __future__ import annotations
"""
ChronoLadder v2 – Hierarchical Memory Ladder with implicit memory‑use
rewards (PyTorch ≥ 2.1)
--------------------------------------------------------------------
Key additions
• **Memory‑influence regularizer** – encourages the model to do better *because*
  rungs are alive (no explicit retrieval labels needed).
• **Gate‑entropy shaping** – pushes each write‑gate towards decisive 0/1 usage
  instead of noisy dithering; promotes sparse, salient memories.

Default λ weights:  λ_mem = 0.1, λ_ent = 0.02  (see Trainer).

All other API surfaces are unchanged.
"""

import random, string, math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        bridge_type: str = "mlp",  # "mlp" | "hier_ae" | "attention"
    ):
        assert bridge_type in {"mlp", "hier_ae", "attention"}, "invalid bridge_type"
        self.tag_dim = tag_dim
        self.use_tags = use_tags
        self.use_contrastive = use_contrastive
        self.use_critic = use_critic
        self.dropout_p = dropout_p
        self.bridge_type = bridge_type

# -----------------------------------------------------------------------------
#  Utility losses
# -----------------------------------------------------------------------------

def horizon_contrastive(latents: List[torch.Tensor]):
    if len(latents) < 2:
        return latents[0].new_zeros([])
    anchors = torch.cat(latents, 0)
    logits = anchors @ anchors.T * 0.1
    labels = torch.arange(len(latents), device=anchors.device)
    return F.cross_entropy(logits, labels)

class SlowTierCritic(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, z):
        return self.head(z)

class AutoEncoder(nn.Module):
    def __init__(self, in_d, lat_d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_d, lat_d * 2), nn.ReLU(), nn.Linear(lat_d * 2, lat_d))
        self.dec = nn.Sequential(nn.Linear(lat_d, lat_d * 2), nn.ReLU(), nn.Linear(lat_d * 2, in_d))
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)

# -----------------------------------------------------------------------------
#  Memory Rung
# -----------------------------------------------------------------------------

class MemoryRung(nn.Module):
    def __init__(self, name, tau, in_d, lat_d, hid, cfg: CLConfig, slots=1):
        super().__init__()
        self.name, self.tau, self.cfg = name, tau, cfg
        self.ae = AutoEncoder(in_d, lat_d)
        self.slots = slots
        tag_vec = F.one_hot(torch.tensor(hid), cfg.tag_dim).float() if cfg.use_tags else torch.zeros(cfg.tag_dim)
        self.register_buffer("tag", tag_vec, persistent=False)
        self.register_buffer("latents", torch.zeros(slots, lat_d))
        self.gate = nn.Sequential(nn.Linear(in_d + lat_d + cfg.tag_dim, 64), nn.GELU(), nn.Linear(64, 1))
        bridge_in_dim = in_d + cfg.tag_dim + lat_d * slots
        if cfg.bridge_type == "mlp":
            self.bridge = nn.Sequential(nn.Linear(bridge_in_dim, bridge_in_dim), nn.GELU(), nn.Linear(bridge_in_dim, in_d))
        elif cfg.bridge_type == "hier_ae":
            self.bridge = AutoEncoder(bridge_in_dim, bridge_in_dim // 2)
        elif cfg.bridge_type == "attention":
            self.q_proj = nn.Linear(in_d, lat_d, bias=False)
            self.k_proj = nn.Linear(bridge_in_dim, lat_d, bias=False)
        self.step = 0
        self._last_gate = torch.tensor(0.5)  # default until first forward

    # ---------------------------------------------------
    def _bridge_process(self, x, lower_lat):
        if self.cfg.bridge_type == "mlp":
            b_in = torch.cat([x, *lower_lat, self.tag], -1) if lower_lat else torch.cat([x, self.tag], -1)
            return self.bridge(b_in)
        if self.cfg.bridge_type == "hier_ae":
            b_in = torch.cat([x, *lower_lat, self.tag], -1) if lower_lat else torch.cat([x, self.tag], -1)
            return self.bridge.encode(b_in)
        if self.cfg.bridge_type == "attention" and lower_lat:
            K_raw = torch.stack(lower_lat)
            K = F.normalize(self.k_proj(K_raw), dim=-1)
            q = F.normalize(self.q_proj(x), dim=-1)
            w = (K @ q).softmax(0)
            context = (w.unsqueeze(-1) * K).sum(0)
            return torch.cat([x, context, self.tag], -1)
        return torch.cat([x, self.tag], -1)

    # ---------------------------------------------------
    def forward(self, x, lower_lat):
        self.step += 1
        if self.step % self.tau == 0:
            enriched = self._bridge_process(x, lower_lat)
            prev = self.latents[0]
            enc = self.ae.encode(enriched)
            p = torch.sigmoid(self.gate(torch.cat([enriched, prev], -1)))
            new_lat = p * enc + (1 - p) * prev
            if self.training and torch.rand(1, device=x.device) < self.cfg.dropout_p:
                new_lat.mul_(0.)
            self.latents[0] = new_lat.detach()
            self._last_gate = p.mean().detach()
        if self.tau >= 64 and self.step % (self.tau * 4) == 0:
            with torch.no_grad():
                self.latents.copy_(self.ae.encode(self.ae.decode(self.latents)))
        return torch.cat([self.tag, self.latents.view(-1)], -1)

# -----------------------------------------------------------------------------
#  ChronoLadder LM
# -----------------------------------------------------------------------------

class ChronoLadderLM(nn.Module):
    def __init__(self, cfg: CLConfig | None = None, backbone="gpt2-medium"):
        super().__init__()
        self.cfg = cfg or CLConfig()
        self.backbone = GPT2LMHeadModel.from_pretrained(backbone)
        h = self.backbone.config.n_embd
        self.rungs = nn.ModuleList([
            MemoryRung("AE1",   1,   h, 256, 0, self.cfg),
            MemoryRung("AE4",   4,   h, 512, 1, self.cfg),
            MemoryRung("AE16", 16,   h, 768, 2, self.cfg, slots=2),
            MemoryRung("AE64", 64,   h,1024, 3, self.cfg, slots=2),
            MemoryRung("AE256",256,  h,2048, 4, self.cfg),
        ])
        fused = sum(r.latents.numel() + self.cfg.tag_dim for r in self.rungs)
        self.mem_proj = nn.Sequential(nn.Linear(fused, h), nn.LayerNorm(h))
        self.critics = nn.ModuleDict({r.name: SlowTierCritic(r.latents.size(-1)) 
                                      for r in self.rungs if r.tau >= 64}) if self.cfg.use_critic else nn.ModuleDict()

    # ---- two helpers for Trainer -------------------------------------------
    def collect_gate_entropy(self):
        ent = 0.0
        for r in self.rungs:
            p = r._last_gate.clamp(1e-5, 1-1e-5)
            ent += -(p*torch.log(p) + (1-p)*torch.log(1-p))
        return ent / len(self.rungs)

    def zero_all_latents(self):
        for r in self.rungs:
            r.latents.zero_()

    # ---- standard forward ---------------------------------------------------
    def forward(self, ids, hidden):
        lower, all_lat = [], []
        for r in self.rungs:
            lat = r(hidden.detach(), lower_lat=lower.copy())
            all_lat.append(lat)
            lower.append(lat)
        mem = self.mem_proj(torch.cat(all_lat, -1))
        out = self.backbone(inputs_embeds=hidden + mem, labels=ids)
        return out.loss, all_lat

# -----------------------------------------------------------------------------
#  Trainer
# -----------------------------------------------------------------------------

class Trainer:
    def __init__(self, model, tok, device=None,
                 λ_mem=0.1, λ_ent=0.02):
        self.m = model.to(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tok = tok
        self.opt = torch.optim.AdamW(self.m.parameters(), lr=3e-5)
        self.device = next(self.m.parameters()).device
        self.λ_mem, self.λ_ent = λ_mem, λ_ent

    def _memory_influence_loss(self, ids, h, lm_loss_mem):
        # save & zero latents
        saved = [r.latents.clone() for r in self.m.rungs]
        self.m.zero_all_latents()
        with torch.no_grad():
            lm_nomem, _ = self.m(ids, h)
        # restore
        for r, buf in zip(self.m.rungs, saved):
            r.latents.copy_(buf)
        gap = (lm_nomem - lm_loss_mem).clamp(min=0)
        return gap

    def step(self, prompts: List[str]):
        ids = self.tok(prompts, return_tensors='pt', padding=True).input_ids.to(self.device)
        with torch.no_grad():
            h = self.m.backbone.transformer.wte(ids)

        # --- main forward with memory ---------------------------------------
        lm_loss, lat = self.m(ids, h)

        # --- auxiliary reconstruction ---------------------------------------
        recon = sum(F.mse_loss(r.ae.decode(r.latents), 
                               h.mean(1).expand_as(r.latents)) 
                    for r in self.m.rungs) * 0.1

        # --- InfoNCE horizon contrastive ------------------------------------
        contr = horizon_contrastive([l.view(1, -1) for l in lat]) * 0.05 \
                if self.m.cfg.use_contrastive else 0.

        # --- critic value heads ---------------------------------------------
        critic = 0.
        if self.m.cfg.use_critic:
            for r, l in zip(self.m.rungs, lat):
                if r.name in self.m.critics:
                    critic += F.mse_loss(self.m.critics[r.name](l.detach()),
                                         torch.zeros_like(self.m.critics[r
