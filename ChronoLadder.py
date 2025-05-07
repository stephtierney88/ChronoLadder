"""
ChronoLadder – reference implementation (PyTorch ≥ 2.1)
-------------------------------------------------------
A minimal yet reasonably complete scaffold that wires a five‑rung cadence
memory ladder onto a vanilla transformer (GPT‑2 Medium by default).

✅ Features baked‑in
• stateful write gate on every rung (prevents memory thrash)
• slot‑dropout (forces robust routing)
• orthogonality penalty (avoids latent soup)
• periodic KL refresh for slow rungs (stops drift/collapse)
• LayerNorm + projection fuse latents back into model hidden space
• copy / delayed‑recall toy dataset + simple Trainer loop

🚫 Not production ready – it’s a hackable skeleton for research iteration.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# -----------------------------------------------------------------------------
#  Utility losses
# -----------------------------------------------------------------------------

def orthogonality_loss(latents: List[torch.Tensor]) -> torch.Tensor:
    """Cheap cosine‑similarity penalty across a list of latents.
    Returns 0 if <2 latents are provided."""
    if len(latents) < 2:
        return latents[0].new_zeros(())
    loss = 0.0
    pairs = 0
    for i in range(len(latents)):
        for j in range(i + 1, len(latents)):
            cos = F.cosine_similarity(latents[i], latents[j], dim=-1)
            loss = loss + cos.mean()
            pairs += 1
    return loss / pairs

# -----------------------------------------------------------------------------
#  AutoEncoder stub (tiny MLP, replace with Conv/ViT for vision latents)
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
    """One cadence rung – maintains its own AE and slots."""

    def __init__(
        self,
        name: str,
        cadence: int,
        in_dim: int,
        latent_dim: int,
        slots: int = 1,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.name = name
        self.tau = cadence
        self.slots = slots
        self.dropout_p = dropout_p
        self.ae = AutoEncoder(in_dim, latent_dim)
        self.gate = nn.Linear(in_dim + latent_dim, 1)
        self.register_buffer("latents", torch.zeros(slots, latent_dim))
        self.step_counter = 0

    # ---------------------------------------------------------------------
    #  Helpers
    # ---------------------------------------------------------------------

    def _stateful_write(self, x: torch.Tensor):
        prev = self.latents[0]
        enc = self.ae.encode(x)
        g = torch.sigmoid(self.gate(torch.cat([x, prev], dim=-1)))
        new_latent = g * enc + (1 - g) * prev
        # slot‑dropout during training to encourage fallback behaviour
        if self.training and torch.rand(1, device=x.device) < self.dropout_p:
            new_latent = new_latent * 0.0
        self.latents[0] = new_latent.detach()

    def _kl_refresh(self):
        # simple decode‑re‑encode to stop drift (only for slow rungs)
        decoded = self.ae.decode(self.latents)
        self.latents.copy_(self.ae.encode(decoded.detach()).detach())

    # ---------------------------------------------------------------------
    #  Public
    # ---------------------------------------------------------------------

    def should_update(self) -> bool:
        return self.step_counter % self.tau == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return latents, optionally update/write/refresh."""
        self.step_counter += 1
        if self.should_update():
            self._stateful_write(x)
        # KL refresh for slow rungs every 4×tau steps
        if self.tau >= 64 and self.step_counter % (self.tau * 4) == 0:
            self._kl_refresh()
        return self.latents.view(1, -1)  # concat slots

    # ----------------- Aux losses --------------------------------------
    def aux_losses(self, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # reconstruct latents back to input space
        recon = self.ae.decode(self.latents)
        recon_loss = F.mse_loss(recon, target.expand_as(recon))
        # orthogonality only matters if multi‑slot
        ortho_loss = (
            orthogonality_loss([self.latents]) if self.slots > 1 else recon_loss.new_zeros(())
        )
        return recon_loss, ortho_loss

# -----------------------------------------------------------------------------
#  ChronoLadder LM wrapper
# -----------------------------------------------------------------------------

class ChronoLadderLM(nn.Module):
    def __init__(self, backbone_name: str = "gpt2-medium"):
        super().__init__()
        self.backbone = GPT2LMHeadModel.from_pretrained(backbone_name)
        hidden_dim = self.backbone.config.n_embd

        self.rungs = nn.ModuleList(
            [
                MemoryRung("AE1", 1, hidden_dim, 256),
                MemoryRung("AE4", 4, hidden_dim, 512),
                MemoryRung("AE16", 16, hidden_dim, 768, slots=2),
                MemoryRung("AE64", 64, hidden_dim, 1024, slots=2),
                MemoryRung("AE256", 256, hidden_dim, 2048),
            ]
        )
        fused_dim = sum(r.latents.numel() // r.slots for r in self.rungs)
        self.mem_proj = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, input_ids: torch.Tensor, hidden_state: torch.Tensor):
        # detach hidden so AE losses stay local per step
        rung_latents = [r(hidden_state.detach()) for r in self.rungs]
        memory_concat = torch.cat(rung_latents, dim=-1)
        mem_cond = self.mem_proj(memory_concat)
        conditioned = hidden_state + mem_cond  # broadcast addition
        outputs = self.backbone(inputs_embeds=conditioned, labels=input_ids)
        return outputs.logits, {
            "task_loss": outputs.loss,
            "rung_latents": rung_latents,
        }

# -----------------------------------------------------------------------------
#  Simple synthetic copy‑task dataset
# -----------------------------------------------------------------------------

def make_copy_dataset(seq_len: int = 20, delay: int = 128, size: int = 5000):
    """Generate (prompt, target) pairs for delayed recall."""
    import random, string

    data = []
    alphabet = list(string.ascii_lowercase)
    for _ in range(size):
        token = random.choice(alphabet)
        prompt = f"remember {token} then wait {delay} steps " + "x " * delay + "now what?"
        target = token
        data.append((prompt, target))
    return data

class CopyDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Tuple[str, str]], tok: GPT2Tokenizer):
        self.tok = tok
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, target = self.samples[idx]
        ids = self.tok(prompt, return_tensors="pt").input_ids[0]
        label_ids = self.tok(target, return_tensors="pt").input_ids[0]
        return ids, label_ids

# -----------------------------------------------------------------------------
#  Trainer
# -----------------------------------------------------------------------------

class LadderTrainer:
    def __init__(
        self,
        model: ChronoLadderLM,
        tokenizer: GPT2Tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.tok = tokenizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-5)
        self.device = device

    def step(self, input_ids: torch.Tensor, labels: torch.Tensor):
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            hidden = self.model.backbone.transformer.wte(input_ids)
        logits, info = self.model(input_ids, hidden)
        task_loss = info["task_loss"]

        # aux losses
        recon_loss = ortho_loss = 0.0
        for rung in self.model.rungs:
            r_recon, r_ortho = rung.aux_losses(hidden.mean(dim=1))
            recon_loss += r_recon
            ortho_loss += r_ortho

        total = task_loss + 0.1 * recon_loss + 0.01 * ortho_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return {
            "total": total.item(),
            "task": task_loss.item(),
            "recon": recon_loss.item(),
            "ortho": ortho_loss.item(),
        }

# -----------------------------------------------------------------------------
#  Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    tok = GPT2Tokenizer.from_pretrained("gpt2-medium")
    model = ChronoLadderLM()
    trainer = LadderTrainer(model, tok)

    data = CopyDataset(make_copy_dataset(), tok)
    loader = DataLoader(data, batch_size=4, shuffle=True, collate_fn=lambda batch: (
        torch.nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True, padding_value=tok.eos_token_id),
        torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True, padding_value=-100),
    ))

    for epoch in range(3):
        for step, (inp, lbl) in enumerate(loader):
            metrics = trainer.step(inp, lbl)
            if step % 50 == 0:
                print(
                    f"ep{epoch} step{step} | total {metrics['total']:.3f} "
                    f"task {metrics['task']:.3f} recon {metrics['recon']:.3f} ortho {metrics['ortho']:.3f}"
                )
