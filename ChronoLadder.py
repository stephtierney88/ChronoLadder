from __future__ import annotations

"""
ChronoLadder – Hierarchical Memory Ladder with Bridge Options (PyTorch ≥ 2.1)
-----------------------------------------------------------------------------
Five‑rung cadence ladder over a GPT‑2‑Medium backbone, **now with three
up‑aggregation designs** you can toggle at runtime:

| Bridge | Flag value | Default | How it works | Pros | Cons |
|--------|------------|---------|--------------|------|------|
| **Shallow MLP** | `bridge_type="mlp"` | ✅ | Concat `[x, lower_latents, tag]` → 2‑layer MLP → AE encode | + 5‑line patch<br>+ Cheap, stable | − Linear mix only |
| **Hier‑AE** | `bridge_type="hier_ae"` |   | Same concat → *Bridge‑AE* (latent ½ size) → fed into slower AE | + True compression<br>+ Can add recon loss | − Extra params<br>− Needs tuning |
| **Cross‑rung Attention** | `bridge_type="attention"` |   | Query = `x`; Keys/Values = lower latents; softmax‑pool → concat with `x` → AE | + Content‑based routing | − Most compute<br>− Needs masking rules |

The **defaults remain production‑leaning**:
* horizon‑tags ✔︎, contrastive ✔︎, critic ✘
* bridge_type = "mlp"  (Design A)

Feel free to flip any flag in `CLConfig`.
"""

import random, string, math
from typing import List, Tuple
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
        self.tag_dim = tag_dim
        self.use_tags = use_tags
        self.use_contrastive = use_contrastive
        self.use_critic = use_critic
        self.dropout_p = dropout_p
        self.bridge_type = bridge_type
        assert bridge_type in {"mlp", "hier_ae", "attention"}, "invalid bridge_type"

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

# -----------------------------------------------------------------------------
#  Critic (optional)
# -----------------------------------------------------------------------------

class SlowTierCritic(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.Linear(256,1))
    def forward(self,z):
        return self.head(z)

# -----------------------------------------------------------------------------
#  Tiny AutoEncoder
# -----------------------------------------------------------------------------

class AutoEncoder(nn.Module):
    def __init__(self, in_d, lat_d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_d,lat_d*2), nn.ReLU(), nn.Linear(lat_d*2,lat_d))
        self.dec = nn.Sequential(nn.Linear(lat_d,lat_d*2), nn.ReLU(), nn.Linear(lat_d*2,in_d))
    def encode(self,x): return self.enc(x)
    def decode(self,z): return self.dec(z)

# -----------------------------------------------------------------------------
#  Memory Rung
# -----------------------------------------------------------------------------

class MemoryRung(nn.Module):
    def __init__(self,name,tau,in_d,lat_d,hid,cfg:CLConfig,slots=1):
        super().__init__(); self.name,self.tau,self.cfg=name,tau,cfg
        self.ae=AutoEncoder(in_d,lat_d); self.slots=slots
        tag = F.one_hot(torch.tensor(hid),cfg.tag_dim).float() if cfg.use_tags else torch.zeros(cfg.tag_dim)
        self.register_buffer('tag',tag,persistent=False)
        self.register_buffer('latents',torch.zeros(slots,lat_d))
        # write gate
        self.gate = nn.Sequential(nn.Linear(in_d+lat_d+cfg.tag_dim,64), nn.GELU(), nn.Linear(64,1))
        # bridge modules
        bridge_in_dim = in_d + cfg.tag_dim + lat_d*slots  # x + tag + cat(lower)
        if cfg.bridge_type=='mlp':
            self.bridge = nn.Sequential(nn.Linear(bridge_in_dim, bridge_in_dim), nn.GELU(), nn.Linear(bridge_in_dim, in_d))
        elif cfg.bridge_type=='hier_ae':
            self.bridge = AutoEncoder(bridge_in_dim, bridge_in_dim//2)
        elif cfg.bridge_type=='attention':
            self.q_proj = nn.Linear(in_d, lat_d, bias=False)
            self.k_proj = nn.Identity()
        self.step=0

    # ---------------------------------------------------
    def _bridge_process(self,x, lower_lat):
        if self.cfg.bridge_type=='mlp':
            b_in=torch.cat([x,*lower_lat,self.tag],-1) if lower_lat else torch.cat([x,self.tag],-1)
            return self.bridge(b_in)
        if self.cfg.bridge_type=='hier_ae':
            b_in=torch.cat([x,*lower_lat,self.tag],-1) if lower_lat else torch.cat([x,self.tag],-1)
            return self.bridge.encode(b_in)
        if self.cfg.bridge_type=='attention' and lower_lat:
            K=torch.stack(lower_lat)  # n x D
            q=self.q_proj(x)          # D
            attn=(K@q)/math.sqrt(K.size(-1))
            w=attn.softmax(0)
            context=(w.unsqueeze(-1)*K).sum(0)
            return torch.cat([x,context,self.tag],-1)
        return torch.cat([x,self.tag],-1)

    # ---------------------------------------------------
    def forward(self,x, lower_lat):
        self.step+=1
        if self.step%self.tau==0:
            enriched=self._bridge_process(x,lower_lat)
            prev=self.latents[0]; enc=self.ae.encode(enriched)
            p=torch.sigmoid(self.gate(torch.cat([enriched,prev],-1)))
            new_lat=p*enc+(1-p)*prev
            if self.training and torch.rand(1,device=x.device)<self.cfg.dropout_p:
                new_lat*=0
            self.latents[0]=new_lat.detach()
        if self.tau>=64 and self.step%(self.tau*4)==0:
            with torch.no_grad(): self.latents.copy_(self.ae.encode(self.ae.decode(self.latents)))
        return torch.cat([self.tag,self.latents.view(-1)],-1)

# -----------------------------------------------------------------------------
#  ChronoLadder LM
# -----------------------------------------------------------------------------

class ChronoLadderLM(nn.Module):
    def __init__(self,cfg:CLConfig|None=None,backbone='gpt2-medium'):
        super().__init__(); self.cfg=cfg or CLConfig()
        self.backbone=GPT2LMHeadModel.from_pretrained(backbone); h=self.backbone.config.n_embd
        self.rungs=nn.ModuleList([
            MemoryRung('AE1',1,h,256,0,self.cfg),
            MemoryRung('AE4',4,h,512,1,self.cfg),
            MemoryRung('AE16',16,h,768,2,self.cfg,slots=2),
            MemoryRung('AE64',64,h,1024,3,self.cfg,slots=2),
            MemoryRung('AE256',256,h,2048,4,self.cfg),
        ])
        fused=sum(r.latents.numel()+self.cfg.tag_dim for r in self.rungs)
        self.mem_proj=nn.Sequential(nn.Linear(fused,h), nn.LayerNorm(h))
        self.critics=nn.ModuleDict({r.name:SlowTierCritic(r.latents.size(-1)) for r in self.rungs if r.tau>=64}) if self.cfg.use_critic else nn.ModuleDict()

    # ---------------------------------------------
    def forward(self,ids,hidden):
        lower=[]; all_lat=[]
        for r in self.rungs:
            lat=r(hidden.detach(),lower_lat=lower.copy())
            all_lat.append(lat); lower.append(lat)
        mem=self.mem_proj(torch.cat(all_lat,-1))
        out=self.backbone(inputs_embeds=hidden+mem, labels=ids)
        return out.loss, all_lat

# -----------------------------------------------------------------------------
#  Trainer
# -----------------------------------------------------------------------------

class Trainer:
    def __init__(self,model,tok,device=None):
        self.m=model.to(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tok=tok; self.opt=torch.optim.AdamW(self.m.parameters(),lr=3e-5)
        self.device=self.m.device
    def step(self,prompts):
        ids=self.tok(prompts,return_tensors='pt',padding=True).input_ids.to(self.device)
        with torch.no_grad(): h=self.m.backbone.transformer.wte(ids)
        lm_loss,lat=self.m(ids,h)
        recon=sum(F.mse_loss(self.m.rungs[i].ae.decode(self.m.rungs[i].latents), h.mean(1).expand_as(self.m.rungs[i].latents)) for i in range(len(self.m.rungs)))*0.1
        contr=horizon_contrastive([l.view(1,-1) for l in lat])*0.05 if self.m.cfg.use_contrastive else 0
        critic=0
        if self.m.cfg.use_critic:
            for r,l in zip(self.m.rungs,lat):
                if r.name in self.m.critics:
                    critic+=F.mse_loss(self.m.critics[r.name](l.detach()), torch.zeros_like(self.m.critics[r.name](l)))*0.02
        loss=lm_loss+recon+contr+critic
        loss.backward(); nn.utils.clip_grad_norm_(self.m.parameters(),1.0)
        self.opt.step(); self.opt.zero_grad()
        return {'total':loss.item(),'task':lm_loss.item(),'recon':recon.item(),'ctr':float(contr),'critic':float(critic)}

# -----------------------------------------------------------------------------
#  Toy data + demo
# -----------------------------------------------------------------------------

def make_copy_dataset(delay=64,size=4000):
    abc=list(string.ascii_lowercase); return [f"remember {random.choice(abc)} then wait {delay} steps " + "x "*delay + "now what?" for _ in range(size)]

if __name__=='__main__':
    tok=GPT2Tokenizer.from_pretrained('gpt2-medium')
    cfg=CLConfig()               # defaults: bridge_type='mlp'
    model=ChronoLadderLM(cfg)
    trainer=Trainer(model,tok)
    data=make_copy_dataset()
    for s in range(500):
        m=trainer.step(random.sample(data,4))
        if s%50==0:
            print(f"{s} | tot {m['total']:.3f} task {m['task']:.3f} rec {m['recon']:.3f} ctr {m['ctr']:.3f} crit {m['critic']:.3f}")
