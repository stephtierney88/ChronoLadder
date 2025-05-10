from __future__ import annotations
"""
ChronoLadder v3 – Hierarchical Memory Ladder with surprise‑triggered promotions
(Python ≥ 3.10 · PyTorch ≥ 2.1)
--------------------------------------------------------------------
Adds over v2
• Surprise‑gated up‑aggregation
• Memory‑influence regulariser (λ_mem)
• Gate‑entropy shaping (λ_ent)
• Promotion tax (λ_prom)
• Orthogonality penalty across latent slots (λ_orth)
Public API: ChronoLadderLM(cfg) and Trainer.
"""

import math, random, string
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ──────────────────────────────────  Configuration  ──────────────────────────

class CLConfig:
    def __init__(
        self,
        tag_dim: int = 32,
        use_tags: bool = True,
        use_contrastive: bool = True,
        use_critic: bool = False,
        dropout_p: float = 0.1,
        bridge_type: str = "mlp",          # "mlp" | "hier_ae" | "attention"
        promote_thresh_start: float = 1.0, # bigger → harder to promote
        promote_cooldown_frac: float = 0.5,
    ):
        assert bridge_type in {"mlp", "hier_ae", "attention"}
        self.tag_dim = tag_dim
        self.use_tags = use_tags
        self.use_contrastive = use_contrastive
        self.use_critic = use_critic
        self.dropout_p = dropout_p
        self.bridge_type = bridge_type
        self.promote_thresh_start = promote_thresh_start
        self.promote_cooldown_frac = promote_cooldown_frac

# ────────────────────────────────  Helper modules  ───────────────────────────

def horizon_contrastive(latents: List[torch.Tensor]):
    if len(latents) < 2:
        return latents[0].new_zeros([])
    a = torch.cat(latents, 0)
    logits = a @ a.T * 0.1
    labels = torch.arange(len(latents), device=a.device)
    return F.cross_entropy(logits, labels)

class SlowTierCritic(nn.Module):
    def __init__(self, d): super().__init__(); self.net = nn.Sequential(
        nn.Linear(d,256), nn.ReLU(), nn.Linear(256,1))
    def forward(self,z): return self.net(z)

class AutoEncoder(nn.Module):
    def __init__(self,in_d:int,lat_d:int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_d,lat_d*2),nn.ReLU(),
                                 nn.Linear(lat_d*2,lat_d))
        self.dec = nn.Sequential(nn.Linear(lat_d,lat_d*2),nn.ReLU(),
                                 nn.Linear(lat_d*2,in_d))
    def encode(self,x): return self.enc(x)
    def decode(self,z): return self.dec(z)

# ──────────────────────────────────  Memory Rung  ────────────────────────────

class MemoryRung(nn.Module):
    def __init__(self,name:str,tau:int,in_d:int,lat_d:int,hid:int,cfg:CLConfig,*,slots:int=1):
        super().__init__()
        self.name,self.tau,self.cfg = name,tau,cfg
        self.ae = AutoEncoder(in_d,lat_d); self.slots = slots
        tag = F.one_hot(torch.tensor(hid),cfg.tag_dim).float() if cfg.use_tags else torch.zeros(cfg.tag_dim)
        self.register_buffer("tag",tag,persistent=False)
        self.register_buffer("latents",torch.zeros(slots,lat_d))
        self.gate = nn.Sequential(nn.Linear(in_d+lat_d+cfg.tag_dim,64),nn.GELU(),nn.Linear(64,1))
        bridge_in = in_d+cfg.tag_dim+lat_d*slots
        if cfg.bridge_type=="mlp":
            self.bridge = nn.Sequential(nn.Linear(bridge_in,bridge_in),nn.GELU(),nn.Linear(bridge_in,in_d))
        elif cfg.bridge_type=="hier_ae":
            self.bridge = AutoEncoder(bridge_in,bridge_in//2)
        else:
            self.q_proj = nn.Linear(in_d,lat_d,bias=False)
            self.k_proj = nn.Linear(bridge_in,lat_d,bias=False)
        self.promote_thresh=cfg.promote_thresh_start; self.cooldown=0
        self._promote=False; self.step=0; self._last_gate=torch.tensor(0.5)

    def _bridge_process(self,x,lower):
        if self.cfg.bridge_type=="attention" and lower:
            K = F.normalize(self.k_proj(torch.stack(lower)),dim=-1)
            q = F.normalize(self.q_proj(x),dim=-1)
            ctx=(K@q).softmax(0).unsqueeze(-1)*K
            return torch.cat([x,ctx.sum(0),self.tag],-1)
        comb = torch.cat([x,*lower,self.tag],-1) if lower else torch.cat([x,self.tag],-1)
        return self.bridge.encode(comb) if self.cfg.bridge_type=="hier_ae" else self.bridge(comb)

    def forward(self,x,lower,*,promote_lower=False):
        self.step+=1
        write_now = (promote_lower and self.cooldown==0) or (self.step%self.tau==0)
        if write_now:
            enriched = self._bridge_process(x,lower)
            prev = self.latents[0]; enc = self.ae.encode(enriched)
            p = torch.sigmoid(self.gate(torch.cat([enriched,prev],-1)))
            new_lat = p*enc + (1-p)*prev
            if self.training and torch.rand(())<self.cfg.dropout_p: new_lat.zero_()
            self.latents[0]=new_lat.detach(); self._last_gate=p.mean().detach()
            surprise = F.mse_loss(enriched.detach(),self.ae.decode(enc).detach()).item()
            self._promote = surprise>self.promote_thresh and self._last_gate>0.5 and self.cooldown==0
            self.promote_thresh = 0.95*self.promote_thresh + 0.05*surprise
            if self._promote: self.cooldown=max(1,int(self.tau*self.cfg.promote_cooldown_frac))
        else: self._promote=False
        if self.cooldown: self.cooldown-=1
        if self.tau>=64 and self.step%(self.tau*4)==0:
            with torch.no_grad(): self.latents.copy_(self.ae.encode(self.ae.decode(self.latents)))
        return torch.cat([self.tag,self.latents.view(-1)],-1)

# ────────────────────────────────  ChronoLadder LM  ──────────────────────────

class ChronoLadderLM(nn.Module):
    def __init__(self,cfg:CLConfig|None=None,backbone="gpt2-medium"):
        super().__init__(); self.cfg=cfg or CLConfig()
        self.backbone = GPT2LMHeadModel.from_pretrained(backbone)
        h=self.backbone.config.n_embd
        self.rungs = nn.ModuleList([
            MemoryRung("AE1",1,h,256,0,self.cfg),
            MemoryRung("AE4",4,h,512,1,self.cfg),
            MemoryRung("AE16",16,h,768,2,self.cfg,slots=2),
            MemoryRung("AE64",64,h,1024,3,self.cfg,slots=2),
            MemoryRung("AE256",256,h,2048,4,self.cfg),
        ])
        fused=sum(r.latents.numel()+self.cfg.tag_dim for r in self.rungs)
        self.mem_proj = nn.Sequential(nn.Linear(fused,h),nn.LayerNorm(h))
        self.critics = nn.ModuleDict({r.name:SlowTierCritic(r.latents.size(-1))
                                      for r in self.rungs if r.tau>=64}) if self.cfg.use_critic else nn.ModuleDict()

    def collect_gate_entropy(self):
        return sum(-(p:=r._last_gate.clamp(1e-5,1-1e-5))*p.log()-(1-p)*(1-p).log()
                   for r in self.rungs)/len(self.rungs)

    def zero_all_latents(self):
        for r in self.rungs: r.latents.zero_()

    def forward(self,ids,hidden):
        lower=[]; all_lat=[]; promote=False
        for r in self.rungs:
            lat=r(hidden.detach(),lower,promote_lower=promote)
            promote=r._promote; lower.append(lat); all_lat.append(lat)
        mem=self.mem_proj(torch.cat(all_lat,-1))
        out=self.backbone(inputs_embeds=hidden+mem,labels=ids)
        return out.loss,all_lat

# ─────────────────────────────────  Trainer  ─────────────────────────────────

class Trainer:
    def __init__(self,model:ChronoLadderLM,tok:GPT2Tokenizer,*,device=None,
                 λ_mem=0.1,λ_ent=0.02,λ_prom=0.01,λ_orth=0.01):
        self.m=model.to(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tok=tok; self.opt=torch.optim.AdamW(self.m.parameters(),lr=3e-5)
        self.device=next(self.m.parameters()).device
        self.λ_mem, self.λ_ent, self.λ_prom, self.λ_orth = λ_mem,λ_ent,λ_prom,λ_orth

    def _mem_gap(self,ids,h,lm_with):
        saved=[r.latents.clone() for r in self.m.rungs]; self.m.zero_all_latents()
        with torch.no_grad(): lm_no,_=self.m(ids,h)
        for r,b in zip(self.m.rungs,saved): r.latents.copy_(b)
        return (lm_no-lm_with).clamp(min=0)

    def _orth_loss(self):
        loss=0.
        for r in self.m.rungs:
            if r.latents.size(0)<2: continue
            z=F.normalize(r.latents,dim=-1); gram=z@z.T
            loss += (gram - torch.eye(z.size(0),device=z.device)).pow(2).mean()
        return loss

    def step(self,prompts:List[str]):
        ids=self.tok(prompts,return_tensors='pt',padding=True).input_ids.to(self.device)
        with torch.no_grad(): h=self.m.backbone.transformer.wte(ids)
        lm_loss,lat=self.m(ids,h)
        recon=sum(F.mse_loss(r.ae.decode(r.latents),h.mean(1).expand_as(r.latents))
                  for r in self.m.rungs)*0.1
        contr=horizon_contrastive([l.view(1,-1) for l in lat])*0.05 \
              if self.m.cfg.use_contrastive else torch.tensor(0.,device=self.device)
        ent=self.m.collect_gate_entropy()*self.λ_ent
        prom_tax=sum(int(r._promote) for r in self.m.rungs)*self.λ_prom
        mem_gap=self._mem_gap(ids,h,lm_loss)*self.λ_mem; mem_loss=-mem_gap
        orth=self._orth_loss()*self.λ_orth
        critic=torch.tensor(0.,device=self.device)
        if self.m.cfg.use_critic:
            for r,l in zip(self.m.rungs,lat):
                if r.name in self.m.critics:
                    critic += F.mse_loss(self.m.critics[r.name](l.detach()),
                                         torch.zeros_like(self.m.critics[r.name](l)))*0.02
        total=lm_loss+recon+contr+ent+prom_tax+mem_loss+orth+critic
        total.backward(); nn.utils.clip_grad_norm_(self.m.parameters(),1.0)
        self.opt.step(); self.opt.zero_grad()
        return dict(total=total.item(),task=lm_loss.item(),recon=recon.item(),
                    nce=contr.item(),entropy=ent.item(),mem_bonus=mem_gap.item(),
                    prom_tax=prom_tax,orth=orth.item())

# ────────────────────────────────  Toy demo  ────────────────────────────────

def make_copy_dataset(delay=64,size=4000):
    abc=list(string.ascii_lowercase)
    return [f"remember {random.choice(abc)} then wait {delay} steps "+"x "*delay+"now what?"
            for _ in range(size)]

if __name__=="__main__":
    tok=GPT2Tokenizer.from_pretrained("gpt2-medium")
    model=ChronoLadderLM(CLConfig())
    trainer=Trainer(model,tok)
    data=make_copy_dataset()
    for step in range(300):
        stats=trainer.step(random.sample(data,4))
        if step%25==0:
            print(f"{step:03d} | total {stats['total']:.3f}  task {stats['task']:.3f}  "
                  f"gap {stats['mem_bonus']:.3f}  orth {stats['orth']:.3f}  prom {stats['prom_tax']}")
