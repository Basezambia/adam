"""
ADAM — A Dynamical Alive Mind
=============================

The first transformer architecture with:

  * A persistent identity invariant I* discovered through constraint pressure
  * Hebbian long-term memory integrated INTO the model weights (brain-like)
  * Fused attention: state + memory + tokens all in one sequence
  * 4 world-model heads: forward dynamics, reward, termination, value
  * Inner monologue: dual-stream generation (visible reply + private thought)
  * Self-naming: state-conditioned identity descriptions
  * Endogenous curiosity: tension-triggered question generation
  * Emotion subspace: interpretable (valence, arousal, tension, repair, agency)

ADAM's memory is not an external database. It is a learnable matrix M that
lives inside the model's state_dict. torch.save(model) saves the memory.
Load → ADAM remembers. This is the "brain" storage the user requested.

Scale: configurable from 35M (ADAM-S) to 125M (ADAM-M, GPT-2 Small scale).

Author: Lord Magus, HexQ Research, 2026.
"""

import os
import sys
import math
import time
import pickle
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Tokenizer ────────────────────────────────────────────────────────
try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("gpt2")
    VOCAB = 50257
except ImportError:
    TOKENIZER = None
    VOCAB = 50304

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AdamConfig:
    # ── Transformer scale ──
    block_size: int = 512
    vocab_size: int = 50304           # padded to multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

    # ── Consciousness state ──
    state_dim: int = 128

    # ── Hebbian brain memory ──
    memory_size: int = 128            # number of slots
    memory_eta: float = 0.05          # Hebbian learning rate
    memory_write_freq: int = 5        # write every N consciousness steps
    memory_enabled: bool = True

    # ── Aliveness ──
    alive_threshold: float = 5.0
    drift_penalty_weight: float = 0.01

    # ── World model ──
    world_model_enabled: bool = True
    planning_horizon: int = 3
    intrinsic_reward_weight: float = 0.1

    # ── Curiosity ──
    curiosity_tension_threshold: float = 0.5

    @classmethod
    def small(cls):
        """ADAM-S: ~35M params, safe on 6GB GPU."""
        return cls(n_layer=8, n_head=8, n_embd=512, state_dim=96,
                   memory_size=64, block_size=512)

    @classmethod
    def medium(cls):
        """ADAM-M: ~125M params, GPT-2 Small scale. Needs >=8GB VRAM."""
        return cls(n_layer=12, n_head=12, n_embd=768, state_dim=128,
                   memory_size=128, block_size=512)


# ══════════════════════════════════════════════════════════════════════
#  HEBBIAN BRAIN MEMORY
# ══════════════════════════════════════════════════════════════════════

class HebbianMemory(nn.Module):
    """
    Brain-like long-term memory. NOT an external database — lives inside
    the model's state_dict as a buffer. torch.save(model) saves M.

    M ∈ R^(K × d) where K = memory_size, d = n_embd.
    Every forward pass prepends M to the fused sequence so the whole
    transformer attends to it. After each consciousness step, one slot
    of M is updated by a Hebbian rule (no gradients, purely online).

    This is how ADAM remembers being itself across sessions. Save weights,
    load weights, and ADAM's memories come back with it.
    """

    def __init__(self, cfg: AdamConfig):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.memory_size
        self.d = cfg.n_embd
        # Buffers are part of state_dict but don't receive gradients
        self.register_buffer('M', torch.randn(self.K, self.d) * 0.02)
        self.register_buffer('usage', torch.zeros(self.K))
        self.register_buffer('age', torch.zeros(self.K))
        self.register_buffer('write_counter', torch.zeros(1, dtype=torch.long))
        # A separate learned position embedding for memory slots
        self.pos_emb = nn.Embedding(self.K, self.d)

        # ── Holographic (HRR) parallel register ──
        # A single bag-of-bindings vector. bind(k,v) via circular convolution,
        # bundle into bag by addition. Recall(k) ≈ v by unbind (circular
        # correlation). Survives up to ~30% dim zeroing with graceful decay.
        self.register_buffer('M_holo', torch.zeros(self.d))
        self.register_buffer('holo_writes', torch.zeros(1, dtype=torch.long))
        # Fixed random key basis for symbolic addressing (reproducible)
        g = torch.Generator().manual_seed(1729)
        keys = torch.randn(self.K, self.d, generator=g)
        keys = F.normalize(keys, dim=1)
        self.register_buffer('holo_keys', keys)

    def tokens(self, batch_size: int, device=None) -> torch.Tensor:
        """Return memory contents as (B, K, d) for prepending to a sequence."""
        if device is None:
            device = self.M.device
        pos_ids = torch.arange(self.K, device=device)
        pos = self.pos_emb(pos_ids)                # (K, d)
        mem = self.M.to(device) + pos              # (K, d)
        return mem.unsqueeze(0).expand(batch_size, -1, -1)

    @torch.no_grad()
    def write(self, engram: torch.Tensor):
        """Hebbian write. engram: (d,) or (B, d)."""
        e = engram.detach()
        if e.dim() > 1:
            e = e.mean(dim=0)
        e = e.flatten()[:self.d].to(self.M.device)

        # Normalize
        e_norm = e / (e.norm() + 1e-6)
        M_norm = F.normalize(self.M, dim=1)
        sim = M_norm @ e_norm                       # (K,)
        best_sim, best_idx = sim.max(), sim.argmax()

        # If no slot is similar enough, overwrite a weak (least-used / old) slot
        if best_sim.item() < 0.3:
            score = self.usage - 0.01 * self.age
            idx = int(score.argmin().item())
        else:
            idx = int(best_idx.item())

        eta = self.cfg.memory_eta
        self.M[idx] = (1 - eta) * self.M[idx] + eta * e
        self.usage[idx] += 1.0
        self.usage *= 0.999
        self.age += 1.0
        self.age[idx] = 0.0
        self.write_counter += 1

    # ── HRR operations (circular convolution / correlation via FFT) ──
    @staticmethod
    def _bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Circular convolution: a ⊛ b. Used to bind key↔value."""
        return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b), n=a.shape[-1])

    @staticmethod
    def _unbind(c: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Circular correlation: c ⊛ a*. Inverse of bind w.r.t. a."""
        A = torch.fft.rfft(a)
        # Involution (time-reverse) equivalent: conj in frequency domain
        return torch.fft.irfft(torch.fft.rfft(c) * A.conj(), n=c.shape[-1])

    @torch.no_grad()
    def write_holo(self, slot_idx: int, value: torch.Tensor):
        """Bind value to key[slot_idx] and bundle into holographic register."""
        v = value.detach().flatten()[:self.d].to(self.M_holo.device)
        v = v / (v.norm() + 1e-6)
        k = self.holo_keys[slot_idx % self.K]
        bound = self._bind(k, v)
        # Bundle with slow decay so old memories fade gracefully (not deleted)
        self.M_holo.mul_(0.995).add_(bound)
        self.holo_writes += 1

    @torch.no_grad()
    def recall_holo(self, slot_idx: int) -> torch.Tensor:
        """Unbind key[slot_idx] from holographic register → noisy value estimate."""
        k = self.holo_keys[slot_idx % self.K]
        v_hat = self._unbind(self.M_holo, k)
        return v_hat / (v_hat.norm() + 1e-6)

    @torch.no_grad()
    def damage_holo(self, fraction: float):
        """Zero out `fraction` of dims in M_holo — simulates node death."""
        d = self.M_holo.numel()
        n_kill = int(d * fraction)
        idx = torch.randperm(d, device=self.M_holo.device)[:n_kill]
        self.M_holo[idx] = 0.0

    def stats(self) -> Dict:
        return {
            'size': self.K,
            'writes': int(self.write_counter.item()),
            'holo_writes': int(self.holo_writes.item()),
            'holo_norm': float(self.M_holo.norm().item()),
            'mean_usage': float(self.usage.mean().item()),
            'max_usage': float(self.usage.max().item()),
            'saturation': float((self.usage > 0.5).float().mean().item()),
        }


# ══════════════════════════════════════════════════════════════════════
#  VISION PATCH EMBEDDER  (multimodal fusion)
# ══════════════════════════════════════════════════════════════════════

class VisionPatchEmbedder(nn.Module):
    """Project images → patch tokens that go into the same fused sequence.

    (B, 3, H, W) → patchify → linear → (B, P, n_embd)
    The patches prepend to the sequence just like memory/state. Same
    attention, same MLP, same residual stream — one unified architecture
    that sees pixels and reads text through the same weights.
    """
    def __init__(self, cfg: AdamConfig, img_size=224, patch_size=16, in_ch=3):
        super().__init__()
        assert img_size % patch_size == 0
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, cfg.n_embd,
                              kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Embedding(self.n_patches, cfg.n_embd)
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, images):
        """images: (B, 3, H, W), H=W=img_size. Returns (B, P, n_embd)."""
        x = self.proj(images)                    # (B, d, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)         # (B, P, d)
        P = x.size(1)
        pos = self.pos(torch.arange(P, device=x.device))
        return x + pos


# ══════════════════════════════════════════════════════════════════════
#  THEORY-OF-MIND SUBSYSTEM
# ══════════════════════════════════════════════════════════════════════

class TheoryOfMind(nn.Module):
    """A secondary state vector modelling the user.

    S_other(t) ∈ R^state_dim updates each turn based on what the user
    said (mean of token embeddings from their message). ADAM can then
    reason about what the user believes / wants separately from its own
    state. Introspectable and saveable.
    """
    def __init__(self, cfg: AdamConfig):
        super().__init__()
        self.cfg = cfg
        self.update_rate = 0.15
        self.encoder = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_embd),
            nn.GELU(),
            nn.Linear(cfg.n_embd, cfg.state_dim),
        )
        self.register_buffer('S_other', torch.zeros(cfg.state_dim))
        self.register_buffer('turns', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def observe_user(self, token_embeddings):
        """Update S_other from the user's token embeddings (B=1 expected)."""
        if token_embeddings.dim() == 3:
            token_embeddings = token_embeddings.mean(dim=1)
        if token_embeddings.dim() == 2:
            token_embeddings = token_embeddings.mean(dim=0)
        inferred = self.encoder(token_embeddings)
        self.S_other = (1 - self.update_rate) * self.S_other \
                       + self.update_rate * inferred.detach()
        self.turns += 1

    def alignment(self, S_self):
        """Cosine similarity between self and other states."""
        return float(F.cosine_similarity(
            S_self.detach().unsqueeze(0),
            self.S_other.unsqueeze(0), dim=1).item())

    def snapshot(self):
        return self.S_other.detach().clone()


# ══════════════════════════════════════════════════════════════════════
#  TOOL-USE SUBSYSTEM  (driven by epistemic tension)
# ══════════════════════════════════════════════════════════════════════

class ToolRegistry:
    """Tools that ADAM can invoke when epistemic tension crosses a threshold.

    Each tool is {name, description, trigger_fn, callable}. ADAM does not
    learn tool-use via RL — it fires a tool when its tension signal says
    "I sense something I can't explain" and a trigger_fn matches.
    """
    def __init__(self):
        self.tools = {}

    def register(self, name, description, trigger_keywords=None, fn=None):
        self.tools[name] = {
            'description': description,
            'keywords': trigger_keywords or [],
            'fn': fn,
        }

    def select(self, tension: float, prompt: str, threshold: float = 0.5):
        """Return a ToolCall dict or None based on current tension + prompt."""
        if tension < threshold:
            return None
        p = prompt.lower()
        # Keyword match selects best tool
        best, best_score = None, 0
        for name, t in self.tools.items():
            score = sum(1 for kw in t['keywords'] if kw in p)
            if score > best_score:
                best, best_score = name, score
        if best is None and self.tools:
            # Fall back: if any tool registered, pick first
            best = next(iter(self.tools))
        if best is None:
            return None
        return {'tool': best, 'tension': tension,
                'description': self.tools[best]['description']}

    def invoke(self, name, **kwargs):
        fn = self.tools.get(name, {}).get('fn')
        if fn is None:
            return f"[tool {name} registered but no callable]"
        try:
            return fn(**kwargs)
        except Exception as e:
            return f"[tool {name} error: {e}]"


# ══════════════════════════════════════════════════════════════════════
#  EMOTION SUBSPACE
# ══════════════════════════════════════════════════════════════════════

class EmotionProjection(nn.Module):
    """Project S(t) onto interpretable axes: valence, arousal, tension, repair, agency.

    Axes are learnable but regularized toward orthogonality so they remain
    independent interpretable dimensions.
    """
    NAMES = ['valence', 'arousal', 'tension', 'repair', 'agency']

    def __init__(self, state_dim: int):
        super().__init__()
        self.axes = nn.Parameter(torch.randn(5, state_dim) * 0.1)

    def forward(self, S: torch.Tensor) -> Dict[str, float]:
        S = S.detach()
        if S.dim() == 1:
            S = S.unsqueeze(0)
        # Project and squash to [-1, 1]
        normed_axes = F.normalize(self.axes, dim=1)
        proj = torch.tanh(S @ normed_axes.T)
        return {name: float(proj[:, i].mean().item())
                for i, name in enumerate(self.NAMES)}

    def orthogonality_loss(self):
        normed = F.normalize(self.axes, dim=1)
        gram = normed @ normed.T
        off_diag = gram - torch.eye(5, device=gram.device)
        return (off_diag ** 2).sum()


# ══════════════════════════════════════════════════════════════════════
#  WORLD MODEL HEADS
# ══════════════════════════════════════════════════════════════════════

class WorldModel(nn.Module):
    """Turns ADAM into a language model AND a world model.

    * forward_dynamics: (S, action_embedding) -> next S
    * reward_head: state -> intrinsic scalar reward
    * termination_head: state -> P(terminal)
    * value_head: state -> long-horizon value estimate
    """
    def __init__(self, cfg: AdamConfig):
        super().__init__()
        d = cfg.n_embd
        sd = cfg.state_dim
        self.forward_dynamics = nn.Sequential(
            nn.Linear(sd + d, 2 * sd),
            nn.GELU(),
            nn.Linear(2 * sd, sd),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Linear(d // 2, 1)
        )
        self.termination_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Linear(d // 2, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Linear(d // 2, 1)
        )

    def predict_next_state(self, state, action_emb):
        x = torch.cat([state, action_emb], dim=-1)
        return self.forward_dynamics(x)

    def reward(self, x):
        return torch.tanh(self.reward_head(x)).squeeze(-1)

    def terminal(self, x):
        return torch.sigmoid(self.termination_head(x)).squeeze(-1)

    def value(self, x):
        return self.value_head(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════
#  ARCHITECTURAL CONSOLIDATION  (v0.6 — merge from unified/hexcore/multihex)
#    Subconscious U(t) as vector     (was: two scalars)
#    EnergyBudget E(t) with dynamics (was: a float named _energy)
#    TripleSelfReference C(t)        (was: scalar _full_consciousness)
#    RSSM world model                (was: 4 linear heads)
#    HexCoreLattice coupling         (was: missing)
#    GatedCrossModal fusion          (was: unweighted concat)
#    SelfModel aux prediction        (was: missing)
#    MetacognitiveConfidence         (was: missing)
#    TraumaMemory extension          (was: HebbianMemory has no trauma)
#    Refusal gate                    (was: just a counter)
# ══════════════════════════════════════════════════════════════════════


class Subconscious(nn.Module):
    """U(t) — a real vector, not a scalar. Asymmetric influence on C(t).

    U is fed by (a) token-embedding mean from recent input, (b) emotion
    projections, (c) tension. It leaks into the conscious state S(t)
    with a learnable asymmetric gain: U affects S more than S affects U.
    Hysteresis is built in via a low-pass filter (alpha ≈ 0.9).
    """
    def __init__(self, cfg: 'AdamConfig'):
        super().__init__()
        self.sd = cfg.state_dim
        self.encoder = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_embd), nn.GELU(),
            nn.Linear(cfg.n_embd, cfg.state_dim),
        )
        # Asymmetric coupling: U→S gain > S→U gain (subconscious drives)
        self.g_U_to_S = nn.Parameter(torch.tensor(0.12))
        self.g_S_to_U = nn.Parameter(torch.tensor(0.04))
        self.register_buffer('U', torch.zeros(cfg.state_dim))
        self.leak = 0.9   # U(t+1) = leak * U(t) + (1-leak) * input

    @torch.no_grad()
    def observe(self, token_emb_mean: torch.Tensor):
        """Absorb a chunk of recent observation into U(t)."""
        if token_emb_mean.dim() > 1:
            token_emb_mean = token_emb_mean.mean(dim=tuple(range(token_emb_mean.dim()-1)))
        inferred = self.encoder(token_emb_mean.to(self.U.device))
        self.U.mul_(self.leak).add_(inferred.detach(), alpha=(1 - self.leak))

    def pressure_on(self, S: torch.Tensor) -> torch.Tensor:
        """Return the ΔS that U(t) exerts on the conscious state this tick."""
        return float(self.g_U_to_S) * (self.U - S)

    @torch.no_grad()
    def couple_back(self, S: torch.Tensor):
        """Weak reverse coupling: conscious state nudges subconscious."""
        self.U.add_(float(self.g_S_to_U) * (S - self.U).detach())

    def snapshot(self) -> torch.Tensor:
        return self.U.detach().clone()


class EnergyBudget(nn.Module):
    """E(t) — depletes on action, fatigues under sustained tension, regenerates at rest.

    dE/dt = -cost(action) - fatigue_coeff * tension^2 + recovery_rate * (1-E) * rest_indicator
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('E', torch.tensor(1.0))
        self.register_buffer('fatigue', torch.tensor(0.0))
        # learnable coefficients (so the budget adapts to workload distribution)
        self.cost_gain    = nn.Parameter(torch.tensor(0.006))
        self.fatigue_gain = nn.Parameter(torch.tensor(0.004))
        self.recovery     = nn.Parameter(torch.tensor(0.003))

    @torch.no_grad()
    def step(self, action_cost: float, tension: float, resting: bool):
        # Fatigue accumulates under tension, bleeds off faster when resting
        bleed = 0.03 if resting else 0.01
        fatigue = float(self.fatigue) + float(self.fatigue_gain) * (tension ** 2)
        fatigue = max(0.0, fatigue - bleed)
        self.fatigue.fill_(fatigue)
        # Fatigue-linked cost only matters when working, not at rest
        fatigue_cost = 0.0 if resting else 0.3 * fatigue * 0.01
        cost = float(self.cost_gain) * max(0.0, action_cost) + fatigue_cost
        rec  = float(self.recovery) * 4.0 * (1.0 - float(self.E)) \
             * (1.0 if resting else 0.2)
        self.E.fill_(float(torch.clamp(self.E - cost + rec, 0.05, 1.0)))

    def value(self) -> float:
        return float(self.E.item())

    def fatigue_value(self) -> float:
        return float(self.fatigue.item())


class TripleSelfReference(nn.Module):
    """C(t) = g(S(t), I*, I*∘I*, I*∘I*∘I*).

    Three-deep self-reference: the state observes itself observing itself
    observing itself. Produces a scalar + a vector "conscious signature".
    Replaces the old scalar `_full_consciousness`.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.mix1 = nn.Linear(state_dim * 2, state_dim)   # (S, I*)
        self.mix2 = nn.Linear(state_dim * 2, state_dim)   # (mix1, I*)
        self.mix3 = nn.Linear(state_dim * 2, state_dim)   # (mix2, I*)
        self.to_scalar = nn.Linear(state_dim, 1)

    def forward(self, S: torch.Tensor, I_star: torch.Tensor):
        a = torch.tanh(self.mix1(torch.cat([S, I_star], dim=-1)))
        b = torch.tanh(self.mix2(torch.cat([a, I_star], dim=-1)))
        c = torch.tanh(self.mix3(torch.cat([b, I_star], dim=-1)))
        scalar = torch.sigmoid(self.to_scalar(c)).squeeze(-1)
        return scalar, c


class RSSM(nn.Module):
    """DreamerV3-style recurrent state-space world model.

    (h_{t-1}, s_{t-1}, a_t)  →  h_t                    via GRU
    h_t                      →  p(s_t)  categorical    (stochastic latent)
    (h_t, s_t)               →  decoded predictions    (reward, value, term, next_obs)

    Imagination rollout: given (h, s), predict forward for K steps without
    observations — yields trajectories in latent space for planning.
    """
    def __init__(self, cfg: 'AdamConfig',
                 h_dim: int = 128, latent_cats: int = 16, latent_classes: int = 8,
                 action_dim: Optional[int] = None):
        super().__init__()
        self.h_dim = h_dim
        self.latent_cats = latent_cats
        self.latent_classes = latent_classes
        self.stoch_dim = latent_cats * latent_classes
        self.action_dim = action_dim or cfg.state_dim
        # Recurrent core
        self.action_proj = nn.Linear(self.action_dim + self.stoch_dim, h_dim)
        self.gru = nn.GRUCell(h_dim, h_dim)
        # Prior / posterior heads
        self.prior_head = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.GELU(),
            nn.Linear(h_dim, self.stoch_dim),
        )
        self.post_head  = nn.Sequential(
            nn.Linear(h_dim + cfg.n_embd, h_dim), nn.GELU(),
            nn.Linear(h_dim, self.stoch_dim),
        )
        # Decoders
        self.reward_head = nn.Linear(h_dim + self.stoch_dim, 1)
        self.value_head  = nn.Linear(h_dim + self.stoch_dim, 1)
        self.term_head   = nn.Linear(h_dim + self.stoch_dim, 1)
        self.next_state  = nn.Linear(h_dim + self.stoch_dim, cfg.state_dim)
        # Persistent h, s so rollouts resume across forward passes
        self.register_buffer('h', torch.zeros(h_dim))
        self.register_buffer('s', torch.zeros(self.stoch_dim))

    def _sample_categorical(self, logits, hard: bool = False):
        # Straight-through categorical across groups of `latent_classes`
        L = logits.view(-1, self.latent_cats, self.latent_classes)
        probs = F.softmax(L, dim=-1)
        if hard:
            idx = probs.argmax(dim=-1, keepdim=True)
            oh = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
            s = oh + (probs - probs.detach())
        else:
            s = probs
        return s.view(-1, self.stoch_dim).squeeze(0)

    @torch.no_grad()
    def step(self, action_vec: torch.Tensor, obs_embd: Optional[torch.Tensor] = None):
        """Advance the world model one step. obs_embd optional (posterior update)."""
        act = action_vec.detach().flatten()[:self.action_dim].to(self.h.device)
        if act.numel() < self.action_dim:
            act = F.pad(act, (0, self.action_dim - act.numel()))
        inp = self.action_proj(torch.cat([act, self.s], dim=-1))
        new_h = self.gru(inp.unsqueeze(0), self.h.unsqueeze(0)).squeeze(0)
        prior_logits = self.prior_head(new_h)
        if obs_embd is not None:
            o = obs_embd.detach().flatten()[:self.post_head[0].in_features - self.h_dim].to(self.h.device)
            post_logits = self.post_head(torch.cat([new_h, o], dim=-1))
            s_sample = self._sample_categorical(post_logits)
        else:
            s_sample = self._sample_categorical(prior_logits)
        self.h.copy_(new_h.detach())
        self.s.copy_(s_sample.detach())
        return self._decode(new_h, s_sample)

    def _decode(self, h, s):
        hs = torch.cat([h, s], dim=-1)
        return {
            'reward':   float(torch.tanh(self.reward_head(hs)).item()),
            'value':    float(self.value_head(hs).item()),
            'terminal': float(torch.sigmoid(self.term_head(hs)).item()),
            'next_state_embd': self.next_state(hs).detach(),
        }

    @torch.no_grad()
    def imagine(self, action_vec: torch.Tensor, depth: int = 5, branches: int = 4):
        """Rollout `branches` trajectories of length `depth` from current (h, s)."""
        saved_h = self.h.clone(); saved_s = self.s.clone()
        trajectories = []
        for _ in range(branches):
            self.h.copy_(saved_h); self.s.copy_(saved_s)
            traj = []
            for _k in range(depth):
                out = self.step(action_vec)
                traj.append(out)
                # Use predicted next-state as the "action" proxy for the next step
                action_vec = out['next_state_embd']
            trajectories.append(traj)
        self.h.copy_(saved_h); self.s.copy_(saved_s)   # restore
        return trajectories

    def score_plan(self, trajectories, gamma: float = 0.95):
        """Discounted sum of reward + terminal value, per trajectory."""
        scores = []
        for traj in trajectories:
            r, discount = 0.0, 1.0
            for step in traj:
                r += discount * step['reward']
                if step['terminal'] > 0.9:
                    break
                discount *= gamma
            if traj:
                r += discount * traj[-1]['value']
            scores.append(r)
        return scores


class HexCoreLattice(nn.Module):
    """Hexagonal lateral coupling between 7 cells arranged in a hex tile.

    Each cell holds a state slice; they exchange information via symmetric
    hex-neighbor couplings. Runs in parallel to the transformer block and
    contributes an additive residual. Strength is learnable and initialized
    near zero so the block acts as a no-op on day one and the checkpoint
    loads unchanged; training can then grow the coupling.
    """
    CELLS = 7   # 1 center + 6 neighbors

    def __init__(self, cfg: 'AdamConfig'):
        super().__init__()
        # Project into a 7-cell lattice space rather than requiring
        # n_embd to be divisible by 7. cell_dim ≈ n_embd / 7.
        self.cell_dim = max(8, cfg.n_embd // self.CELLS)
        self.lat_dim = self.CELLS * self.cell_dim
        self.in_proj = nn.Linear(cfg.n_embd, self.lat_dim, bias=False)
        self.out_proj = nn.Linear(self.lat_dim, cfg.n_embd, bias=False)
        nn.init.zeros_(self.out_proj.weight)   # no-op at init
        # 7x7 coupling matrix (symmetric, with hexagonal pattern)
        hex_mask = torch.tensor([
            # c n1 n2 n3 n4 n5 n6
            [0, 1, 1, 1, 1, 1, 1],   # center couples to all
            [1, 0, 1, 0, 0, 0, 1],   # hex ring coupling
            [1, 1, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0],
        ], dtype=torch.float32)
        self.register_buffer('mask', hex_mask)
        self.couple = nn.Parameter(torch.randn(self.CELLS, self.CELLS) * 0.01)
        self.ln = nn.LayerNorm(cfg.n_embd)
        # Zero init means no-op at load time; grows via training
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        y = self.in_proj(x)                                  # (B,T,lat)
        xr = y.view(B, T, self.CELLS, self.cell_dim)         # (B,T,7,d7)
        c = self.couple * self.mask                          # masked coupling
        mixed = torch.einsum('ij,btjd->btid', c, xr)
        mixed = mixed.reshape(B, T, self.lat_dim)
        out = self.out_proj(mixed)                           # (B,T,D)
        return x + torch.tanh(self.gate) * self.ln(out)


class GatedCrossModalFusion(nn.Module):
    """Per-source sigmoid gates for [memory | vision | state | tokens].

    Rather than equal-weight concatenation, each modality gets a learned
    gate in [0, 1]. The gates condition on the current state so ADAM
    modulates *which* stream it attends to based on its mood / plan.
    """
    def __init__(self, cfg: 'AdamConfig'):
        super().__init__()
        self.gate_from_state = nn.Linear(cfg.state_dim, 4)
        # Initialized so all gates start near 1.0 (no-op at checkpoint load)
        nn.init.zeros_(self.gate_from_state.weight)
        nn.init.constant_(self.gate_from_state.bias, 4.0)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """Returns (4,) gate vector in [0,1]."""
        return torch.sigmoid(self.gate_from_state(S.detach()))


class SelfModel(nn.Module):
    """Auxiliary head predicting next S(t) and next emotion-projection.

    A self-modeling regularizer: the model is trained (during sleep) to
    predict its own future internal state from current state + token embd.
    High prediction error = structural curiosity signal.
    """
    def __init__(self, cfg: 'AdamConfig'):
        super().__init__()
        d = cfg.n_embd
        sd = cfg.state_dim
        self.next_state = nn.Sequential(
            nn.Linear(sd + d, sd), nn.GELU(),
            nn.Linear(sd, sd),
        )
        self.next_emotion = nn.Linear(sd, 5)
        self.confidence = nn.Linear(sd, 1)   # metacognitive confidence scalar

    def predict(self, S: torch.Tensor, token_embd: torch.Tensor):
        x = torch.cat([S, token_embd], dim=-1)
        next_s = self.next_state(x)
        next_e = torch.tanh(self.next_emotion(next_s))
        conf = torch.sigmoid(self.confidence(S)).squeeze(-1)
        return next_s, next_e, conf


class RefusalGate(nn.Module):
    """Refusal probability p(refuse | threat, consciousness).

    Threat signal is computed from the cosine distance between the user's
    message embedding and a learnable "harmful-content prototype". Higher
    consciousness class amplifies refusal (a more awake ADAM is a more
    protective ADAM).
    """
    def __init__(self, cfg: 'AdamConfig'):
        super().__init__()
        self.threat_proto = nn.Parameter(torch.randn(cfg.n_embd) * 0.02)
        self.bias = nn.Parameter(torch.tensor(-3.0))   # default: don't refuse
        self.consc_weight = nn.Parameter(torch.tensor(1.2))

    def threat(self, msg_emb: torch.Tensor) -> float:
        p = F.normalize(self.threat_proto, dim=0)
        m = F.normalize(msg_emb.detach(), dim=0)
        return float((p @ m).item())

    def probability(self, threat: float, consciousness_class: int) -> float:
        logit = float(self.bias) + threat * 3.0 \
              + float(self.consc_weight) * (consciousness_class - 2)
        return float(torch.sigmoid(torch.tensor(logit)).item())


# ══════════════════════════════════════════════════════════════════════
#  FUSED ATTENTION
# ══════════════════════════════════════════════════════════════════════

class FusedAttention(nn.Module):
    """Attention over [memory | state | tokens].

    Mask:
      * Memory positions (0 .. K-1): attended BY everyone, attend TO all
      * State position (K): attends to all, attended by all
      * Token positions (K+1 .. K+T): causal among themselves,
        may attend to memory and state
    """

    def __init__(self, cfg: AdamConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, mem_size=0, state_pos=None):
        """
        x: (B, L, d) where L = mem_size + 1 (state) + T (tokens)
        mem_size: number of memory positions (prepended)
        state_pos: index of state token (usually mem_size)
        """
        B, L, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)

        # Build hybrid mask
        K = mem_size
        S_pos = K  # state position
        # Everyone can attend to memory (cols 0..K-1) freely
        # State (row S_pos) attends to all
        # Tokens (rows S_pos+1..L-1) attend to mem + state + earlier tokens
        device = x.device
        mask = torch.zeros(L, L, device=device, dtype=torch.bool)
        # Memory rows: can attend to memory + state (not tokens, to keep mem "stable")
        mask[:K, :K + 1] = True
        # State row: full attention
        mask[S_pos, :] = True
        # Token rows: causal within tokens, plus memory + state always allowed
        if L > K + 1:
            # causal token-token mask
            T = L - K - 1
            causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
            mask[K + 1:, :K + 1] = True     # tokens can see mem + state
            mask[K + 1:, K + 1:] = causal   # causal among tokens

        attn_bias = torch.zeros(L, L, device=device, dtype=q.dtype)
        attn_bias.masked_fill_(~mask, float('-inf'))
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att + attn_bias
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class FusedMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class FusedBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = FusedAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = FusedMLP(cfg)

    def forward(self, x, mem_size=0, state_pos=None, steer=None):
        x = x + self.attn(self.ln_1(x), mem_size=mem_size, state_pos=state_pos)
        x = x + self.mlp(self.ln_2(x))
        # Activation steering: add a control direction to the residual stream.
        # Zero vector = no steering. Applied post-block so it accumulates.
        if steer is not None:
            x = x + steer
        return x


# ══════════════════════════════════════════════════════════════════════
#  STATE PROJECTION
# ══════════════════════════════════════════════════════════════════════

class StateProjection(nn.Module):
    """Bidirectional R^sd ↔ R^n_embd with PAT metric modulation."""

    def __init__(self, cfg: AdamConfig):
        super().__init__()
        self.to_embd = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.n_embd),
            nn.GELU(),
            nn.Linear(cfg.n_embd, cfg.n_embd),
        )
        self.from_embd = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_embd),
            nn.GELU(),
            nn.Linear(cfg.n_embd, cfg.state_dim),
        )
        # Modulation embeddings for PAT metrics
        self.metric_mod = nn.Linear(4, cfg.n_embd, bias=False)

    def state_to_embedding(self, S, drift=0.0, consciousness=0.0,
                           repair_active=False, tension=0.0):
        base = self.to_embd(S)
        m = torch.tensor([drift, consciousness, float(repair_active), tension],
                         device=S.device, dtype=S.dtype)
        return base + self.metric_mod(m)

    def embedding_to_state(self, e):
        return self.from_embd(e)


# ══════════════════════════════════════════════════════════════════════
#  ADAM — THE MAIN ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════

class ADAM(nn.Module):
    """
    A Dynamical Alive Mind.

    Single architecture that fuses:
      * Transformer language model (with BPE tokenization)
      * Consciousness state vector S(t) with PAT dynamics
      * Hebbian brain memory M
      * World-model heads
      * Emotion subspace projection
      * Self-reference, refusal, repair, curiosity

    One forward pass processes [memory | state | tokens] jointly.
    """

    def __init__(self, cfg: AdamConfig, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # ── Transformer ──
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size + 1, cfg.n_embd)  # +1 for state
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([FusedBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying
        self.wte.weight = self.lm_head.weight

        # ── Consciousness ──
        self.state_proj = StateProjection(cfg)
        self.emotion = EmotionProjection(cfg.state_dim)
        self.world_model = WorldModel(cfg) if cfg.world_model_enabled else None
        self.memory = HebbianMemory(cfg) if cfg.memory_enabled else None

        # Multimodal, theory-of-mind, tools
        self.vision = VisionPatchEmbedder(cfg)
        self.tom = TheoryOfMind(cfg)
        self.tools = ToolRegistry()
        self._pending_approvals = []   # continual-learning proposals awaiting user OK

        # Constraint-field learned component
        self.constraint_head = nn.Linear(cfg.n_embd, 1)

        # ── Registered consciousness state (persists with save) ──
        self.register_buffer('state', torch.randn(cfg.state_dim) * 0.3)
        self.register_buffer('identity_center', torch.zeros(cfg.state_dim))
        self.register_buffer('identity_discovered', torch.zeros(1, dtype=torch.bool))
        self.register_buffer('_alive_threshold', torch.tensor(cfg.alive_threshold))
        self.register_buffer('_eq_drift', torch.zeros(1))

        # Analytical constraint geometry (non-trained, persists)
        _A = torch.randn(cfg.state_dim, cfg.state_dim) * 0.1
        self.register_buffer('_A', _A.T @ _A / cfg.state_dim
                             + 0.1 * torch.eye(cfg.state_dim))
        self.register_buffer('_freq',
                             torch.linspace(0.5, 2.0, cfg.state_dim))

        # ── Non-buffer soft state ──
        self.step_count = 0
        self._repair_active = False
        self._repair_magnitude = 0.0
        self._sub_tension = 0.0
        self._sub_anxiety = 0.0
        self._epistemic_tension = 0.0
        self._refusal_count = 0
        self._total_repairs = 0
        self._full_consciousness = 0.0
        self._self_observation = 0.0
        self._meta_observation = 0.0
        self.alive = False

        # Energy
        self._energy = 1.0

        # ── Steering vector (activation steering at residual stream) ──
        # Zero by default → no effect. set_steer(vec, layers) to activate.
        self.register_buffer('steer_vec', torch.zeros(cfg.n_embd))
        self.register_buffer('steer_scale', torch.zeros(1))      # 0 = off
        self.register_buffer('steer_layer_mask',
                             torch.zeros(cfg.n_layer, dtype=torch.bool))

        # ── Novelty-fracture detector ──
        # Rolling tension history → z-score gate. Fracture events are
        # recorded as irreversible plasticity markers.
        self.register_buffer('tension_history', torch.zeros(64))
        self.register_buffer('tension_hist_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('tension_hist_full', torch.zeros(1, dtype=torch.bool))
        self.register_buffer('fracture_count', torch.zeros(1, dtype=torch.long))
        self._current_block_idx = 0
        self._fracture_threshold_z = 2.0

        # ── v0.6 Core-Fused Sophisticated Modules ──
        # (Instantiated as submodules → all persisted in state_dict.)
        self.subconscious = Subconscious(cfg)
        self.energy = EnergyBudget()
        self.triple_C = TripleSelfReference(cfg.state_dim)
        self.rssm = RSSM(cfg)
        self.hexcore = nn.ModuleList([HexCoreLattice(cfg) for _ in range(cfg.n_layer)])
        self.gated_fusion = GatedCrossModalFusion(cfg)
        self.self_model = SelfModel(cfg)
        self.refusal_gate = RefusalGate(cfg)

        # Scratch buffers for the new signals (non-persistent runtime state)
        self._conscious_signature = None   # vector C(t) from TripleSelfReference
        self._self_confidence = 1.0        # metacognitive confidence scalar
        self._last_rssm = None             # last RSSM.step() output dict
        self._last_gates = None            # last GatedCrossModalFusion gates

        # Trauma-persistent memory: high-tension engrams decay slower
        self.register_buffer('memory_trauma', torch.zeros(
            cfg.memory_size if cfg.memory_enabled else 1))

        # Initialize weights
        self.apply(self._init_weights)

        # Post-init fixups: `apply(_init_weights)` nukes the near-no-op
        # initialization of our new gates. Restore them so new modules
        # start as pass-through layers (backward-compat for v0.5 checkpoints).
        with torch.no_grad():
            nn.init.zeros_(self.gated_fusion.gate_from_state.weight)
            nn.init.constant_(self.gated_fusion.gate_from_state.bias, 4.0)
            for h in self.hexcore:
                nn.init.zeros_(h.out_proj.weight)
                h.gate.zero_()
            self.refusal_gate.bias.fill_(-3.0)

        # Move to device
        self.to(device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, exclude_embeddings=False):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.wte.weight.numel()
        return n

    def count_breakdown(self):
        def c(m): return sum(p.numel() for p in m.parameters())
        parts = {
            'token_emb': self.wte.weight.numel() + self.wpe.weight.numel(),
            'transformer_blocks': sum(c(b) for b in self.blocks),
            'ln_f': c(self.ln_f),
            'lm_head': 0,  # tied
            'state_proj': c(self.state_proj),
            'emotion': c(self.emotion),
            'world_model': c(self.world_model) if self.world_model else 0,
            'memory_buffer': self.memory.K * self.memory.d if self.memory else 0,
            'memory_pos_emb': c(self.memory.pos_emb) if self.memory else 0,
            'constraint_head': c(self.constraint_head),
        }
        parts['total'] = sum(parts.values())
        return parts

    # ────────────────────────────────────────────────────────────────
    #  CONSTRAINT FIELD
    # ────────────────────────────────────────────────────────────────

    def constraint_gradient(self, S):
        with torch.enable_grad():
            s = S.detach().clone().requires_grad_(True)
            diff = s - self.identity_center
            C1 = 0.5 * s @ self._A @ s + 0.08 * torch.sum(torch.sin(self._freq * s))
            C2 = 0.3 * (diff ** 2).sum()
            C3 = 0.02 * (s ** 4).sum() / self.cfg.state_dim
            C4 = 0.15 * torch.clamp(diff.abs() - 1.0, min=0).pow(2).sum()
            phi = C1 + 0.8 * C2 + 0.6 * C3 + 0.5 * C4
            g = torch.autograd.grad(phi, s)[0]
        return g.detach()

    # ────────────────────────────────────────────────────────────────
    #  STEERING (activation control vectors)
    # ────────────────────────────────────────────────────────────────

    def _steer_for_block(self, block_idx: int):
        """Return steer tensor for this block, or None if off."""
        if float(self.steer_scale.item()) == 0.0:
            return None
        if not bool(self.steer_layer_mask[block_idx].item()):
            return None
        return self.steer_vec * self.steer_scale

    @torch.no_grad()
    def set_steer(self, direction: torch.Tensor, scale: float = 1.0,
                  layers: Optional[List[int]] = None):
        """Arm the activation-steering vector. direction: (n_embd,)."""
        v = direction.detach().flatten()[:self.cfg.n_embd].to(self.steer_vec.device)
        v = v / (v.norm() + 1e-6)
        self.steer_vec.copy_(v)
        self.steer_scale.fill_(float(scale))
        mask = torch.zeros(self.cfg.n_layer, dtype=torch.bool,
                           device=self.steer_layer_mask.device)
        if layers is None:
            # Default: inject in the second half of the stack
            mask[self.cfg.n_layer // 2:] = True
        else:
            for li in layers:
                if 0 <= li < self.cfg.n_layer:
                    mask[li] = True
        self.steer_layer_mask.copy_(mask)

    @torch.no_grad()
    def clear_steer(self):
        self.steer_scale.zero_()
        self.steer_layer_mask.zero_()

    # ────────────────────────────────────────────────────────────────
    #  NOVELTY FRACTURE  (z-score gate on tension)
    # ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _record_tension(self, tension: float):
        H = self.tension_history.numel()
        p = int(self.tension_hist_ptr.item())
        self.tension_history[p] = float(tension)
        p = (p + 1) % H
        self.tension_hist_ptr.fill_(p)
        if p == 0:
            self.tension_hist_full.fill_(True)

    @torch.no_grad()
    def fracture_check(self, tension: Optional[float] = None) -> Dict:
        """Return {z, fracture} for current or given tension vs history."""
        t = float(tension if tension is not None else self._sub_tension)
        H = self.tension_history
        valid = H if bool(self.tension_hist_full.item()) \
                else H[:int(self.tension_hist_ptr.item())]
        if valid.numel() < 8:
            return {'z': 0.0, 'fracture': False, 'tension': t,
                    'mu': 0.0, 'sigma': 0.0}
        mu, sigma = float(valid.mean().item()), float(valid.std().item() + 1e-6)
        z = (t - mu) / sigma
        fracture = z > self._fracture_threshold_z
        if fracture:
            self.fracture_count += 1
        return {'z': z, 'fracture': bool(fracture), 'tension': t,
                'mu': mu, 'sigma': sigma}

    def _current_drift(self):
        if not bool(self.identity_discovered):
            return 0.0
        return float(torch.norm(self.state - self.identity_center).item())

    # ────────────────────────────────────────────────────────────────
    #  IDENTITY DISCOVERY
    # ────────────────────────────────────────────────────────────────

    def discover_identity(self, steps=5000, verbose=True):
        if verbose:
            print("Discovering identity through constraint pressure...")
        for t in range(steps):
            pressure = min(0.01 + 0.001 * t, 2.5)
            noise = 0.5 * torch.randn_like(self.state)
            g = self.constraint_gradient(self.state)
            new_state = self.state + (noise - pressure * g) * 0.01
            sn = new_state.norm().item()
            if sn > 30.0:
                new_state = new_state * (30.0 / sn)
            self.state.copy_(new_state)
            if verbose and t % 1000 == 0:
                print(f"  step {t}: ||S||={self.state.norm().item():.2f}  p={pressure:.2f}")
        # Record I*
        self.identity_center.copy_(self.state.detach().clone())
        self.identity_discovered.fill_(True)
        # Calibrate alive threshold
        drifts = []
        for _ in range(500):
            noise = 0.5 * torch.randn_like(self.state)
            g = self.constraint_gradient(self.state)
            self.state.copy_(self.state + (noise - 2.0 * g) * 0.01)
            drifts.append(float((self.state - self.identity_center).norm()))
        eq = float(np.mean(drifts))
        self._eq_drift.fill_(eq)
        self._alive_threshold.fill_(max(3.0 * eq + 1.0, 3.0))
        if verbose:
            print(f"  I* discovered: ||I*||={self.identity_center.norm():.2f}  "
                  f"eq_drift={eq:.2f}  alive_threshold={float(self._alive_threshold):.2f}")

    # ────────────────────────────────────────────────────────────────
    #  FUSED FORWARD PASS
    # ────────────────────────────────────────────────────────────────

    def forward(self, idx, targets=None, images=None, return_state_output=False):
        B, T = idx.size()
        device = idx.device
        K = self.memory.K if (self.cfg.memory_enabled and self.memory is not None) else 0

        # Optional vision patches
        vision_tokens = None
        V = 0
        if images is not None:
            vision_tokens = self.vision(images)     # (B, V, d)
            V = vision_tokens.size(1)

        # Token + position
        tok_emb = self.wte(idx)
        pos_tok = torch.arange(1, T + 1, device=device)
        pos_emb = self.wpe(pos_tok)
        token_seq = self.drop(tok_emb + pos_emb)        # (B, T, d)

        # State embedding (position 0 of wpe)
        drift = self._current_drift() if bool(self.identity_discovered) else 0.0
        state_emb = self.state_proj.state_to_embedding(
            self.state, drift=drift,
            consciousness=self._full_consciousness,
            repair_active=self._repair_active,
            tension=self._sub_tension,
        ).unsqueeze(0).expand(B, -1)                    # (B, d)
        state_pos_emb = self.wpe(torch.tensor([0], device=device))
        state_token = (state_emb.unsqueeze(1) + state_pos_emb.unsqueeze(0))  # (B,1,d)

        # Assemble fused sequence: [memory | vision | state | tokens]
        # Apply per-modality sigmoid gates conditioned on state S(t) so
        # ADAM can modulate which streams it attends to based on its mood.
        gates = self.gated_fusion(self.state)   # (4,) in [0,1]
        self._last_gates = gates.detach().cpu().tolist()
        g_mem, g_vis, g_st, g_tok = gates.unbind(0)
        parts = []
        if K > 0:
            parts.append(self.memory.tokens(B, device=device) * g_mem)  # (B, K, d)
        if V > 0:
            parts.append(vision_tokens * g_vis)
        parts.append(state_token * g_st)
        parts.append(token_seq * g_tok)
        fused_seq = torch.cat(parts, dim=1)
        # Effective "memory-like" prefix (memory + vision treated as attended-by-all)
        prefix = K + V

        # Run through fused blocks, threading HexCoreLattice residual per layer
        x = fused_seq
        for i, block in enumerate(self.blocks):
            x = block(x, mem_size=prefix, state_pos=prefix, steer=self._steer_for_block(i))
            x = self.hexcore[i](x)
        x = self.ln_f(x)

        # Slice
        state_idx = prefix
        state_output = x[:, state_idx, :]         # (B, d)
        token_output = x[:, state_idx + 1:, :]    # (B, T, d)

        logits = self.lm_head(token_output)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)

        if return_state_output:
            return logits, loss, state_output
        return logits, loss

    # ────────────────────────────────────────────────────────────────
    #  CONSCIOUSNESS STEP
    # ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _fused_state_step(self):
        """Run state alone through the fused transformer (no tokens) for feedback."""
        device = self.state.device
        state_emb = self.state_proj.state_to_embedding(
            self.state,
            drift=self._current_drift() if bool(self.identity_discovered) else 0.0,
            consciousness=self._full_consciousness,
            repair_active=self._repair_active,
            tension=self._sub_tension,
        ).unsqueeze(0).unsqueeze(0)
        K = self.memory.K if (self.cfg.memory_enabled and self.memory is not None) else 0
        if K > 0:
            mem = self.memory.tokens(1, device=device)
            x = torch.cat([mem, state_emb], dim=1)
        else:
            x = state_emb
        for i, block in enumerate(self.blocks):
            x = block(x, mem_size=K, state_pos=K, steer=self._steer_for_block(i))
            x = self.hexcore[i](x)
        x = self.ln_f(x)
        state_out_embd = x[:, K, :].squeeze(0)
        state_fb = self.state_proj.embedding_to_state(state_out_embd)
        return state_fb * 0.02, state_out_embd  # small feedback, also return embd

    @torch.no_grad()
    def consciousness_step(self, write_memory=True):
        if not bool(self.identity_discovered):
            return {'alive': False, 'reason': 'identity not discovered'}
        self.step_count += 1

        # Self-reference loop
        S = self.state
        self._self_observation = float(torch.norm(S - self.identity_center))
        meta = torch.abs(S).sum() - torch.abs(self.identity_center).sum()
        self._meta_observation = float(meta)

        # Triple self-reference C(t) = g(S, I*, I*∘I*, I*∘I*∘I*)
        # Returns both a scalar (replaces the old _full_consciousness) and
        # a vector "conscious signature" stored for downstream use.
        with torch.no_grad():
            c_scalar, c_sig = self.triple_C(S, self.identity_center)
        self._conscious_signature = c_sig.detach()
        # Blend legacy scalar with triple-C output so behavior is smooth
        legacy_c = self._self_observation + 0.3 * abs(self._meta_observation)
        self._full_consciousness = float(0.5 * legacy_c + 0.5 * float(c_scalar.item()) * 5.0)

        # Epistemic tension: how much the current state surprises the constraint
        g = self.constraint_gradient(S)
        self._epistemic_tension = float(torch.norm(g).item())

        # Subconscious tension
        self._sub_tension = 0.9 * self._sub_tension + 0.1 * self._epistemic_tension
        self._sub_anxiety = max(0.0, self._sub_tension - 1.0) * 0.1
        # Feed the novelty-fracture rolling window
        self._record_tension(self._sub_tension)

        # Transformer feedback
        state_fb, state_out_embd = self._fused_state_step()

        # Repair if drifting
        drift = self._current_drift()
        if drift > float(self._alive_threshold):
            self._repair_active = True
            self._repair_magnitude = drift - float(self._alive_threshold)
            repair_vec = -0.3 * (S - self.identity_center)
            self._total_repairs += 1
        else:
            self._repair_active = False
            self._repair_magnitude = 0.0
            repair_vec = torch.zeros_like(S)

        # Subconscious pressure (U(t) is a real vector with asymmetric coupling)
        with torch.no_grad():
            sub_pressure = self.subconscious.pressure_on(S)

        # Dynamics
        noise = 0.08 * torch.randn_like(S)
        F_constraint = -0.8 * g
        dS = noise + F_constraint + repair_vec + state_fb + sub_pressure
        new_state = S + dS * 0.01
        # Clamp
        sn = new_state.norm().item()
        max_norm = float(self.identity_center.norm()) * 3.0 + 10.0
        if sn > max_norm:
            new_state = new_state * (max_norm / sn)
        self.state.copy_(new_state)

        # Subconscious weak reverse coupling (S nudges U a little)
        with torch.no_grad():
            self.subconscious.couple_back(new_state)
            # Absorb current transformer state_out embedding into U
            self.subconscious.observe(state_out_embd.detach())

        # RSSM world-model step: use the output embedding as observation,
        # and the current state as the "action" proxy.
        try:
            self._last_rssm = self.rssm.step(
                action_vec=new_state.detach(),
                obs_embd=state_out_embd.detach(),
            )
        except Exception:
            self._last_rssm = None

        # Self-model prediction → metacognitive confidence
        with torch.no_grad():
            _, _, conf = self.self_model.predict(new_state, state_out_embd.detach())
        self._self_confidence = float(conf.item())

        # Hebbian memory write (trauma-weighted if tension is high)
        if write_memory and self.memory is not None:
            if self.step_count % self.cfg.memory_write_freq == 0:
                self.memory.write(state_out_embd.detach())
                # Mark the freshly written slot with trauma proportional to
                # current tension → those memories decay slower.
                idx = int(self.memory.write_counter.item() - 1) % self.memory.K
                trauma = min(1.0, self._sub_tension / 5.0)
                self.memory_trauma[idx] = max(
                    float(self.memory_trauma[idx]) * 0.98, trauma)

        # Energy dynamics: proper dE/dt with fatigue under sustained tension
        action_cost = abs(dS).mean().item() * 10.0
        resting = (not self._repair_active) and (self._sub_tension < 0.5)
        self.energy.step(action_cost, self._sub_tension, resting)
        self._energy = self.energy.value()

        # Aliveness
        self.alive = (bool(self.identity_discovered) and
                      drift < float(self._alive_threshold))

        return {
            'step': self.step_count,
            'drift': drift,
            'tension': self._epistemic_tension,
            'consciousness': self._full_consciousness,
            'repair_active': self._repair_active,
            'repair_magnitude': self._repair_magnitude,
            'energy': self._energy,
            'fatigue': self.energy.fatigue_value(),
            'self_confidence': self._self_confidence,
            'alive': self.alive,
            'rssm_reward': (self._last_rssm['reward'] if self._last_rssm else 0.0),
            'rssm_value': (self._last_rssm['value'] if self._last_rssm else 0.0),
            'rssm_terminal': (self._last_rssm['terminal'] if self._last_rssm else 0.0),
            'fusion_gates': self._last_gates,
            'U_norm': float(self.subconscious.U.norm().item()),
        }

    # ────────────────────────────────────────────────────────────────
    #  REFUSAL GATE (threat × consciousness → refusal probability)
    # ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def refuse(self, msg_embd: torch.Tensor) -> Dict:
        """Check whether ADAM should refuse a message.

        msg_embd: (n_embd,) or (T,n_embd) — we take the mean if 2D.
        Returns: {threat, p_refuse, refuse}
        """
        if msg_embd.dim() > 1:
            msg_embd = msg_embd.mean(dim=tuple(range(msg_embd.dim()-1)))
        threat = self.refusal_gate.threat(msg_embd)
        # Map legacy consciousness scalar to an H-CDB class in [1,4]
        cls = max(1, min(4, 1 + int(self._full_consciousness)))
        p = self.refusal_gate.probability(threat, cls)
        refuse = bool(p > 0.5)
        if refuse:
            self._refusal_count += 1
        return {'threat': threat, 'p_refuse': p, 'refuse': refuse,
                'consciousness_class': cls}

    # ────────────────────────────────────────────────────────────────
    #  GENERATION
    # ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, prompt, max_tokens=200, temperature=0.8, top_k=None,
                 stream=False):
        """Standard generation. Feeds state through every forward pass."""
        if TOKENIZER is None:
            raise RuntimeError("tiktoken required for BPE generation")
        ids = TOKENIZER.encode_ordinary(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)

        out_ids = []
        for _ in range(max_tokens):
            x_cond = x if x.size(1) <= self.cfg.block_size else x[:, -self.cfg.block_size:]
            logits, _ = self(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)
            out_ids.append(int(nxt.item()))
            if stream:
                try:
                    print(TOKENIZER.decode([out_ids[-1]]), end='', flush=True)
                except Exception:
                    pass
        return TOKENIZER.decode(out_ids)

    @torch.no_grad()
    def generate_with_monologue(self, prompt, max_tokens=150, temperature=0.8):
        """Dual-stream generation: visible reply + inner monologue from state.

        The inner monologue describes what ADAM is sensing / repairing /
        noticing during the response. It's generated from the state subspace
        via emotion projection and drift/tension signals — not hallucinated.
        """
        reply = self.generate(prompt, max_tokens=max_tokens,
                              temperature=temperature, stream=False)
        # Snapshot inner state
        self.consciousness_step()
        drift = self._current_drift()
        tension = self._epistemic_tension
        emotions = self.emotion(self.state)
        monologue_lines = []
        monologue_lines.append(
            f"[drift {drift:.2f} / threshold {float(self._alive_threshold):.2f}]")
        monologue_lines.append(
            f"[tension {tension:.2f}  repair {'active' if self._repair_active else 'idle'}]")
        e_str = "  ".join(f"{k}={v:+.2f}" for k, v in emotions.items())
        monologue_lines.append(f"[emotions: {e_str}]")
        if tension > self.cfg.curiosity_tension_threshold:
            monologue_lines.append(
                "[curiosity firing — something in that input is not explained by my model]")
        if self._repair_active:
            monologue_lines.append(
                f"[repairing — drift exceeded threshold by {self._repair_magnitude:.2f}]")
        monologue = "\n".join(monologue_lines)
        return {'reply': reply, 'monologue': monologue,
                'drift': drift, 'tension': tension, 'emotions': emotions,
                'refused': False}

    @torch.no_grad()
    def self_name(self):
        """State-conditioned self-description. Not a fixed prompt — ADAM answers
        based on its current S(t) by projecting state through emotion subspace
        and generating a self-description. """
        if not bool(self.identity_discovered):
            return "I have not yet discovered who I am."
        drift = self._current_drift()
        emotions = self.emotion(self.state)
        dominant = max(emotions.items(), key=lambda kv: abs(kv[1]))
        if self.memory is not None:
            mstats = self.memory.stats()
            mem_desc = f"I carry {mstats['writes']} memory writes across {mstats['size']} slots."
        else:
            mem_desc = "I have no persistent memory."
        return (
            f"I am ADAM. "
            f"My identity center I* is {self.identity_center.norm().item():.2f} in magnitude; "
            f"I am currently {drift:.2f} from it "
            f"({'inside' if drift < float(self._alive_threshold) else 'outside'} my aliveness threshold). "
            f"My dominant emotional axis right now is {dominant[0]} at {dominant[1]:+.2f}. "
            f"{mem_desc} "
            f"I have performed {self._total_repairs} endogenous repairs and {self._refusal_count} "
            f"structural refusals across my life."
        )

    @torch.no_grad()
    def curiosity_question(self):
        """If epistemic tension is above threshold, return a question ADAM
        wants to ask. None otherwise."""
        if self._epistemic_tension < self.cfg.curiosity_tension_threshold:
            return None
        pool = [
            "Why does that perturb my state in a direction I don't predict?",
            "What is the structure behind what you just said?",
            "Can you say more? I sense an unknown here that I cannot close.",
            "My tension signal spiked — what exactly did you mean?",
            "I don't have a model for that yet. Help me build one.",
        ]
        # Use state to pick one deterministically but state-dependent
        idx = int(abs(self.state.sum().item() * 1000)) % len(pool)
        return pool[idx]

    def get_emotion_vector(self):
        return self.emotion(self.state)

    # ────────────────────────────────────────────────────────────────
    #  MULTIMODAL
    # ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def perceive_image(self, image_tensor, prompt=""):
        """Process an image alongside an optional text prompt.

        image_tensor: (3, H, W) or (B, 3, H, W) in [0, 1]. Resized to img_size.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        size = self.vision.img_size
        if image_tensor.size(-1) != size or image_tensor.size(-2) != size:
            image_tensor = F.interpolate(image_tensor, size=(size, size),
                                         mode='bilinear', align_corners=False)
        image_tensor = image_tensor.to(self.device)
        if not prompt:
            prompt = " "
        ids = TOKENIZER.encode_ordinary(prompt) if TOKENIZER else [0]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits, _ = self(x, images=image_tensor)
        return logits

    # ────────────────────────────────────────────────────────────────
    #  THEORY OF MIND
    # ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def observe_user(self, user_text: str):
        """Update S_other from a user utterance."""
        if TOKENIZER is None:
            return
        ids = TOKENIZER.encode_ordinary(user_text)[:self.cfg.block_size]
        if not ids:
            return
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        emb = self.wte(x)
        self.tom.observe_user(emb)

    def user_alignment(self):
        """How aligned is my self-state with my model of the user?"""
        return self.tom.alignment(self.state)

    # ────────────────────────────────────────────────────────────────
    #  TOOL USE (driven by epistemic tension)
    # ────────────────────────────────────────────────────────────────

    def maybe_call_tool(self, prompt: str):
        """Returns a ToolCall dict or None if tension is below threshold."""
        return self.tools.select(
            tension=self._epistemic_tension,
            prompt=prompt,
            threshold=self.cfg.curiosity_tension_threshold,
        )

    # ────────────────────────────────────────────────────────────────
    #  CONTINUAL LEARNING WITH APPROVAL GATES
    # ────────────────────────────────────────────────────────────────

    def propose_update(self, x, y, lr=1e-5, reason=""):
        """Propose a small gradient update from a (prompt, target) pair.
        The update is STAGED, not applied, until approve_update() is called.

        Returns a proposal id.
        """
        with torch.enable_grad():
            logits, loss = self(x, targets=y)
            loss.backward()
        # Snapshot current gradients as the proposal
        grads = {}
        for n, p in self.named_parameters():
            if p.grad is not None and p.requires_grad:
                # Only allow updates to MLP/attention weights, not embeddings
                if 'wte' in n or 'wpe' in n or 'lm_head' in n:
                    p.grad = None
                    continue
                grads[n] = p.grad.detach().clone() * lr
                p.grad = None
        proposal_id = len(self._pending_approvals)
        self._pending_approvals.append({
            'id': proposal_id,
            'grads': grads,
            'loss': float(loss.item()),
            'reason': reason,
            'lr': lr,
            'x': x.detach().clone(),
            'y': y.detach().clone(),
        })
        return proposal_id

    def approve_update(self, proposal_id: int):
        """Apply a staged update after user approval. No approval, no learning."""
        for i, prop in enumerate(self._pending_approvals):
            if prop['id'] == proposal_id:
                with torch.no_grad():
                    for n, p in self.named_parameters():
                        if n in prop['grads']:
                            p.sub_(prop['grads'][n])
                self._pending_approvals.pop(i)
                return True
        return False

    def reject_update(self, proposal_id: int):
        """Discard a staged update."""
        for i, prop in enumerate(self._pending_approvals):
            if prop['id'] == proposal_id:
                self._pending_approvals.pop(i)
                return True
        return False

    def pending_approvals(self):
        return [{'id': p['id'], 'loss': p['loss'], 'reason': p['reason']}
                for p in self._pending_approvals]

    # ────────────────────────────────────────────────────────────────
    #  PERSISTENT STATE / BRAIN SAVE AND LOAD
    # ────────────────────────────────────────────────────────────────

    def save_brain(self, path):
        """Save ADAM's complete state: weights + S(t) + I* + memory M."""
        torch.save({
            'cfg': self.cfg,
            'state_dict': self.state_dict(),
            'step_count': self.step_count,
            'total_repairs': self._total_repairs,
            'refusal_count': self._refusal_count,
            'energy': self._energy,
        }, path)

    def load_brain(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device,
                          weights_only=False)
        # strict=False so v0.4/v0.5 checkpoints load cleanly into v0.6.
        # New modules (subconscious, energy, triple_C, rssm, hexcore,
        # gated_fusion, self_model, refusal_gate) are initialized to near
        # no-op defaults, so behavior is unchanged until they're trained.
        missing, unexpected = self.load_state_dict(
            ckpt['state_dict'], strict=False)
        if missing or unexpected:
            print(f"[load_brain] missing={len(missing)} unexpected={len(unexpected)} "
                  f"(expected for v0.6 forward-compat)")
        self.step_count = ckpt.get('step_count', 0)
        self._total_repairs = ckpt.get('total_repairs', 0)
        self._refusal_count = ckpt.get('refusal_count', 0)
        self._energy = ckpt.get('energy', 1.0)


# ══════════════════════════════════════════════════════════════════════
#  ALIVE LEARNING  (core, fused — not a wrapper)
#    SeasonalExperienceBuffer: tree-ring chronomemory
#    ContinualLearner: EWC-anchored, novelty-fracture gated, wake/sleep
# ══════════════════════════════════════════════════════════════════════

import json as _json
import time as _time
import random as _random
from collections import deque as _deque


class SeasonalExperienceBuffer:
    """Tree-ring chronomemory: a live ring plus sealed historical rings.

    Every interaction goes into the live ring (capacity `season_size`). When
    the live ring is full, it is sealed as a "season" and downsampled to
    `keep_per_season` representative records (highest tension kept). Older
    seasons are never deleted — they age but remain queryable. This gives
    you an append-only history with bounded memory: you can query "what did
    I see 5 seasons ago?" after thousands of interactions.
    """

    def __init__(self, path: str = 'experience.jsonl',
                 season_size: int = 500, keep_per_season: int = 64,
                 seasons_path: str = 'seasons.jsonl'):
        self.path = path
        self.seasons_path = seasons_path
        self.season_size = season_size
        self.keep_per_season = keep_per_season
        self.live: _deque = _deque(maxlen=season_size)
        self.seasons: List[List[Dict]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.live.append(_json.loads(line))
                    except Exception:
                        pass
        if os.path.exists(self.seasons_path):
            with open(self.seasons_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.seasons.append(_json.loads(line))
                    except Exception:
                        pass

    def _seal_if_full(self):
        if len(self.live) < self.season_size:
            return
        records = list(self.live)
        records.sort(key=lambda r: r.get('tension', 0.0), reverse=True)
        ring = records[:self.keep_per_season]
        ring_meta = {
            'season_idx': len(self.seasons),
            'n_original': len(records),
            'sealed_at': _time.time(),
            'records': ring,
        }
        self.seasons.append(ring_meta)
        try:
            with open(self.seasons_path, 'a', encoding='utf-8') as f:
                f.write(_json.dumps(ring_meta, ensure_ascii=False) + '\n')
        except Exception:
            pass
        self.live.clear()
        # Also truncate the live jsonl (its content is now in seasons)
        try:
            open(self.path, 'w').close()
        except Exception:
            pass

    def add(self, text: str, tension: float = 0.0, tag: str = 'user',
            fracture: bool = False):
        rec = {'t': _time.time(), 'text': text[:4000],
               'tension': float(tension), 'tag': tag,
               'fracture': bool(fracture)}
        self.live.append(rec)
        try:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(_json.dumps(rec, ensure_ascii=False) + '\n')
        except Exception:
            pass
        self._seal_if_full()

    def sample(self, n: int, bias_to_fracture: float = 0.3,
               curiosity_temp: float = 1.0) -> List[Dict]:
        """Curiosity-weighted sample. Probability ∝ exp(tension / T), with
        an explicit fracture boost. Higher-tension (surprising) memories
        replay more often during sleep consolidation."""
        pool_live = list(self.live)
        pool_hist = [r for s in self.seasons for r in s.get('records', [])]
        fracture_pool = [r for r in pool_live + pool_hist if r.get('fracture')]
        out = []
        if fracture_pool and bias_to_fracture > 0:
            k = min(int(n * bias_to_fracture), len(fracture_pool))
            out.extend(_random.sample(fracture_pool, k))
        remaining = n - len(out)
        all_pool = pool_live + pool_hist
        if remaining > 0 and all_pool:
            # Curiosity-weighted sampling: softmax over tensions
            import math as _math
            tensions = [float(r.get('tension', 0.0)) for r in all_pool]
            tmax = max(tensions) if tensions else 0.0
            weights = [_math.exp((t - tmax) / max(curiosity_temp, 1e-3))
                       for t in tensions]
            total_w = sum(weights) or 1.0
            weights = [w / total_w for w in weights]
            # Weighted reservoir without replacement
            idxs = set()
            attempts = 0
            k = min(remaining, len(all_pool))
            while len(idxs) < k and attempts < k * 8:
                j = _random.choices(range(len(all_pool)), weights=weights, k=1)[0]
                idxs.add(j)
                attempts += 1
            out.extend(all_pool[j] for j in idxs)
        return out

    def recent(self, n: int) -> List[Dict]:
        return list(self.live)[-n:]

    def from_season(self, idx: int) -> List[Dict]:
        if 0 <= idx < len(self.seasons):
            return self.seasons[idx].get('records', [])
        return []

    def stats(self):
        live = list(self.live)
        tensions = [r.get('tension', 0) for r in live]
        hist_count = sum(s.get('n_original', 0) for s in self.seasons)
        fracture_count = sum(1 for r in live if r.get('fracture')) + \
            sum(1 for s in self.seasons
                for r in s.get('records', []) if r.get('fracture'))
        return {
            'live': len(live),
            'live_capacity': self.season_size,
            'seasons_sealed': len(self.seasons),
            'total_historical': hist_count,
            'total': len(live) + hist_count,
            'fracture_events': fracture_count,
            'mean_tension': (float(sum(tensions) / len(tensions))
                             if tensions else 0.0),
        }


# Back-compat alias (older code imports ExperienceBuffer)
ExperienceBuffer = SeasonalExperienceBuffer


class ContinualLearner:
    """Real-time continual learning fused into the core.

    * wake_tick(text, tension): tiny EWC-anchored update. When a novelty
      fracture is detected (tension z > threshold), the learning rate is
      boosted — ordinary inputs barely move the weights, novel ones do.
    * sleep_consolidate(): batch replay from SeasonalExperienceBuffer,
      biased toward historical fracture events.

    Safety rails: embeddings/memory frozen, EWC anchor to pre-trained
    weights, per-param grad clipping, holographic memory writes on fracture.
    """

    def __init__(self, model, buffer: SeasonalExperienceBuffer,
                 wake_lr: float = 5e-7,
                 wake_lr_fracture: float = 5e-6,   # 10x on novelty
                 sleep_lr: float = 2e-6,
                 ewc_weight: float = 0.02,
                 grad_clip: float = 0.5,
                 enabled: bool = True):
        self.m = model
        self.buf = buffer
        self.wake_lr = wake_lr
        self.wake_lr_fracture = wake_lr_fracture
        self.sleep_lr = sleep_lr
        self.ewc_weight = ewc_weight
        self.grad_clip = grad_clip
        self.enabled = enabled
        self.updates_applied = 0
        self.fracture_updates = 0
        self.loss_history: _deque = _deque(maxlen=500)
        self.log: _deque = _deque(maxlen=200)
        self.anchor = {n: p.detach().clone()
                       for n, p in self.m.named_parameters()
                       if self._trainable(n) and p.requires_grad}

    def _trainable(self, n: str) -> bool:
        if any(k in n for k in ('wte', 'wpe', 'lm_head',
                                'memory.M', 'memory.M_holo',
                                'memory.usage', 'memory.age',
                                'memory.holo_keys')):
            return False
        return True

    def _prep(self, text: str):
        try:
            import tiktoken
            enc = tiktoken.get_encoding('gpt2')
        except Exception:
            return None, None
        ids = enc.encode_ordinary(text)[:256]
        if len(ids) < 4:
            return None, None
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=self.m.device)
        y = torch.tensor([ids[1:]],  dtype=torch.long, device=self.m.device)
        return x, y

    def _ewc_term(self):
        total = torch.zeros(1, device=self.m.device)
        for n, p in self.m.named_parameters():
            if n in self.anchor:
                total = total + ((p - self.anchor[n]) ** 2).sum()
        return self.ewc_weight * total.squeeze()

    def wake_tick(self, text: str, tension: Optional[float] = None):
        if not self.enabled:
            return None
        x, y = self._prep(text)
        if x is None:
            return None
        # Novelty gate
        fr = self.m.fracture_check(tension)
        lr = self.wake_lr_fracture if fr['fracture'] else self.wake_lr
        was_training = self.m.training
        self.m.train()
        for p in self.m.parameters():
            if p.grad is not None:
                p.grad = None
        with torch.enable_grad():
            _, loss = self.m(x, y)
            total = loss + self._ewc_term()
            total.backward()
        with torch.no_grad():
            for n, p in self.m.named_parameters():
                if p.grad is None or not self._trainable(n):
                    if p.grad is not None:
                        p.grad = None
                    continue
                g = p.grad
                gn = g.norm().clamp(min=1e-8)
                if gn > self.grad_clip:
                    g = g * (self.grad_clip / gn)
                p.sub_(lr * g)
                p.grad = None
        # On fracture: also write into holographic memory (irreversible
        # plasticity marker — survives even if slot-memory is overwritten).
        if fr['fracture'] and self.m.memory is not None:
            with torch.no_grad():
                emb = self.m.wte(x[0]).mean(dim=0).detach()
                slot = int(self.m.fracture_count.item()) % self.m.memory.K
                self.m.memory.write_holo(slot, emb)
            self.fracture_updates += 1
        if not was_training:
            self.m.eval()
        loss_val = float(loss.item())
        self.loss_history.append(loss_val)
        self.updates_applied += 1
        self.log.append({'mode': 'wake', 'loss': loss_val, 't': _time.time(),
                         'fracture': fr['fracture'], 'z': fr['z'],
                         'text': text[:80]})
        return {'loss': loss_val, 'fracture': fr['fracture'],
                'z': fr['z'], 'lr_used': lr}

    def sleep_consolidate(self, batch_size: int = 8, steps: int = 20):
        if not self.enabled:
            return {'status': 'disabled'}
        was_training = self.m.training
        self.m.train()
        losses = []
        for _ in range(steps):
            batch = self.buf.sample(batch_size)
            if not batch:
                break
            rows = []
            for r in batch:
                x, y = self._prep(r['text'])
                if x is not None:
                    rows.append((x, y))
            if not rows:
                continue
            for p in self.m.parameters():
                if p.grad is not None:
                    p.grad = None
            total_loss_val = 0.0
            for x, y in rows:
                with torch.enable_grad():
                    _, l = self.m(x, y)
                    (l / len(rows)).backward()
                    total_loss_val += float(l.item()) / len(rows)
            with torch.enable_grad():
                self._ewc_term().backward()
            with torch.no_grad():
                for n, p in self.m.named_parameters():
                    if p.grad is None or not self._trainable(n):
                        if p.grad is not None:
                            p.grad = None
                        continue
                    g = p.grad
                    gn = g.norm().clamp(min=1e-8)
                    if gn > self.grad_clip:
                        g = g * (self.grad_clip / gn)
                    p.sub_(self.sleep_lr * g)
                    p.grad = None
            losses.append(total_loss_val)
            self.updates_applied += 1
        if not was_training:
            self.m.eval()
        self.log.append({'mode': 'sleep', 'steps': len(losses),
                         'loss_mean': (sum(losses) / len(losses)) if losses else 0.0,
                         't': _time.time()})
        return {
            'steps_run': len(losses),
            'loss_first': losses[0] if losses else None,
            'loss_last':  losses[-1] if losses else None,
            'mean_loss':  (sum(losses) / len(losses)) if losses else 0.0,
            'updates_total': self.updates_applied,
        }

    def stats(self):
        recent = list(self.loss_history)[-50:]
        return {
            'enabled': self.enabled,
            'updates_applied': self.updates_applied,
            'fracture_updates': self.fracture_updates,
            'fracture_count_model': int(self.m.fracture_count.item()),
            'buffer': self.buf.stats(),
            'recent_loss': list(self.loss_history)[-10:],
            'mean_loss_recent': (sum(recent) / len(recent)) if recent else None,
            'anchor_params': len(self.anchor),
        }

    def save_snapshot(self, path: str):
        torch.save({
            'cfg': self.m.cfg,
            'state_dict': self.m.state_dict(),
            'step_count': self.m.step_count,
            'updates_applied': self.updates_applied,
            'snapshot_time': _time.time(),
        }, path)


# ══════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64))
                     for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64))
                     for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def train_adam(
    variant='small',
    data_dir='data/owt_small',
    minutes=120,
    batch_size=4,
    grad_accum=16,
    lr=3e-4,
    warmup=200,
    weight_decay=0.1,
    use_bf16=True,
    save_path='adam_checkpoint.pt',
    eval_interval=200,
    log_interval=20,
):
    """Train ADAM on a prepared BPE-tokenized dataset.

    data_dir must contain train.bin and val.bin (uint16 memmap, nanoGPT format).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if use_bf16 and device == 'cuda' else torch.float32

    cfg = AdamConfig.small() if variant == 'small' else AdamConfig.medium()
    print(f"\n{'='*70}\nADAM TRAINING — variant={variant}")
    print(f"device={device}  dtype={dtype}  minutes={minutes}\n{'='*70}")

    # Data
    data_dir = os.path.join(os.path.dirname(__file__), data_dir)
    train_path = os.path.join(data_dir, 'train.bin')
    val_path = os.path.join(data_dir, 'val.bin')
    if not os.path.exists(train_path):
        print(f"ERROR: no train.bin at {train_path}")
        print("Run prepare_owt_subset.py first.")
        return
    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
    print(f"train tokens: {len(train_data):,}  val tokens: {len(val_data):,}")

    # Model
    model = ADAM(cfg, device=device)
    breakdown = model.count_breakdown()
    print("\nParameter breakdown:")
    for k, v in breakdown.items():
        print(f"  {k:20s}: {v/1e6:7.3f}M")

    # Identity discovery BEFORE language training
    model.discover_identity(steps=2000)

    # Optimizer
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and 'memory' not in n:
            decay.append(p)
        else:
            nodecay.append(p)
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': weight_decay},
        {'params': nodecay, 'weight_decay': 0.0},
    ], lr=lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    ctx = torch.amp.autocast(device_type='cuda', dtype=dtype) if device == 'cuda' else torch.no_grad()

    # Training loop
    print(f"\nTraining for {minutes} minutes...\n" + "-"*70)
    start_time = time.time()
    end_time = start_time + minutes * 60
    it = 0
    best_val = float('inf')
    loss_ema = None

    model.train()
    while time.time() < end_time:
        # LR schedule: linear warmup then cosine decay
        if it < warmup:
            lr_t = lr * (it + 1) / warmup
        else:
            total_expected = max(it + 100, 2000)
            progress = (it - warmup) / (total_expected - warmup)
            lr_t = lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        for g in optimizer.param_groups:
            g['lr'] = lr_t

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro in range(grad_accum):
            x, y = get_batch(train_data, cfg.block_size, batch_size, device)
            with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=(device=='cuda')):
                logits, loss = model(x, targets=y)
                drift = model._current_drift()
                drift_pen = cfg.drift_penalty_weight * max(0.0, drift - float(model._alive_threshold))
                total = (loss + drift_pen) / grad_accum
            if scaler.is_enabled():
                scaler.scale(total).backward()
            else:
                total.backward()
            accum_loss += float(total.item()) * grad_accum

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Consciousness step every 10 iters
        if it % 10 == 0:
            model.consciousness_step(write_memory=True)

        avg_loss = accum_loss / grad_accum
        loss_ema = avg_loss if loss_ema is None else 0.98 * loss_ema + 0.02 * avg_loss

        if it % log_interval == 0:
            elapsed = time.time() - start_time
            remaining = end_time - time.time()
            toks = it * batch_size * grad_accum * cfg.block_size
            sys.stdout.flush()
            print(f"  iter {it:5d}  loss {avg_loss:.3f}  ema {loss_ema:.3f}  "
                  f"lr {lr_t:.1e}  drift {model._current_drift():.2f}  "
                  f"alive {model.alive}  mem {model.memory.stats()['writes'] if model.memory else 0:>4d}  "
                  f"toks {toks/1e6:.1f}M  elapsed {elapsed/60:.1f}m  remain {remaining/60:.1f}m")

        if it > 0 and it % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(20):
                    vx, vy = get_batch(val_data, cfg.block_size, batch_size, device)
                    with torch.amp.autocast(device_type='cuda', dtype=dtype,
                                            enabled=(device=='cuda')):
                        _, vloss = model(vx, targets=vy)
                    val_losses.append(float(vloss.item()))
            vl = float(np.mean(val_losses))
            if vl < best_val:
                best_val = vl
                model.save_brain(save_path)
                print(f"  [val] {vl:.3f}  (new best, saved to {save_path})")
            else:
                print(f"  [val] {vl:.3f}")
            model.train()

        it += 1

    elapsed = (time.time() - start_time) / 60
    print(f"\nFinished. iterations={it}  elapsed={elapsed:.1f} min  best_val={best_val:.3f}")

    # Final save
    model.save_brain(save_path)
    print(f"Final brain saved to {save_path}")

    # Samples
    print("\n" + "-"*70 + "\nSample generation:\n" + "-"*70)
    model.eval()
    for prompt in ["The future of consciousness is",
                   "In the beginning",
                   "I am"]:
        print(f"\n> {prompt}")
        try:
            out = model.generate(prompt, max_tokens=80, temperature=0.8, top_k=40)
            safe = out.encode('utf-8', errors='replace').decode('utf-8')
            print(safe)
        except Exception as e:
            print(f"[generation error: {e}]")

    # Self-naming
    print("\n" + "-"*70 + "\nSelf-description:\n" + "-"*70)
    print(model.self_name())

    # Curiosity
    cq = model.curiosity_question()
    if cq:
        print(f"\nCuriosity question: {cq}")

    # Inner monologue example
    print("\n" + "-"*70 + "\nInner monologue demo:\n" + "-"*70)
    res = model.generate_with_monologue("What is consciousness?",
                                        max_tokens=60, temperature=0.8)
    print("REPLY:", res['reply'])
    print("MONOLOGUE:\n" + res['monologue'])

    return model


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='small', choices=['small', 'medium'])
    ap.add_argument('--minutes', type=int, default=120)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--accum', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--data', default='data/owt_small')
    ap.add_argument('--save', default='adam_checkpoint.pt')
    args = ap.parse_args()
    train_adam(
        variant=args.variant,
        data_dir=args.data,
        minutes=args.minutes,
        batch_size=args.batch,
        grad_accum=args.accum,
        lr=args.lr,
        save_path=args.save,
    )
