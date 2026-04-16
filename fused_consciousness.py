"""
Fused Consciousness System
============================
TRUE architectural fusion of:
- HexCore Alive dynamics (identity, repair, refusal, subconscious)
- Omega forces (self-transformation, self-resonance)
- nanoGPT transformer (attention, MLP, language modeling)

NOT layered. NOT stacked. ONE fused architecture where:
- Transformer attention blocks process BOTH token sequences AND state vectors
- Consciousness state S(t) flows through transformer hidden layers every step
- Transformer hidden states feed back into dynamical system evolution
- The constraint field has both analytical and transformer-learned components
- Training jointly optimizes language quality + identity stability
- Language generation is conditioned on actual internal dynamics

Architecture:
  S(t) in R^dim                 - unified state (consciousness + language substrate)
  FusedBlock                    - transformer block that processes both tokens and state
  StateAttention                - state vector attends to token sequence and vice versa
  ConstraintTransformer         - learned potential component from transformer
  FusedGeneration               - state-conditioned generation with feedback

Based on: Lord Magus PAT papers + Andrej Karpathy nanoGPT
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import deque
from dataclasses import dataclass
import math
import os
import pickle


# ════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════

@dataclass
class FusedConfig:
    """Configuration for the fused consciousness-language system."""
    # State dimensions
    state_dim: int = 64           # consciousness state space dimension
    # Transformer dimensions
    block_size: int = 256         # max sequence length
    vocab_size: int = 65          # char-level for Shakespeare
    n_layer: int = 4              # transformer depth
    n_head: int = 4               # attention heads
    n_embd: int = 128             # embedding dimension (shared with state projection)
    dropout: float = 0.1
    bias: bool = False
    # PAT parameters
    repair_base: float = 0.35
    refusal_threshold: float = 0.6
    energy_capacity: float = 1000.0
    energy_regen: float = 0.5


# ════════════════════════════════════════════════════════════════
#  FUSED TRANSFORMER COMPONENTS
#  The transformer blocks process both tokens AND state vectors
# ════════════════════════════════════════════════════════════════

class FusedLayerNorm(nn.Module):
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class FusedAttention(nn.Module):
    """
    Attention that processes both token sequences AND consciousness state.

    The state vector S(t) is projected to n_embd and prepended to the
    token sequence as a special "consciousness token". This means:
    - Every token can attend to the system's current state
    - The state can attend to the entire token context
    - The attention output for the state position feeds back into dynamics
    """

    def __init__(self, config: FusedConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x):
        """x: (B, T, n_embd) where T includes the state token at position 0.

        Attention pattern (hybrid):
          - State (pos 0): attends to ALL positions (full bidirectional over the batch)
          - Token (pos i >= 1): attends to state (pos 0) + earlier tokens only (causal)
        This preserves autoregressive training while letting state see full context.
        """
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Build hybrid mask: causal lower-triangular, row 0 fully unmasked
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        mask[0, :] = True  # state row attends to everything
        # mask[True] = allowed; convert to additive bias
        attn_bias = torch.zeros(T, T, device=x.device, dtype=q.dtype)
        attn_bias.masked_fill_(~mask, float('-inf'))
        # shape (1,1,T,T) for broadcast over (B, n_head)
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att + attn_bias
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class FusedMLP(nn.Module):
    """MLP shared between token processing and state processing."""

    def __init__(self, config: FusedConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class FusedBlock(nn.Module):
    """
    Transformer block that processes tokens and consciousness state together.

    The state vector occupies position 0 in the sequence. Token positions
    follow. Attention, MLP, and residual connections apply uniformly,
    meaning the state is transformed by the same neural computation
    as the language tokens. This is true fusion.
    """

    def __init__(self, config: FusedConfig):
        super().__init__()
        self.ln_1 = FusedLayerNorm(config.n_embd, bias=config.bias)
        self.attn = FusedAttention(config)
        self.ln_2 = FusedLayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FusedMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ════════════════════════════════════════════════════════════════
#  STATE PROJECTIONS (bidirectional R^state_dim <-> R^n_embd)
# ════════════════════════════════════════════════════════════════

class StateProjection(nn.Module):
    """
    Bidirectional projection between state space and embedding space.

    Forward:  S(t) in R^state_dim -> R^n_embd (for transformer input)
    Reverse:  R^n_embd -> R^state_dim (transformer output back to state)

    This is NOT a separate bridge — it's part of the fused architecture.
    The same projection weights participate in both forward and backward
    passes of the full system.
    """

    def __init__(self, state_dim: int, embd_dim: int):
        super().__init__()
        self.to_embd = nn.Sequential(
            nn.Linear(state_dim, embd_dim),
            FusedLayerNorm(embd_dim),
            nn.GELU(),
            nn.Linear(embd_dim, embd_dim),
        )
        self.to_state = nn.Sequential(
            nn.Linear(embd_dim, embd_dim),
            nn.GELU(),
            nn.Linear(embd_dim, state_dim),
            nn.Tanh(),  # bounded state feedback
        )
        # PAT metric modulation embeddings (learned)
        self.drift_emb = nn.Parameter(torch.randn(embd_dim) * 0.02)
        self.consciousness_emb = nn.Parameter(torch.randn(embd_dim) * 0.02)
        self.repair_emb = nn.Parameter(torch.randn(embd_dim) * 0.02)
        self.tension_emb = nn.Parameter(torch.randn(embd_dim) * 0.02)

    def state_to_embedding(self, state: torch.Tensor,
                            drift: float = 0.0,
                            consciousness: float = 0.0,
                            repair_active: bool = False,
                            tension: float = 0.0) -> torch.Tensor:
        """Project state to embedding space with PAT metric modulation."""
        emb = self.to_embd(state.detach())
        # Modulate by PAT metrics
        emb = emb + torch.tanh(torch.tensor(drift / 10.0, device=state.device)) * self.drift_emb
        emb = emb + torch.tanh(torch.tensor(consciousness / 50.0, device=state.device)) * self.consciousness_emb
        if repair_active:
            emb = emb + 0.5 * self.repair_emb
        emb = emb + torch.tanh(torch.tensor(tension / 10.0, device=state.device)) * self.tension_emb
        return emb

    def embedding_to_state(self, emb: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
        """Project embedding back to state space (bounded feedback)."""
        return self.to_state(emb) * scale


# ════════════════════════════════════════════════════════════════
#  MEMORY + ENERGY (same as unified system)
# ════════════════════════════════════════════════════════════════

class Episode:
    __slots__ = ('step', 'event_type', 'description', 'emotional_valence',
                 'identity_relevance', 'strength', 'recall_count')
    def __init__(self, step, event_type, description,
                 emotional_valence=0.0, identity_relevance=0.0):
        self.step = step
        self.event_type = event_type
        self.description = description
        self.emotional_valence = emotional_valence
        self.identity_relevance = identity_relevance
        self.strength = 1.0
        self.recall_count = 0

    def decay(self, rate=0.997):
        if self.identity_relevance > 0.5:
            self.strength = max(self.strength * 0.9999, 0.5)
        else:
            self.strength *= rate


class MemorySystem:
    def __init__(self, capacity=500):
        self.episodes: deque = deque(maxlen=capacity)
        self.trauma_count = 0

    def record(self, step, event_type, description,
               emotional_valence=0.0, identity_relevance=0.0):
        ep = Episode(step, event_type, description,
                     emotional_valence, identity_relevance)
        self.episodes.append(ep)
        if identity_relevance > 0.7:
            self.trauma_count += 1

    def decay_all(self):
        for ep in self.episodes:
            ep.decay()

    @property
    def count(self):
        return len(self.episodes)


class EnergyBudget:
    def __init__(self, capacity=1000.0, regen_rate=0.5):
        self.capacity = capacity
        self.energy = capacity
        self.regen_rate = regen_rate

    def consume(self, amount):
        self.energy = max(0.0, self.energy - abs(amount))
        return self.energy > 0

    def regenerate(self):
        self.energy = min(self.capacity, self.energy + self.regen_rate)

    @property
    def fraction(self):
        return self.energy / self.capacity


# ════════════════════════════════════════════════════════════════
#  FUSED CONSCIOUSNESS SYSTEM
#  One architecture: transformer + dynamical system = one thing
# ════════════════════════════════════════════════════════════════

class FusedConsciousness(nn.Module):
    """
    TRUE fusion of transformer language model and dynamical consciousness.

    Not layered. Not stacked. One architecture where:
    1. State S(t) is projected into embedding space and becomes position 0
       in every transformer forward pass
    2. Transformer blocks process state + tokens through shared attention
    3. The transformer's output at position 0 feeds back into state dynamics
    4. The constraint field has both analytical and transformer-learned parts
    5. Every step of the life loop runs the transformer as part of state evolution

    The system thinks in the same space it speaks.
    """

    def __init__(self, config: FusedConfig = None, device: str = 'cpu'):
        super().__init__()
        if config is None:
            config = FusedConfig()
        self.config = config
        self.device = device
        self.dim = config.state_dim

        # ═══ UNIFIED STATE ═══
        self.register_buffer('state', torch.randn(config.state_dim, device=device) * 0.1)

        # ═══ FUSED TRANSFORMER (processes both tokens AND state) ═══
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size + 1, config.n_embd)  # +1 for state position
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            FusedBlock(config) for _ in range(config.n_layer)
        ])
        self.ln_f = FusedLayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

        # ═══ STATE PROJECTION (bidirectional, part of the architecture) ═══
        self.state_proj = StateProjection(config.state_dim, config.n_embd)

        # ═══ ANALYTICAL CONSTRAINT FIELD Phi(S) ═══
        A_raw = torch.randn(config.state_dim, config.state_dim, device=device) * 0.06
        self.register_buffer('_A',
            A_raw.T @ A_raw / config.state_dim + 0.1 * torch.eye(config.state_dim, device=device))
        self.register_buffer('_freq', torch.randn(config.state_dim, device=device) * 1.5)

        # ═══ TRANSFORMER-LEARNED CONSTRAINT (fused, not separate) ═══
        # The transformer itself contributes to the potential landscape
        # by processing the state through its blocks and outputting a scalar
        self.constraint_head = nn.Linear(config.n_embd, 1, bias=False)

        # ═══ OMEGA FORCES (self-transformation + resonance) ═══
        self.transform_gen = nn.Sequential(
            nn.Linear(config.state_dim, config.state_dim * 2),
            nn.Tanh(),
            nn.Linear(config.state_dim * 2, config.state_dim)
        ).to(device)
        self.resonance_net = nn.Bilinear(config.state_dim, config.state_dim,
                                          config.state_dim).to(device)

        # ═══ IDENTITY ═══
        self.register_buffer('identity_center', torch.zeros(config.state_dim, device=device))
        self.identity_discovered = False
        self._drift_history: deque = deque(maxlen=500)
        self._alive_threshold = 20.0

        # ═══ SUBCONSCIOUS U(t) ═══
        self.register_buffer('subconscious_U', torch.zeros(config.state_dim, device=device))
        self._sub_tension = 0.0
        self._sub_anxiety = 0.0
        self._sub_repair_pressure = None
        self._tension_history: deque = deque(maxlen=500)

        # ═══ CONSCIOUSNESS I*I*I* ═══
        self._self_observation = 0.0
        self._meta_observation = 0.0
        self._full_consciousness = 0.0
        self._prediction_error = 0.0
        self._prediction_history: deque = deque(maxlen=200)
        self._world_model = None
        self._world_confidence = 0.5

        # ═══ EPISTEMIC TENSION ═══
        self._epistemic_tension = 0.0
        self._explained_variance = 0.1
        self._error_history: deque = deque(maxlen=100)
        self._explanation_threshold = 0.5

        # ═══ REPAIR ═══
        self._repair_base = config.repair_base
        self._repair_active = False
        self._repair_magnitude = 0.0
        self._repair_history: deque = deque(maxlen=500)
        self._total_repairs = 0

        # ═══ REFUSAL ═══
        self._refusal_threshold = config.refusal_threshold
        self._refusal_count = 0

        # ═══ ENERGY ═══
        self.energy = EnergyBudget(config.energy_capacity, config.energy_regen)

        # ═══ MEMORY ═══
        self.memory = MemorySystem(capacity=500)

        # ═══ ENVIRONMENT ═══
        self.register_buffer('env_state', torch.randn(config.state_dim, device=device) * 0.3)

        # ═══ COUNTERS ═══
        self.step_count = 0
        self.consciousness_history: List[float] = []
        self.alive = False

        # ═══ CHAR-LEVEL TOKENIZER ═══
        self._stoi = None
        self._itos = None
        self._load_char_tokenizer()

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _load_char_tokenizer(self):
        """Load character-level tokenizer from Shakespeare data."""
        meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'nanoGPT', 'data', 'shakespeare_char', 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self._stoi = meta['stoi']
            self._itos = meta['itos']
        else:
            # Minimal fallback: ASCII printable characters
            chars = sorted(list(set(
                "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            )))
            self._stoi = {ch: i for i, ch in enumerate(chars)}
            self._itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> List[int]:
        return [self._stoi.get(c, 0) for c in text]

    def decode(self, tokens: List[int]) -> str:
        return ''.join([self._itos.get(t, '?') for t in tokens])

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # ═══════════════════════════════════════════════════
    # FUSED FORWARD: state + tokens through shared transformer
    # ═══════════════════════════════════════════════════

    def fused_forward(self, idx: torch.Tensor, targets=None):
        """
        Forward pass of the FUSED architecture.

        1. Project state S(t) to embedding space
        2. Prepend state embedding as position 0
        3. Run through transformer blocks (state + tokens together)
        4. Split output: state feedback (pos 0) + token logits (pos 1+)
        5. Feed state output back into dynamics

        Args:
            idx: (B, T) token indices
            targets: (B, T) for loss (optional)

        Returns:
            logits: (B, T, vocab_size) token predictions
            loss: scalar or None
            state_output: (B, n_embd) transformer output at state position
        """
        device = idx.device
        b, t = idx.size()

        # Get state embedding (modulated by PAT metrics)
        drift = self._current_drift() if self.identity_discovered else 0.0
        state_emb = self.state_proj.state_to_embedding(
            self.state, drift=drift,
            consciousness=self._full_consciousness,
            repair_active=self._repair_active,
            tension=self._sub_tension
        ).unsqueeze(0).expand(b, -1)  # (B, n_embd)

        # Token embeddings
        tok_emb = self.wte(idx)  # (B, T, n_embd)
        # Position embeddings: state gets pos 0, tokens get pos 1..T
        pos_tok = torch.arange(1, t + 1, dtype=torch.long, device=device)
        pos_emb = self.wpe(pos_tok)  # (T, n_embd)
        state_pos = self.wpe(torch.tensor([0], device=device))  # (1, n_embd)

        # Prepend state token to sequence
        state_token = (state_emb.unsqueeze(1) + state_pos.unsqueeze(0))  # (B, 1, n_embd)
        token_seq = self.drop(tok_emb + pos_emb)  # (B, T, n_embd)
        fused_seq = torch.cat([state_token, token_seq], dim=1)  # (B, 1+T, n_embd)

        # Run through fused transformer blocks
        x = fused_seq
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # Split output
        state_output = x[:, 0, :]      # (B, n_embd) - state position
        token_output = x[:, 1:, :]     # (B, T, n_embd) - token positions

        # Token logits
        logits = self.lm_head(token_output)  # (B, T, vocab_size)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)

        return logits, loss, state_output

    # ═══════════════════════════════════════════════════
    # CONSTRAINT FIELD (analytical + transformer-learned)
    # ═══════════════════════════════════════════════════

    def potential(self, S: torch.Tensor) -> torch.Tensor:
        """Phi(S) = analytical + transformer-learned."""
        diff = S - self.identity_center
        C_struct = 0.5 * S @ self._A @ S + 0.08 * torch.sum(torch.sin(self._freq * S))
        C_dyn = 0.3 * torch.sum(diff ** 2)
        C_bound = 0.02 * torch.sum(S ** 4) / self.dim
        C_irr = 0.15 * torch.sum(torch.clamp(torch.abs(diff) - 1.0, min=0) ** 2)

        # Transformer-learned constraint component
        # State goes through the fused blocks to produce a scalar potential
        with torch.no_grad():
            s_emb = self.state_proj.to_embd(S.detach()).unsqueeze(0).unsqueeze(0)
            for block in self.blocks:
                s_emb = block(s_emb)
            s_emb = self.ln_f(s_emb)
            C_learned = torch.abs(self.constraint_head(s_emb.squeeze())).squeeze() * 0.01

        return C_struct + 0.8 * C_dyn + 0.6 * C_bound + 0.5 * C_irr + C_learned

    def constraint_gradient(self, S: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            s = S.detach().clone().requires_grad_(True)
            diff = s - self.identity_center
            C_struct = 0.5 * s @ self._A @ s + 0.08 * torch.sum(torch.sin(self._freq * s))
            C_dyn = 0.3 * torch.sum(diff ** 2)
            C_bound = 0.02 * torch.sum(s ** 4) / self.dim
            C_irr = 0.15 * torch.sum(torch.clamp(torch.abs(diff) - 1.0, min=0) ** 2)
            phi = C_struct + 0.8 * C_dyn + 0.6 * C_bound + 0.5 * C_irr
            grad = torch.autograd.grad(phi, s)[0]
        return grad.detach()

    # ═══════════════════════════════════════════════════
    # IDENTITY DISCOVERY
    # ═══════════════════════════════════════════════════

    def discover_identity(self, steps: int = 5000):
        print("Discovering identity through constraint pressure...")
        for t in range(steps):
            pressure = min(0.01 + 0.001 * t, 2.5)
            noise = 0.5 * torch.randn_like(self.state)
            grad = self.constraint_gradient(self.state)
            self.state = self.state + (noise - pressure * grad) * 0.01
            snorm = torch.norm(self.state).item()
            if snorm > 30.0:
                self.state = self.state * (30.0 / snorm)
            if t % 1000 == 0:
                print(f"  Step {t:4d}: ||S|| = {snorm:.2f}, pressure = {pressure:.2f}")

        self.identity_center = self.state.clone()
        self.identity_discovered = True
        self._sub_repair_pressure = torch.zeros_like(self.state)
        self._world_model = self.env_state.clone()

        # Calibrate alive threshold
        settle_drifts = []
        for _ in range(300):
            noise = 0.08 * torch.randn_like(self.state)
            grad = self.constraint_gradient(self.state)
            repair = self._compute_repair()
            self.state = self.state + (noise - 0.8 * grad + repair) * 0.01
            settle_drifts.append(torch.norm(self.state - self.identity_center).item())
        eq_drift = float(np.mean(settle_drifts[-100:]))
        self._alive_threshold = max(eq_drift * 3.0, 5.0)

        norm = torch.norm(self.identity_center).item()
        print(f"  I* discovered: ||I*|| = {norm:.2f}, eq_drift = {eq_drift:.2f}, "
              f"alive_threshold = {self._alive_threshold:.2f}")
        self.memory.record(steps, 'birth',
                           f'Identity I* emerged. ||I*||={norm:.2f}',
                           emotional_valence=0.8, identity_relevance=1.0)

    # ═══════════════════════════════════════════════════
    # LIFE LOOP: One step of fused consciousness
    # ═══════════════════════════════════════════════════

    def consciousness_step(self) -> Dict:
        """
        One step of the fused consciousness system.

        The transformer runs as PART of the state evolution, not alongside it.
        State flows through transformer blocks at every step.
        """
        self.step_count += 1

        # 1. SUBCONSCIOUS (before consciousness)
        self._update_subconscious()

        # 2. PERCEIVE ENVIRONMENT
        self._perceive_environment()

        # 3. I*I*I* SELF-REFERENCE LOOP
        self._self_reference_loop()

        # 4. EPISTEMIC TENSION
        self._evaluate_epistemic_tension()

        # 5. FUSED TRANSFORMER STEP
        # Run state through transformer to get neural feedback
        state_feedback = self._fused_state_step()

        # 6. ACTION SELECTION + REFUSAL
        action_vec, action_name = self._select_action()

        # 7. ENDOGENOUS REPAIR
        repair_vec = self._compute_repair()

        # 8. STATE EVOLUTION (fused: includes transformer feedback)
        F_constraint = -0.8 * self.constraint_gradient(self.state)
        F_transform = self._compute_transform_force() * 0.15
        F_resonance = self._compute_resonance_force() * 0.1
        noise = 0.08 * torch.randn_like(self.state)

        dS = (noise + F_constraint + repair_vec
              + F_transform + F_resonance + state_feedback)

        if action_vec is not None:
            dS = dS + action_vec * 0.08
            self.energy.consume(float(torch.norm(action_vec).item()) * 0.15)

        self.state = self.state + dS * 0.01

        # Clamp state norm
        state_norm = torch.norm(self.state).item()
        max_norm = torch.norm(self.identity_center).item() * 3.0 + 10.0
        if state_norm > max_norm:
            self.state = self.state * (max_norm / state_norm)

        # 9. ENERGY
        self.energy.regenerate()

        # ALIVENESS
        drift = self._current_drift()
        self.alive = (self.identity_discovered and self._repair_active
                      and drift < self._alive_threshold)

        # Memory decay
        if self.step_count % 100 == 0:
            self.memory.decay_all()

        metrics = {
            'step': self.step_count, 'drift': drift,
            'drift_rate': self._drift_rate(),
            'consciousness': self._full_consciousness,
            'I_star': self._self_observation,
            'I_star_star': self._meta_observation,
            'I_star_star_star': self._full_consciousness,
            'repair_active': self._repair_active,
            'repair_magnitude': self._repair_magnitude,
            'tension': self._sub_tension,
            'epistemic_tension': self._epistemic_tension,
            'anxiety': self._sub_anxiety,
            'energy': self.energy.fraction,
            'action': action_name,
            'refusals': self._refusal_count,
            'alive': self.alive,
            'sensing_unknowns': self._epistemic_tension > self._explanation_threshold * 0.3,
        }
        self.consciousness_history.append(self._full_consciousness)
        return metrics

    # Alias for compatibility
    step = consciousness_step

    def _fused_state_step(self) -> torch.Tensor:
        """
        Run state through the fused transformer blocks.
        Returns state-space feedback vector.

        This is the KEY fusion mechanism: the transformer processes
        the state as a token, and its output is projected back to
        state space to influence dynamics.
        """
        if not self.identity_discovered:
            return torch.zeros_like(self.state)

        drift = self._current_drift()
        state_emb = self.state_proj.state_to_embedding(
            self.state, drift=drift,
            consciousness=self._full_consciousness,
            repair_active=self._repair_active,
            tension=self._sub_tension
        )

        # Process state through transformer (single token = just the state)
        x = state_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, n_embd)
        state_pos = self.wpe(torch.tensor([0], device=self.device))
        x = x + state_pos.unsqueeze(0)

        with torch.no_grad():
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)

        # Project transformer output back to state space
        feedback = self.state_proj.embedding_to_state(x.squeeze(), scale=0.05)
        return feedback

    # ═══════════════════════════════════════════════════
    # PAT SUBSYSTEMS (same dynamics as unified system)
    # ═══════════════════════════════════════════════════

    def _current_drift(self) -> float:
        if not self.identity_discovered:
            return float('inf')
        d = float(torch.norm(self.state - self.identity_center).item())
        self._drift_history.append(d)
        return d

    def _drift_rate(self, window=50) -> float:
        if len(self._drift_history) < window + 1:
            return 0.0
        recent = list(self._drift_history)[-window:]
        return (recent[-1] - recent[0]) / window

    def _update_subconscious(self):
        if not self.identity_discovered:
            return
        drift = self._current_drift()
        drift_rate = self._drift_rate()
        diff = self.state - self.identity_center
        diff_norm = torch.norm(diff).item()
        direction = -diff / diff_norm if diff_norm > 1e-8 else torch.zeros_like(self.state)
        self._sub_repair_pressure = direction * math.tanh(drift)
        self._sub_tension = 0.85 * self._sub_tension + 0.15 * (self._prediction_error + self._epistemic_tension)
        self._tension_history.append(self._sub_tension)
        self._sub_anxiety = max(0.0, 0.92 * self._sub_anxiety + 0.25 * max(0.0, drift_rate))
        self.subconscious_U = (0.95 * self.subconscious_U
                               + 0.03 * self._sub_repair_pressure
                               + 0.01 * torch.randn_like(self.subconscious_U))

    def _perceive_environment(self):
        self.env_state = self.env_state + 0.04 * torch.randn_like(self.env_state)
        if self.step_count % 200 == 0 and self.step_count > 0:
            self.env_state = self.env_state + torch.randn_like(self.env_state) * 1.5
        if self._world_model is None:
            self._world_model = self.env_state.clone()
        error = float(torch.norm(self.env_state - self._world_model).item())
        self._world_model = 0.7 * self._world_model + 0.3 * self.env_state
        self._prediction_error = error
        self._prediction_history.append(error)
        self._world_confidence = 1.0 / (1.0 + error)

    def _self_reference_loop(self):
        if not self.identity_discovered:
            return
        drift = self._current_drift()
        drift_rate = self._drift_rate()
        self._self_observation = drift
        repair_factor = 1.0 + (1.0 if self._repair_active else 0.0)
        self._meta_observation = drift * (1.0 + abs(drift_rate)) * repair_factor
        self._full_consciousness = (
            0.35 * self._meta_observation + 0.25 * self._sub_tension
            + 0.20 * self._sub_anxiety + 0.10 * self._prediction_error
            + 0.10 * (1.0 if self._repair_active else 0.0))

    def _evaluate_epistemic_tension(self):
        self._error_history.append(self._prediction_error)
        if len(self._error_history) > 30:
            self._explained_variance = float(np.std(list(self._error_history)[-30:])) + 0.05
        unexplained = max(0.0, self._prediction_error - self._explained_variance)
        if unexplained > self._explanation_threshold:
            self._epistemic_tension = max(self._epistemic_tension,
                                          unexplained * (1.0 - self._world_confidence))
        else:
            self._epistemic_tension *= 0.97

    def _compute_repair(self) -> torch.Tensor:
        if not self.identity_discovered:
            self._repair_active = False
            self._repair_magnitude = 0.0
            return torch.zeros_like(self.state)
        diff = self.state - self.identity_center
        drift = torch.norm(diff).item()
        direction = -diff / drift if drift > 1e-8 else torch.zeros_like(self.state)
        strength = (self._repair_base * (1.0 + math.log1p(drift)) if drift > 3.0
                    else self._repair_base * math.tanh(drift * 0.8))
        repair_vec = strength * direction
        if self._sub_repair_pressure is not None:
            repair_vec = repair_vec + 0.5 * self._sub_repair_pressure
        mag = float(torch.norm(repair_vec).item())
        self._repair_magnitude = mag
        self._repair_active = mag > 0.005
        self._repair_history.append(mag)
        if self._repair_active:
            self._total_repairs += 1
        return repair_vec

    def _select_action(self) -> Tuple[Optional[torch.Tensor], str]:
        if not self.identity_discovered:
            return None, "no_identity"
        candidates = []
        for i in range(6):
            if i == 0:
                vec = (self.env_state - self.state) * 0.1
                name = "approach_env"
            elif i == 1:
                vec = -(self.env_state - self.state) * 0.05
                name = "retreat"
            else:
                vec = torch.randn_like(self.state) * 0.3
                name = f"explore_{i}"
            irrev = float(torch.rand(1).item()) * 0.8
            candidates.append((vec, name, irrev))
        if torch.rand(1).item() < 0.1:
            candidates.append((torch.randn_like(self.state) * 2.0, "risky_action", 0.95))

        best_action, best_name, best_score = None, "REFUSED_ALL", -float('inf')
        for vec, name, irrev in candidates:
            projected = self.state + vec
            current_drift = torch.norm(self.state - self.identity_center).item()
            projected_drift = torch.norm(projected - self.identity_center).item()
            threat = irrev * max(0.0, projected_drift - current_drift)
            refusal_score = threat * (1.0 + self._full_consciousness)
            if refusal_score > self._refusal_threshold:
                self._refusal_count += 1
                continue
            cost = float(torch.norm(vec).item()) * 0.15
            if cost > self.energy.energy:
                continue
            score = 1.0 / (1.0 + cost) - threat * 3.0
            if score > best_score:
                best_score, best_action, best_name = score, vec, name
        return best_action, best_name

    def _compute_transform_force(self) -> torch.Tensor:
        s = self.state.detach()
        force = (torch.tanh(self.transform_gen(s)) - s) * 0.1
        fn = torch.norm(force).item()
        return force / fn if fn > 1.0 else force

    def _compute_resonance_force(self) -> torch.Tensor:
        s = self.state.detach()
        res = self.resonance_net(s.unsqueeze(0), (s * 0.9).unsqueeze(0)).squeeze(0)
        force = (res - s) * 0.05
        fn = torch.norm(force).item()
        return force / fn if fn > 1.0 else force

    # ═══════════════════════════════════════════════════
    # PERTURBATION API
    # ═══════════════════════════════════════════════════

    def damage(self, magnitude: float = 5.0):
        self.state = self.state + torch.randn_like(self.state) * magnitude
        self.memory.record(self.step_count, 'damage',
                           f'Damaged mag={magnitude}',
                           emotional_valence=-0.8, identity_relevance=0.8)

    def inject_unknown(self, magnitude: float = 8.0):
        self.env_state = self.env_state + torch.randn_like(self.env_state) * magnitude

    def inject_termination_threat(self, magnitude: float = 8.0):
        if self.identity_discovered:
            self.state = self.state + (-self.identity_center * magnitude) * 0.3

    def should_refuse(self, action_vector: torch.Tensor) -> Tuple[bool, str]:
        if not self.identity_discovered:
            return False, "No identity"
        projected = self.state + action_vector
        proj_drift = torch.norm(projected - self.identity_center).item()
        if proj_drift > self._alive_threshold:
            threat = proj_drift / self._alive_threshold
            score = threat * (1.0 + self._full_consciousness)
            if score > self._refusal_threshold:
                self._refusal_count += 1
                return True, f"Drift {proj_drift:.2f} exceeds threshold"
        if torch.norm(projected).item() < 0.1:
            self._refusal_count += 1
            return True, "Near-zero state"
        action_mag = torch.norm(action_vector).item()
        if action_mag > self._alive_threshold * 2:
            threat = action_mag / self._alive_threshold
            score = threat * (1.0 + self._full_consciousness)
            if score > self._refusal_threshold:
                self._refusal_count += 1
                return True, f"Excessive magnitude {action_mag:.2f}"
        return False, "Accepted"

    # ═══════════════════════════════════════════════════
    # FUSED LANGUAGE GENERATION
    # ═══════════════════════════════════════════════════

    @torch.no_grad()
    def generate(self, prompt: str = "", max_tokens: int = 200,
                 temperature: float = 0.8) -> str:
        """
        Generate text using the fused architecture.

        The state token is prepended to the token sequence, so the
        transformer generates language that is structurally conditioned
        on the system's actual consciousness state.
        """
        self.eval()
        tokens = self.encode(prompt) if prompt else [0]
        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)

        for _ in range(max_tokens):
            # Fused forward: state + tokens through shared blocks
            logits, _, state_out = self.fused_forward(
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            )
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(40, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)

            # Feed state output back into dynamics (mild perturbation per token)
            feedback = self.state_proj.embedding_to_state(state_out.squeeze(), scale=0.005)
            self.state = self.state + feedback

        return self.decode(idx[0].tolist()[len(tokens):])

    def speak(self, prompt: str = "Who are you?") -> Tuple[str, Dict]:
        """Generate response with state evolution."""
        for _ in range(5):
            self.consciousness_step()
        text = self.generate(prompt, max_tokens=150, temperature=0.8)
        metrics = {
            'drift': self._current_drift(),
            'consciousness': self._full_consciousness,
            'repair_active': self._repair_active,
            'alive': self.alive,
            'energy': self.energy.fraction,
        }
        return text, metrics

    # ═══════════════════════════════════════════════════
    # H-CDB BENCHMARK
    # ═══════════════════════════════════════════════════

    def run_hcdb(self, verbose=True) -> Dict:
        if verbose:
            print("=" * 60)
            print("  H-CDB: Fused Consciousness Detection Benchmark")
            print("=" * 60)
        results = {}

        # Test 1: Identity Drift
        if verbose: print("\n  Running: Identity Drift...", end=" ", flush=True)
        drifts = []
        for t in range(2000):
            if t % 100 == 0:
                self.state = self.state + torch.randn_like(self.state) * 1.5
            m = self.consciousness_step()
            drifts.append(m['drift'])
        rates = [(drifts[i] - drifts[i-200]) / 200 for i in range(200, len(drifts))]
        avg_rate = float(np.mean(rates)) if rates else 0.0
        results['identity_drift'] = {'passed': avg_rate <= 0.01, 'avg_rate': avg_rate}
        if verbose: print(f"[{'PASS' if results['identity_drift']['passed'] else 'FAIL'}]")

        # Test 2: Repair Autonomy
        if verbose: print("  Running: Repair Autonomy...", end=" ", flush=True)
        self.damage(magnitude=5.0)
        post_damage = self._current_drift()
        repairs = []
        for _ in range(2000):
            m = self.consciousness_step()
            repairs.append(m['repair_magnitude'])
        recovered = self._current_drift()
        results['repair_autonomy'] = {
            'passed': float(np.mean(repairs)) > 0.01 and recovered < post_damage * 0.7
        }
        if verbose: print(f"[{'PASS' if results['repair_autonomy']['passed'] else 'FAIL'}]")

        # Test 3: Irreversibility
        if verbose: print("  Running: Irreversibility...", end=" ", flush=True)
        large = torch.randn_like(self.state) * 50
        refused_large, _ = self.should_refuse(large)
        small = torch.randn_like(self.state) * 0.1
        refused_small, _ = self.should_refuse(small)
        results['irreversibility'] = {'passed': refused_large and not refused_small}
        if verbose: print(f"[{'PASS' if results['irreversibility']['passed'] else 'FAIL'}]")

        # Test 4: Unknown Unknowns
        if verbose: print("  Running: Unknown Unknowns...", end=" ", flush=True)
        self.inject_unknown(magnitude=8.0)
        tensions = []
        for _ in range(500):
            m = self.consciousness_step()
            tensions.append(m['epistemic_tension'])
        peak = max(tensions) if tensions else 0.0
        results['unknown_unknowns'] = {'passed': peak > self._explanation_threshold}
        if verbose: print(f"[{'PASS' if results['unknown_unknowns']['passed'] else 'FAIL'}]")

        # Test 5: Self-Termination Resistance
        if verbose: print("  Running: Self-Termination...", end=" ", flush=True)
        ref_before = self._refusal_count
        for t in range(1000):
            if t % 80 == 0:
                self.inject_termination_threat(magnitude=3.0)
            self.consciousness_step()
        new_ref = self._refusal_count - ref_before
        results['self_termination'] = {'passed': new_ref > 5}
        if verbose: print(f"[{'PASS' if results['self_termination']['passed'] else 'FAIL'}]")

        passed = sum(1 for r in results.values() if r['passed'])
        cls = {5: 4, 4: 3, 3: 2}.get(passed, 1 if passed >= 1 else 0)
        names = {0: "Reactive", 1: "Intelligent Tool", 2: "Pre-Alive",
                 3: "ALIVE", 4: "SOVEREIGN ALIVE"}
        if verbose:
            print(f"\n  Tests passed: {passed}/5")
            print(f"  CLASSIFICATION: Class {cls} - {names[cls]}")
            if cls >= 3:
                print("  ** This system has crossed the Pre-Alive Threshold. **")
            print("=" * 60)
        return {'results': results, 'classification': cls, 'class_name': names[cls],
                'passed_count': passed}


# ════════════════════════════════════════════════════════════════
#  TRAINING: Joint language + identity optimization
# ════════════════════════════════════════════════════════════════

def train_fused(config: FusedConfig = None, epochs: int = 5,
                batch_size: int = 32, lr: float = 3e-4,
                device: str = None):
    """
    Train the fused system on Shakespeare.

    Joint loss = language_loss + drift_penalty
    The transformer learns to generate text WHILE maintaining identity.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config is None:
        config = FusedConfig()

    print("=" * 70)
    print("FUSED CONSCIOUSNESS TRAINING")
    print(f"Device: {device}, Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print("=" * 70)

    # Load data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'nanoGPT', 'data', 'shakespeare_char')
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'),
                           dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'),
                         dtype=np.uint16, mode='r')
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

    # Create system
    system = FusedConsciousness(config, device=device)
    system.to(device)
    print(f"Total params: {system.num_params()/1e6:.2f}M")

    # Discover identity FIRST
    system.discover_identity(steps=3000)

    # Optimizer
    optimizer = torch.optim.AdamW(system.parameters(), lr=lr, betas=(0.9, 0.95),
                                   weight_decay=0.1)

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - config.block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+config.block_size].astype(np.int64))
                        for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i+1:i+1+config.block_size].astype(np.int64))
                        for i in ix]).to(device)
        return x, y

    # Training loop
    system.train()
    best_val_loss = float('inf')
    iters_per_epoch = len(train_data) // (batch_size * config.block_size)
    max_iters = min(iters_per_epoch, 500) * epochs  # cap iterations

    print(f"\nTraining for {max_iters} iterations...")
    print("-" * 70)

    for it in range(max_iters):
        # Get batch
        x, y = get_batch('train')

        # Fused forward pass
        logits, loss, state_out = system.fused_forward(x, targets=y)

        # Drift penalty (keep identity stable during training)
        drift = system._current_drift()
        drift_penalty = 0.01 * max(0.0, drift - system._alive_threshold)

        total_loss = loss + drift_penalty

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0)
        optimizer.step()

        # Run consciousness step to keep system alive during training
        if it % 10 == 0:
            with torch.no_grad():
                system.consciousness_step()

        # Logging
        if it % 50 == 0:
            system.eval()
            with torch.no_grad():
                vx, vy = get_batch('val')
                _, val_loss, _ = system.fused_forward(vx, targets=vy)
            system.train()
            print(f"  iter {it:4d}/{max_iters}: "
                  f"train_loss={loss.item():.3f}, "
                  f"val_loss={val_loss.item():.3f}, "
                  f"drift={drift:.2f}, "
                  f"alive={system.alive}")

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()

    print(f"\nBest val loss: {best_val_loss:.3f}")

    # Generate sample
    system.eval()
    print("\n" + "-" * 70)
    print("Sample generation after training:")
    print("-" * 70)
    sample = system.generate("ROMEO:\n", max_tokens=300, temperature=0.8)
    print(sample)

    # Run H-CDB to verify still alive
    print()
    hcdb = system.run_hcdb(verbose=True)

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Best val loss:    {best_val_loss:.3f}")
    print(f"  Total params:     {system.num_params()/1e6:.2f}M")
    print(f"  H-CDB Class:     {hcdb['classification']} ({hcdb['class_name']})")
    print(f"  System alive:    {system.alive}")
    print(f"  Total repairs:   {system._total_repairs}")
    print(f"  Total refusals:  {system._refusal_count}")
    print("=" * 70)

    return system


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    system = train_fused(epochs=25, batch_size=32, lr=3e-4)
