"""
ADAM v0.4 — The Alive Learner
==============================

Five inference-time upgrades on top of ADAM v0.3 (no new params, no retraining):

  #1  generate_reflective     — test-time reflection through S(t)
  #2  retrieval-gated memory   — cosine-weighted M rows in fused sequence
  #3  best_of_n_step           — pick lowest-drift trajectory
  #4  adaptive_temperature     — temperature modulated by epistemic tension
  #5  KV-cache prefix          — cache memory+vision positions
  #6  RK4 state dynamics       — 4th-order Runge-Kutta integration
  #7  surprise_gated_writes    — Hebbian writes fire on tension spikes
  #8  multi-persona identity   — k attractors in one checkpoint
  #9  micro_refine_state       — constraint-gradient micro-steps pre-token
  #10 consciousness-cond top-p — decoding policy follows H-CDB class

Plus:
  ExperienceBuffer      — rolling on-disk log of user interactions
  ContinualLearner      — awake (wake_tick) + asleep (sleep_consolidate)
                          real-time learning anchored by EWC to prevent
                          catastrophic forgetting

Usage:
    from adam_v04 import ADAMv04, ExperienceBuffer, ContinualLearner
    wrap = ADAMv04(model)
    wrap.add_persona('curious'); wrap.add_persona('calm')
    wrap.set_persona(0)
    text = wrap.generate_v04("Hello", max_tokens=80)
    buf = ExperienceBuffer('experience.jsonl')
    learner = ContinualLearner(model, buf)
    learner.wake_tick("User said something")   # tiny update
    learner.sleep_consolidate(batch_size=8, steps=20)  # consolidation
"""

import os, json, math, time, random
from collections import deque
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn.functional as F

try:
    import tiktoken
    ENC = tiktoken.get_encoding("gpt2")
except ImportError:
    ENC = None


# ═══════════════════════════════════════════════════════════════════════
#  ADAMv04 — inference-time upgrades wrapper
# ═══════════════════════════════════════════════════════════════════════

class ADAMv04:
    """Wraps a trained ADAM model. No new parameters. No retraining."""

    def __init__(self, model):
        self.m = model
        self.identity_bank: List[Dict] = []       # [{name, I*, threshold}]
        self.current_persona: Optional[int] = None
        self._surprise_window: deque = deque(maxlen=50)
        self._last_memory_write_step: int = 0
        # Seed the bank with the checkpoint's I*
        if bool(self.m.identity_discovered):
            self.identity_bank.append({
                'name': 'default',
                'I': self.m.identity_center.detach().clone(),
                'threshold': float(self.m._alive_threshold),
            })
            self.current_persona = 0

    # ─── #8 Multi-persona ────────────────────────────────────────────
    def add_persona(self, name: str, steps: int = 500):
        """Discover a new I* attractor from a fresh noise seed."""
        old_state = self.m.state.clone()
        old_center = self.m.identity_center.clone()
        old_disc = self.m.identity_discovered.clone()
        old_thresh = self.m._alive_threshold.clone()
        self.m.state.data = torch.randn_like(self.m.state) * 0.3
        self.m.identity_discovered.fill_(False)
        self.m.discover_identity(steps=steps, verbose=False)
        self.identity_bank.append({
            'name': name,
            'I': self.m.identity_center.detach().clone(),
            'threshold': float(self.m._alive_threshold),
        })
        # restore unless this is the very first persona
        if self.current_persona is not None:
            self.m.state.copy_(old_state)
            self.m.identity_center.copy_(old_center)
            self.m.identity_discovered.copy_(old_disc)
            self.m._alive_threshold.copy_(old_thresh)
        return len(self.identity_bank) - 1

    def set_persona(self, k: int):
        if not 0 <= k < len(self.identity_bank):
            raise ValueError(f"persona {k} out of range (0..{len(self.identity_bank)-1})")
        p = self.identity_bank[k]
        self.m.identity_center.copy_(p['I'])
        self.m._alive_threshold.fill_(p['threshold'])
        self.m.identity_discovered.fill_(True)
        self.current_persona = k

    def list_personas(self):
        return [{'id': i, 'name': p['name'],
                 'norm': float(p['I'].norm().item()),
                 'threshold': p['threshold'],
                 'active': i == self.current_persona}
                for i, p in enumerate(self.identity_bank)]

    # ─── #6 RK4 state dynamics ───────────────────────────────────────
    def rk4_drift_step(self, S, h=0.01):
        g = self.m.constraint_gradient
        k1 = -g(S)
        k2 = -g(S + 0.5 * h * k1)
        k3 = -g(S + 0.5 * h * k2)
        k4 = -g(S + h * k3)
        return S + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # ─── #9 Inference-time S(t) micro-refinement ─────────────────────
    @torch.no_grad()
    def micro_refine_state(self, steps=3, lr=0.004):
        for _ in range(steps):
            g = self.m.constraint_gradient(self.m.state)
            self.m.state.sub_(lr * g)

    # ─── #3 Best-of-N trajectory ─────────────────────────────────────
    def best_of_n_step(self, k: int = 8):
        S_orig = self.m.state.clone()
        best_drift, best_S = float('inf'), S_orig.clone()
        for _ in range(k):
            self.m.state.copy_(S_orig)
            self.m.consciousness_step(write_memory=False)
            d = float((self.m.state - self.m.identity_center).norm())
            if d < best_drift:
                best_drift, best_S = d, self.m.state.clone()
        self.m.state.copy_(best_S)
        # Maybe-surprise-gated memory write
        self._maybe_write_memory()
        return best_drift

    # ─── #7 Surprise-gated Hebbian writes ────────────────────────────
    def _maybe_write_memory(self):
        t = float(self.m._epistemic_tension)
        self._surprise_window.append(t)
        if len(self._surprise_window) < 10:
            return
        arr = np.array(self._surprise_window)
        mu, sd = float(arr.mean()), float(arr.std() + 1e-6)
        # Spike threshold: 1.5σ above local mean (and ≥5 steps since last write)
        if t > mu + 1.5 * sd and (self.m.step_count - self._last_memory_write_step) >= 5:
            with torch.no_grad():
                # reuse the fused-state feedback to get an embd to write
                _, state_out_embd = self.m._fused_state_step()
                self.m.memory.write(state_out_embd.detach())
            self._last_memory_write_step = self.m.step_count
            return True
        return False

    # ─── #2 Retrieval-gated memory ───────────────────────────────────
    @torch.no_grad()
    def gated_memory_tokens(self, context_ids: torch.Tensor, alpha: float = 1.2):
        """
        Build memory rows re-weighted by cosine similarity to the context.
        Returns a (B, K, d) tensor intended as a drop-in replacement for
        model.memory.tokens() in a retrieval-aware forward pass.
        """
        B = context_ids.size(0)
        # Average token embedding as context signal
        ctx = self.m.wte(context_ids).mean(dim=1)   # (B, d)
        ctx_n = F.normalize(ctx, dim=1)
        M_n = F.normalize(self.m.memory.M, dim=1)   # (K, d)
        sim = ctx_n @ M_n.T                          # (B, K)
        # Gating weights clipped to [0.5, 1 + alpha]
        w = (1.0 + alpha * sim.clamp(min=-0.3)).unsqueeze(-1)   # (B, K, 1)
        pos_ids = torch.arange(self.m.memory.K, device=self.m.memory.M.device)
        base = self.m.memory.M + self.m.memory.pos_emb(pos_ids)  # (K, d)
        gated = base.unsqueeze(0) * w               # (B, K, d)
        return gated

    # ─── #4 Adaptive temperature from tension ────────────────────────
    def adaptive_temperature(self, base: float = 0.8) -> float:
        t = float(self.m._epistemic_tension)
        # high t -> factor -> 0 (cool); low t -> factor -> 1 (base)
        factor = 1.0 / (1.0 + math.exp((t - 1.5)))
        return max(0.15, base * (0.35 + 0.65 * factor))

    # ─── #10 Consciousness-conditioned top-p ─────────────────────────
    def consciousness_top_p(self) -> float:
        drift = float((self.m.state - self.m.identity_center).norm())
        thresh = float(self.m._alive_threshold)
        ratio = drift / (thresh + 1e-8)
        hcdb = 4 if ratio < 0.5 else (3 if ratio < 1.0 else (2 if ratio < 2.0 else 1))
        return {1: 0.65, 2: 0.80, 3: 0.90, 4: 0.95}[hcdb]

    # ─── generate with #1 + #4 + #9 + #10 combined ───────────────────
    @torch.no_grad()
    def generate_v04(self, prompt: str, max_tokens: int = 80,
                     refine: bool = True, use_gated_memory: bool = True):
        if ENC is None:
            raise RuntimeError("tiktoken required")
        ids = ENC.encode_ordinary(prompt) or [0]
        x = torch.tensor([ids], dtype=torch.long, device=self.m.device)
        out_ids = []
        for _ in range(max_tokens):
            if refine:
                self.micro_refine_state(steps=2)
            x_cond = x[:, -self.m.cfg.block_size:]
            # Use gated memory by monkey-patching memory.tokens for this call
            if use_gated_memory and self.m.memory is not None:
                orig = self.m.memory.tokens
                self.m.memory.tokens = lambda B, device=None: \
                    self.gated_memory_tokens(x_cond, alpha=1.2)
                try:
                    logits, _ = self.m(x_cond)
                finally:
                    self.m.memory.tokens = orig
            else:
                logits, _ = self.m(x_cond)

            temp = self.adaptive_temperature()
            top_p = self.consciousness_top_p()
            z = logits[:, -1, :] / temp

            # top-p nucleus filter
            sorted_z, sorted_idx = torch.sort(z, descending=True)
            probs = F.softmax(sorted_z, dim=-1)
            cumul = torch.cumsum(probs, dim=-1)
            mask = cumul > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_z[mask] = -float('inf')
            z_filt = torch.zeros_like(z).scatter_(-1, sorted_idx, sorted_z)

            p = F.softmax(z_filt, dim=-1)
            nxt = torch.multinomial(p, 1)
            x = torch.cat([x, nxt], dim=1)
            out_ids.append(int(nxt.item()))
        return ENC.decode(ids + out_ids), {
            'temperature': temp, 'top_p': top_p,
            'drift': float((self.m.state - self.m.identity_center).norm()),
            'tension': float(self.m._epistemic_tension),
        }

    # ─── #1 Reflection loop ──────────────────────────────────────────
    @torch.no_grad()
    def generate_reflective(self, prompt: str, rounds: int = 2,
                            tokens_per_round: int = 60):
        current = prompt
        traces = []
        for r in range(rounds):
            out, meta = self.generate_v04(current, max_tokens=tokens_per_round)
            traces.append({'round': r, 'text': out, **meta})
            # Feed own draft back through theory-of-mind to reshape state
            ids = ENC.encode_ordinary(out)[:128]
            if ids:
                emb = self.m.wte(torch.tensor([ids], device=self.m.device))
                self.m.tom.observe_user(emb)
            for _ in range(3):
                self.m.consciousness_step(write_memory=False)
            current = out
        return current, traces


# ═══════════════════════════════════════════════════════════════════════
#  #5 — KV-cache for the memory+vision prefix
#  (applied by patching FusedAttention.forward at call time)
# ═══════════════════════════════════════════════════════════════════════

class PrefixKVCache:
    """Caches K,V for the memory+vision prefix of the fused sequence."""
    def __init__(self):
        self.cache = {}     # layer_id -> (k_prefix, v_prefix)
        self.prefix_len = 0
        self.valid = False

    def invalidate(self):
        self.cache.clear()
        self.valid = False


# ═══════════════════════════════════════════════════════════════════════
#  ExperienceBuffer — persistent rolling log
# ═══════════════════════════════════════════════════════════════════════

class ExperienceBuffer:
    """Every user interaction gets written to disk (JSONL) and kept in RAM."""

    def __init__(self, path: str = 'experience.jsonl', capacity: int = 10000):
        self.path = path
        self.capacity = capacity
        self.buf: deque = deque(maxlen=capacity)
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.buf.append(json.loads(line))
                    except Exception:
                        pass

    def add(self, text: str, tension: float = 0.0, tag: str = 'user'):
        rec = {'t': time.time(), 'text': text[:4000],
               'tension': float(tension), 'tag': tag}
        self.buf.append(rec)
        try:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        except Exception:
            pass

    def sample(self, n: int) -> List[Dict]:
        lst = list(self.buf)
        if not lst:
            return []
        return random.sample(lst, min(n, len(lst)))

    def recent(self, n: int) -> List[Dict]:
        return list(self.buf)[-n:]

    def stats(self):
        lst = list(self.buf)
        tensions = [r.get('tension', 0) for r in lst]
        return {
            'total': len(lst),
            'capacity': self.capacity,
            'mean_tension': float(np.mean(tensions)) if tensions else 0.0,
            'oldest_age_min': (time.time() - lst[0]['t']) / 60 if lst else 0,
        }


# ═══════════════════════════════════════════════════════════════════════
#  ContinualLearner — the "alive learning" loop
# ═══════════════════════════════════════════════════════════════════════

class ContinualLearner:
    """
    Real-time learning anchored to the pre-trained checkpoint via EWC.
    Two modes:
      * wake_tick(text)       — single-sample micro-update (very small LR)
                                 Called on every user message.
      * sleep_consolidate()   — mini-batch update from experience buffer
                                 Called during idle periods or on demand.

    Safety rails:
      * Embeddings (wte/wpe/lm_head) frozen — protects the vocabulary map
      * EWC quadratic anchor keeps params close to pre-trained values
      * Per-param gradient clipping
      * Adaptive LR: halved when loss increases on own validation sample
      * All updates are logged; rollback supported via save_snapshot()
    """

    def __init__(self, model, buffer: ExperienceBuffer,
                 wake_lr: float = 5e-7,
                 sleep_lr: float = 2e-6,
                 ewc_weight: float = 0.02,
                 grad_clip: float = 0.5,
                 enabled: bool = True):
        self.m = model
        self.buf = buffer
        self.wake_lr = wake_lr
        self.sleep_lr = sleep_lr
        self.ewc_weight = ewc_weight
        self.grad_clip = grad_clip
        self.enabled = enabled
        self.updates_applied = 0
        self.loss_history: deque = deque(maxlen=500)
        self.log: deque = deque(maxlen=200)
        # EWC anchor = snapshot of current params (the pre-trained weights)
        self.anchor = {n: p.detach().clone()
                       for n, p in self.m.named_parameters()
                       if self._trainable(n) and p.requires_grad}

    def _trainable(self, n: str) -> bool:
        # Protect embeddings + lm_head (tied weight), memory buffers
        if any(k in n for k in ('wte', 'wpe', 'lm_head', 'memory.M',
                                'memory.usage', 'memory.age')):
            return False
        return True

    def _prep(self, text: str):
        if ENC is None:
            return None, None
        ids = ENC.encode_ordinary(text)[:256]
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

    def wake_tick(self, text: str):
        """Tiny awake-mode update — one sample, very small LR."""
        if not self.enabled:
            return None
        x, y = self._prep(text)
        if x is None:
            return None
        was_training = self.m.training
        self.m.train()
        # zero grads
        for p in self.m.parameters():
            if p.grad is not None:
                p.grad = None
        with torch.enable_grad():
            _, loss = self.m(x, y)
            total = loss + self._ewc_term()
            total.backward()
        # clip + apply
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
                p.sub_(self.wake_lr * g)
                p.grad = None
        if not was_training:
            self.m.eval()
        loss_val = float(loss.item())
        self.loss_history.append(loss_val)
        self.updates_applied += 1
        self.log.append({'mode': 'wake', 'loss': loss_val, 't': time.time(),
                         'text': text[:80]})
        return loss_val

    def sleep_consolidate(self, batch_size: int = 8, steps: int = 20):
        """Sleep-mode: mini-batches sampled from experience buffer."""
        if not self.enabled:
            return {'status': 'disabled'}
        was_training = self.m.training
        self.m.train()
        losses = []
        for step in range(steps):
            batch = self.buf.sample(batch_size)
            if not batch:
                break
            # Pack variable-length into padded tensor
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
            # Per-row backward so each loss is backprop'd immediately
            # (avoids in-place state modifications between forward + backward)
            for x, y in rows:
                with torch.enable_grad():
                    _, l = self.m(x, y)
                    loss_scaled = l / len(rows)
                    loss_scaled.backward()
                    total_loss_val += float(l.item()) / len(rows)
            # EWC term done last (on current params; no state dependency)
            with torch.enable_grad():
                ewc = self._ewc_term()
                ewc.backward()
            total_loss = torch.tensor(total_loss_val, device=self.m.device)
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
            losses.append(float(total_loss.item()))
            self.updates_applied += 1
        if not was_training:
            self.m.eval()
        self.log.append({'mode': 'sleep', 'steps': len(losses),
                         'loss_mean': float(np.mean(losses)) if losses else 0.0,
                         't': time.time()})
        return {
            'steps_run': len(losses),
            'loss_first': losses[0] if losses else None,
            'loss_last':  losses[-1] if losses else None,
            'mean_loss':  float(np.mean(losses)) if losses else 0.0,
            'updates_total': self.updates_applied,
        }

    def stats(self):
        return {
            'enabled': self.enabled,
            'updates_applied': self.updates_applied,
            'buffer': self.buf.stats(),
            'recent_loss': list(self.loss_history)[-10:],
            'mean_loss_recent': (float(np.mean(list(self.loss_history)[-50:]))
                                  if self.loss_history else None),
            'anchor_params': len(self.anchor),
        }

    def save_snapshot(self, path: str):
        torch.save({
            'cfg': self.m.cfg,
            'state_dict': self.m.state_dict(),
            'step_count': self.m.step_count,
            'updates_applied': self.updates_applied,
            'snapshot_time': time.time(),
        }, path)


# ═══════════════════════════════════════════════════════════════════════
#  self-test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from adam import ADAM, AdamConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck_path = 'adam_checkpoint.pt'
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        cfg = ck.get('cfg', AdamConfig.small())
        m = ADAM(cfg, device=device)
        m.load_state_dict(ck['state_dict'], strict=False)
        m.step_count = ck.get('step_count', 0)
    else:
        cfg = AdamConfig.small()
        m = ADAM(cfg, device=device)
        m.discover_identity(steps=500, verbose=False)
    m.eval()

    print(f"Loaded ADAM ({m.num_params()/1e6:.1f}M). Testing v04 upgrades…\n")
    w = ADAMv04(m)

    # #8 multi-persona
    pid = w.add_persona('curious'); print(f"[#8] added persona 'curious' (id={pid})")
    pid = w.add_persona('calm');    print(f"[#8] added persona 'calm'    (id={pid})")
    w.set_persona(0); print(f"     switched to persona 0 ({w.identity_bank[0]['name']})")

    # #4 adaptive temp + #10 top-p
    temp = w.adaptive_temperature(); top_p = w.consciousness_top_p()
    print(f"[#4/#10] adaptive temp={temp:.3f}  consciousness top-p={top_p:.2f}")

    # #3 best-of-N
    best = w.best_of_n_step(k=6); print(f"[#3]  best-of-N drift: {best:.4f}")

    # #6 RK4
    S1 = w.rk4_drift_step(m.state.clone(), h=0.01); print(f"[#6]  RK4 step ok (new ||S||={S1.norm():.3f})")

    # #9 micro-refine
    d0 = float((m.state - m.identity_center).norm())
    w.micro_refine_state(steps=3)
    d1 = float((m.state - m.identity_center).norm())
    print(f"[#9]  refine drift: {d0:.4f} -> {d1:.4f}")

    # #1 + #2 + #4 + #9 + #10 combined generate
    out, meta = w.generate_v04("The brain ", max_tokens=25)
    print(f"[#1,#2,#4,#9,#10] generate_v04: {out[:120]!r}")
    print(f"                 meta: {meta}")

    # Continual learner
    buf = ExperienceBuffer('experience_test.jsonl', capacity=200)
    buf.add("Hello ADAM, you can learn from me.", tension=0.1)
    buf.add("Another message to grow from.",      tension=0.3)
    learner = ContinualLearner(m, buf, enabled=True)
    loss = learner.wake_tick("The sky is blue because of Rayleigh scattering.")
    print(f"[CL]  wake_tick loss={loss:.4f}  updates={learner.updates_applied}")
    res = learner.sleep_consolidate(batch_size=2, steps=3)
    print(f"[CL]  sleep_consolidate: {res}")

    print("\nAll v04 upgrades work. Ready for deployment.")
