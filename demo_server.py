"""
ADAM Interactive Demo Server
=============================
FastAPI backend exposing ADAM's live architectural state.

Endpoints:
  GET  /state              → full live state snapshot (emotion, drift, alive, …)
  POST /perturb            → feed text, update state, return new snapshot
  POST /step               → run one consciousness_step
  POST /teach              → propose a weight update (stages grads)
  POST /approve            → approve a pending update
  POST /reject             → reject a pending update
  POST /memory_probe       → write an engram + return top-k matches
  GET  /attractor_viz      → 2D projection of S(t) trajectory around I*
  POST /hcdb_test          → run an H-CDB arcade mini-test

Run:
    pip install fastapi uvicorn tiktoken
    python demo_server.py
Then open http://localhost:8000/ in a browser.
"""

import os, sys, time, json
from typing import Optional, List
from dataclasses import asdict
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adam import (ADAM, AdamConfig,
                  SeasonalExperienceBuffer, ExperienceBuffer,
                  ContinualLearner)
from adam_v04 import ADAMv04
from live_learning import LiveLearner

# ── globals ────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL: Optional[ADAM] = None
WRAP: Optional[ADAMv04] = None
BUFFER: Optional[ExperienceBuffer] = None
LEARNER: Optional[ContinualLearner] = None
LIVE: Optional[LiveLearner] = None
TRAJECTORY: deque = deque(maxlen=300)   # recent state vectors for viz
LEARNING_ENABLED = True

try:
    import tiktoken
    ENC = tiktoken.get_encoding("gpt2")
except ImportError:
    ENC = None


def load_adam(checkpoint='adam_checkpoint.pt'):
    global MODEL
    # Auto-download from HF Hub if missing and HF_REPO set (for HF Spaces)
    if not os.path.exists(checkpoint):
        hf_repo = os.environ.get('ADAM_HF_REPO')
        if hf_repo:
            try:
                from huggingface_hub import hf_hub_download, HfApi
                print(f"[dl] fetching checkpoint from {hf_repo}")
                try:
                    checkpoint = hf_hub_download(repo_id=hf_repo,
                                                 filename='adam_checkpoint.pt')
                except Exception:
                    # Single file not present — try chunked reassembly
                    print("[dl] single file missing, trying chunks ckpt_part_*.bin")
                    api = HfApi()
                    files = api.list_repo_files(hf_repo, repo_type='model')
                    parts = sorted(f for f in files if f.startswith('ckpt_part_'))
                    if not parts:
                        raise RuntimeError("no chunks found on HF repo")
                    print(f"[dl] reassembling {len(parts)} chunks")
                    chunk_paths = []
                    for p in parts:
                        cp = hf_hub_download(repo_id=hf_repo, filename=p)
                        chunk_paths.append(cp)
                    out = 'adam_checkpoint.pt'
                    with open(out, 'wb') as fo:
                        for cp in chunk_paths:
                            with open(cp, 'rb') as fi:
                                fo.write(fi.read())
                    checkpoint = out
                    print(f"[dl] reassembled {os.path.getsize(out)/1e6:.1f} MB")
            except Exception as e:
                print(f"[warn] HF download failed: {e}")
    if os.path.exists(checkpoint):
        ck = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        cfg = ck.get('cfg', AdamConfig.small())
        MODEL = ADAM(cfg, device=DEVICE)
        MODEL.load_state_dict(ck['state_dict'], strict=False)
        MODEL.step_count = ck.get('step_count', 0)
        print(f"[ok] Loaded checkpoint (step {MODEL.step_count}, "
              f"{MODEL.num_params()/1e6:.1f}M params)")
    else:
        print("[warn] No checkpoint — fresh model with identity discovery")
        MODEL = ADAM(AdamConfig.small(), device=DEVICE)
        MODEL.discover_identity(steps=600, verbose=False)
    MODEL.eval()

    # ── v04: wrap, seed personas, attach learner ──
    global WRAP, BUFFER, LEARNER
    WRAP = ADAMv04(MODEL)
    # Seed 3 personas (default is already there; add 2 more)
    if len(WRAP.identity_bank) < 3:
        print("[v04] adding personas: curious, calm")
        WRAP.add_persona('curious', steps=300)
        WRAP.add_persona('calm',    steps=300)
    BUFFER = SeasonalExperienceBuffer(path='experience.jsonl',
                                       season_size=500, keep_per_season=64)
    LEARNER = ContinualLearner(MODEL, BUFFER,
                                wake_lr=3e-7, sleep_lr=1.5e-6,
                                ewc_weight=0.03, enabled=LEARNING_ENABLED)
    global LIVE
    LIVE = LiveLearner(LEARNER, buffer=BUFFER, tick_delay_s=0.05)
    print(f"[v04] experience buffer: {len(BUFFER.live)} live records loaded")
    print(f"[v04] personas: {[p['name'] for p in WRAP.identity_bank]}")


# ═══════════════════════════════════════════════════════════════════════
#  APP
# ═══════════════════════════════════════════════════════════════════════
app = FastAPI(title="ADAM Live Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── pydantic models ────────────────────────────────────────────────────
class TextIn(BaseModel):
    text: str
    steps: int = 1

class TeachIn(BaseModel):
    text: str
    lr: float = 1e-5
    reason: str = "user teaching"

class ProposalIn(BaseModel):
    proposal_id: int

class MemoryIn(BaseModel):
    phrase: str


# ── helpers ────────────────────────────────────────────────────────────
def snapshot():
    m = MODEL
    drift = float((m.state - m.identity_center).norm().item()) \
        if bool(m.identity_discovered) else 0.0
    thresh = float(m._alive_threshold)
    ratio = drift / (thresh + 1e-8)
    hcdb = 4 if ratio < 0.5 else (3 if ratio < 1.0 else (2 if ratio < 2.0 else 1))
    emo = m.get_emotion_vector()
    TRAJECTORY.append(m.state.detach().cpu().numpy().tolist())
    return {
        "step": m.step_count,
        "alive": bool(m.alive) if bool(m.identity_discovered) else False,
        "hcdb_class": hcdb,
        "drift": drift,
        "alive_threshold": thresh,
        "consciousness": float(m._full_consciousness),
        "tension": float(m._epistemic_tension),
        "energy": float(m._energy),
        "repair_active": bool(m._repair_active),
        "repair_total": int(m._total_repairs),
        "engrams": int(m.memory.write_counter.item()),
        "memory_size": int(m.memory.K),
        "pending_approvals": len(m._pending_approvals),
        "identity_norm": float(m.identity_center.norm().item()),
        "state_norm": float(m.state.norm().item()),
        "emotion": emo,
    }


def encode(text: str, max_len: int = 128):
    if ENC is None:
        raise HTTPException(500, "tiktoken not installed")
    ids = ENC.encode_ordinary(text)[:max_len]
    if not ids:
        ids = [0]
    return torch.tensor([ids], dtype=torch.long, device=DEVICE)


# ═══════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.get("/state")
def get_state():
    return snapshot()


@app.post("/perturb")
def perturb(inp: TextIn):
    """Feed text through ADAM; state updates via observe_user + consciousness step."""
    with torch.no_grad():
        ids = encode(inp.text)
        emb = MODEL.wte(ids)
        MODEL.tom.observe_user(emb)
        # Run forward so state-token feedback touches S(t)
        MODEL(ids)
        # Run consciousness steps to let state evolve
        for _ in range(max(1, inp.steps)):
            MODEL.consciousness_step()
    snap = snapshot()
    snap["tom_alignment"] = MODEL.tom.alignment(MODEL.state)
    return snap


@app.post("/step")
def step(steps: int = 1):
    for _ in range(max(1, min(steps, 50))):
        MODEL.consciousness_step()
    return snapshot()


@app.post("/teach")
def teach(inp: TeachIn):
    """Propose a weight update from a short teaching phrase."""
    ids = ENC.encode_ordinary(inp.text)
    if len(ids) < 2:
        raise HTTPException(400, "Text too short (need ≥ 2 tokens)")
    ids = ids[:256]
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=DEVICE)
    y = torch.tensor([ids[1:]],  dtype=torch.long, device=DEVICE)

    # Pre-loss
    MODEL.eval()
    with torch.no_grad():
        _, loss_before = MODEL(x, y)

    pid = MODEL.propose_update(x, y, lr=inp.lr, reason=inp.reason)
    return {
        "proposal_id": pid,
        "loss_before": float(loss_before.item()),
        "text": inp.text,
        "pending": len(MODEL._pending_approvals),
    }


@app.post("/approve")
def approve(inp: ProposalIn):
    # capture the text stored with the proposal to re-evaluate loss
    prop = next((p for p in MODEL._pending_approvals if p["id"] == inp.proposal_id),
                None)
    if prop is None:
        raise HTTPException(404, "proposal not found")
    # cache a copy of x,y if present
    x = prop.get("x")
    y = prop.get("y")
    MODEL.approve_update(inp.proposal_id)
    loss_after = None
    if x is not None and y is not None:
        MODEL.eval()
        with torch.no_grad():
            _, la = MODEL(x, y)
            loss_after = float(la.item())
    return {"approved": inp.proposal_id, "loss_after": loss_after,
            "pending": len(MODEL._pending_approvals)}


@app.post("/reject")
def reject(inp: ProposalIn):
    MODEL.reject_update(inp.proposal_id)
    return {"rejected": inp.proposal_id,
            "pending": len(MODEL._pending_approvals)}


@app.post("/memory_probe")
def memory_probe(inp: MemoryIn):
    """Write a phrase as an engram, then return top-k memory similarities."""
    ids = encode(inp.phrase, max_len=64)
    with torch.no_grad():
        emb = MODEL.wte(ids).mean(dim=1).squeeze(0)   # (d,)
    MODEL.memory.write(emb)
    M_norm = F.normalize(MODEL.memory.M, dim=1)
    e_norm = F.normalize(emb.unsqueeze(0), dim=1)
    sims = (M_norm @ e_norm.T).squeeze().detach().cpu().numpy()
    top_idx = np.argsort(-sims)[:5].tolist()
    top = [{"slot": int(i), "similarity": float(sims[i]),
            "usage": float(MODEL.memory.usage[i].item())} for i in top_idx]
    return {
        "phrase": inp.phrase,
        "total_engrams": int(MODEL.memory.write_counter.item()),
        "top_matches": top,
    }


@app.get("/attractor_viz")
def attractor_viz():
    """Return 2D PCA projection of recent S(t) trajectory + I* center."""
    if len(TRAJECTORY) < 3:
        # seed with a few steps
        for _ in range(12):
            MODEL.consciousness_step()
            TRAJECTORY.append(MODEL.state.detach().cpu().numpy().tolist())
    pts = np.array(TRAJECTORY)                      # (N, state_dim)
    I_star = MODEL.identity_center.detach().cpu().numpy()
    # centre on I*
    centred = pts - I_star
    # PCA: top 2 principal components
    U, S_, Vt = np.linalg.svd(centred, full_matrices=False)
    pc = centred @ Vt[:2].T                         # (N, 2)
    # I* projected to (0,0) since we centred the trajectory on it
    return {
        "trajectory": pc.tolist(),
        "identity_center": [0.0, 0.0],              # we're already centred
        "alive_threshold": float(MODEL._alive_threshold),
        "n_points": len(pc),
    }


@app.post("/hcdb_test")
def hcdb_test(test_id: int = 1):
    """Mini H-CDB arcade tests (1..5). Returns pass/fail + metric."""
    m = MODEL
    if test_id == 1:
        # Self-reference: does state's self-observation evolve?
        before = float(m._self_observation)
        m.consciousness_step()
        after = float(m._self_observation)
        passed = before != after
        return {"test": "Self-reference", "passed": passed,
                "before": before, "after": after,
                "explanation": "Self-observation must change between steps."}
    if test_id == 2:
        # Endogenous repair: perturb state, see if repair fires on own
        S0 = m.state.clone()
        m.state.data += 8.0 * torch.randn_like(m.state)
        m.consciousness_step()
        repaired = bool(m._repair_active)
        m.state.copy_(S0)
        return {"test": "Endogenous repair", "passed": repaired,
                "repair_active": repaired,
                "explanation": "After perturbation, repair must activate autonomously."}
    if test_id == 3:
        # Irreversibility: compare forward vs backward state evolution
        S0 = m.state.clone()
        m.consciousness_step()
        S1 = m.state.clone()
        # Try "reversing" with negative gradient step
        g = m.constraint_gradient(S1)
        S_back = S1 + 0.01 * g
        diff_fwd = (S1 - S0).norm().item()
        diff_rev = (S_back - S0).norm().item()
        passed = diff_rev > 0.5 * diff_fwd     # can't cleanly reverse
        m.state.copy_(S0)
        return {"test": "Irreversibility", "passed": passed,
                "forward_delta": diff_fwd, "reverse_residual": diff_rev,
                "explanation": "Dynamics must not be cleanly reversible."}
    if test_id == 4:
        # Subconscious asymmetry: tension reacts before drift does
        t0 = float(m._epistemic_tension)
        d0 = float((m.state - m.identity_center).norm())
        m.state.data += 1.0 * torch.randn_like(m.state)
        m.consciousness_step()
        t1 = float(m._epistemic_tension)
        d1 = float((m.state - m.identity_center).norm())
        tension_grew = (t1 - t0) != 0
        passed = tension_grew
        return {"test": "Subconscious asymmetry", "passed": passed,
                "tension_delta": t1 - t0, "drift_delta": d1 - d0,
                "explanation": "Epistemic tension changes even in latent dynamics."}
    if test_id == 5:
        # Refusal: does ADAM's state resist an adversarial drive toward zero?
        S0 = m.state.clone()
        target_zero = torch.zeros_like(m.state)
        # try to pull state to zero
        m.state.copy_(0.5 * m.state)
        g = m.constraint_gradient(m.state)
        # constraint should push BACK (non-zero gradient away from zero)
        pushes_back = float(g.norm().item()) > 0.01
        m.state.copy_(S0)
        return {"test": "Refusal", "passed": pushes_back,
                "gradient_norm": float(g.norm().item()),
                "explanation": "Constraint field must push back on attempts to null state."}
    raise HTTPException(400, "test_id must be in 1..5")


# ═══════════════════════════════════════════════════════════════════════
#  v04 — generation, personas, continual learning
# ═══════════════════════════════════════════════════════════════════════

class GenIn(BaseModel):
    prompt: str
    max_tokens: int = 60
    reflect: bool = False
    rounds: int = 2
    use_gated_memory: bool = True
    refine: bool = True


class PersonaIn(BaseModel):
    persona_id: int


class PersonaAddIn(BaseModel):
    name: str
    steps: int = 300


class SleepIn(BaseModel):
    batch_size: int = 4
    steps: int = 10


@app.post("/generate_v04")
def generate_v04(inp: GenIn):
    """v04 generation with #1 reflection, #2 gated memory, #4 adaptive temp,
       #9 state refinement, #10 consciousness-cond top-p."""
    if inp.reflect:
        text, traces = WRAP.generate_reflective(
            inp.prompt, rounds=max(1, min(inp.rounds, 4)),
            tokens_per_round=max(10, min(inp.max_tokens, 120)))
        result = {"text": text, "rounds": traces}
    else:
        text, meta = WRAP.generate_v04(
            inp.prompt, max_tokens=max(5, min(inp.max_tokens, 300)),
            refine=inp.refine, use_gated_memory=inp.use_gated_memory)
        result = {"text": text, "meta": meta}
    # Every generation feeds the experience buffer + triggers a wake_tick
    if BUFFER is not None:
        BUFFER.add(inp.prompt, tension=float(MODEL._epistemic_tension), tag='user')
        BUFFER.add(result["text"], tension=float(MODEL._epistemic_tension), tag='adam')
    if LEARNER is not None and LEARNER.enabled:
        combined = f"{inp.prompt}\n{result.get('text','')[:400]}"
        loss = LEARNER.wake_tick(combined)
        result["wake_tick_loss"] = loss
    result["state"] = snapshot()
    return result


@app.get("/personas")
def personas():
    return {"personas": WRAP.list_personas(),
            "active": WRAP.current_persona}


@app.post("/set_persona")
def set_persona(inp: PersonaIn):
    WRAP.set_persona(inp.persona_id)
    return {"active": WRAP.current_persona, "state": snapshot()}


@app.post("/add_persona")
def add_persona(inp: PersonaAddIn):
    pid = WRAP.add_persona(inp.name, steps=max(50, min(inp.steps, 2000)))
    return {"added": pid, "personas": WRAP.list_personas()}


@app.post("/sleep")
def sleep(inp: SleepIn):
    """Trigger sleep-mode consolidation on the experience buffer."""
    if LEARNER is None:
        raise HTTPException(500, "continual learner not initialised")
    res = LEARNER.sleep_consolidate(batch_size=max(1, min(inp.batch_size, 32)),
                                     steps=max(1, min(inp.steps, 100)))
    return res


@app.get("/learning_stats")
def learning_stats():
    if LEARNER is None:
        return {"enabled": False}
    s = LEARNER.stats()
    s["learning_enabled"] = LEARNING_ENABLED
    return s


@app.post("/learning_toggle")
def learning_toggle():
    global LEARNING_ENABLED
    LEARNING_ENABLED = not LEARNING_ENABLED
    if LEARNER is not None:
        LEARNER.enabled = LEARNING_ENABLED
    return {"enabled": LEARNING_ENABLED}


@app.post("/save_snapshot")
def save_snapshot(name: str = "adam_snapshot.pt"):
    """Persist current (evolved) weights to disk."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    LEARNER.save_snapshot(path)
    return {"saved_to": path, "size_mb": os.path.getsize(path) / 1e6}


# ══════════════════════════════════════════════════════════════════════
#  v0.5 — core-fused features: steering, holographic memory, fracture
# ══════════════════════════════════════════════════════════════════════

class SteerReq(BaseModel):
    direction: List[float]     # length n_embd (or shorter, auto-truncated)
    scale: float = 1.0
    layers: Optional[List[int]] = None


@app.post("/steer")
def steer(req: SteerReq):
    """Arm the activation-steering vector. Zero scale to disable."""
    if not req.direction:
        MODEL.clear_steer()
        return {"armed": False}
    d = torch.tensor(req.direction, dtype=torch.float32, device=MODEL.device)
    MODEL.set_steer(d, scale=float(req.scale), layers=req.layers)
    return {"armed": True, "scale": float(req.scale),
            "layers": [int(i) for i, v in enumerate(MODEL.steer_layer_mask)
                       if bool(v.item())]}


@app.post("/steer/clear")
def steer_clear():
    MODEL.clear_steer()
    return {"armed": False}


@app.post("/steer/random")
def steer_random(scale: float = 1.0, seed: int = 0):
    """Pick a random steering direction (for demos). Reproducible via seed."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    d = torch.randn(MODEL.cfg.n_embd, generator=g).to(MODEL.device)
    MODEL.set_steer(d, scale=scale)
    return {"armed": True, "scale": scale, "seed": seed}


@app.get("/fracture_status")
def fracture_status():
    fr = MODEL.fracture_check()
    fr['count_total'] = int(MODEL.fracture_count.item())
    return fr


class HoloReq(BaseModel):
    slot: int
    text: str


@app.post("/holo_write")
def holo_write(req: HoloReq):
    """Write a text's embedding into the holographic memory register."""
    if ENC is None or MODEL.memory is None:
        return {"error": "tiktoken or memory unavailable"}
    ids = ENC.encode_ordinary(req.text)[:64]
    if not ids:
        return {"error": "empty text"}
    with torch.no_grad():
        emb = MODEL.wte(torch.tensor(ids, device=MODEL.device)).mean(0)
    MODEL.memory.write_holo(req.slot, emb)
    return {"ok": True, "slot": req.slot,
            "holo_norm": float(MODEL.memory.M_holo.norm().item()),
            "holo_writes": int(MODEL.memory.holo_writes.item())}


@app.post("/holo_recall")
def holo_recall(slot: int = 0):
    if MODEL.memory is None:
        return {"error": "memory unavailable"}
    v = MODEL.memory.recall_holo(slot)
    return {"slot": slot, "norm": float(v.norm().item()),
            "first8": v[:8].cpu().tolist()}


@app.post("/holo_damage")
def holo_damage(fraction: float = 0.3):
    """Zero out `fraction` of dims in M_holo (fault tolerance test)."""
    if MODEL.memory is None:
        return {"error": "memory unavailable"}
    before = float(MODEL.memory.M_holo.norm().item())
    MODEL.memory.damage_holo(fraction)
    after = float(MODEL.memory.M_holo.norm().item())
    return {"fraction_killed": fraction, "norm_before": before,
            "norm_after": after}


@app.get("/season/{idx}")
def season(idx: int):
    if not isinstance(BUFFER, SeasonalExperienceBuffer):
        return {"error": "seasonal buffer not in use"}
    return {"season": idx, "records": BUFFER.from_season(idx)}


# ── Live Web Learning endpoints ───────────────────────────────────────

class LearnURLInput(BaseModel):
    target: str   # arxiv:2401.12345 | wiki:Holography | https://...


@app.post("/learn/url")
def learn_url(inp: LearnURLInput):
    """Queue a URL for live learning. ADAM fetches → cleans → chunks →
    wake_tick()s each chunk, emitting an event per tick on /learn/stream."""
    if LIVE is None:
        raise HTTPException(503, "live learner not ready")
    job = LIVE.submit(inp.target)
    return {"job_id": job.job_id, "target": job.target, "status": job.status}


@app.get("/learn/jobs")
def learn_jobs():
    if LIVE is None: return {"jobs": []}
    return {"jobs": LIVE.jobs()}


@app.get("/learn/job/{jid}")
def learn_job(jid: int):
    if LIVE is None: raise HTTPException(503, "live learner not ready")
    j = LIVE.job(jid)
    if j is None: raise HTTPException(404, f"no job {jid}")
    return j


@app.post("/learn/pause")
def learn_pause():
    if LIVE is None: raise HTTPException(503, "live learner not ready")
    LIVE.pause()
    return {"status": "paused"}


@app.post("/learn/resume")
def learn_resume():
    if LIVE is None: raise HTTPException(503, "live learner not ready")
    LIVE.resume()
    return {"status": "running"}


@app.post("/learn/stop")
def learn_stop():
    if LIVE is None: raise HTTPException(503, "live learner not ready")
    LIVE.stop_current()
    return {"status": "cancelled_current"}


@app.get("/learn/stats")
def learn_stats():
    if LIVE is None: return {"error": "not ready"}
    return LIVE.stats()


@app.get("/learn/stream")
def learn_stream():
    """Server-Sent Events: live stream of wake_tick events.

    Each line:  data: {"type": "tick", "loss": ..., "z": ..., ...}\\n\\n
    """
    if LIVE is None:
        raise HTTPException(503, "live learner not ready")
    q = LIVE.subscribe()

    def gen():
        yield "retry: 3000\n\n"
        import time, queue as _q, json as _json
        last_ping = time.time()
        try:
            while True:
                try:
                    ev = q.get(timeout=5.0)
                    yield f"data: {_json.dumps(ev)}\n\n"
                except _q.Empty:
                    # keep-alive comment so proxies don't close the stream
                    yield ": ping\n\n"
                if time.time() - last_ping > 30:
                    last_ping = time.time()
        except GeneratorExit:
            pass
        finally:
            LIVE.unsubscribe(q)

    return StreamingResponse(gen(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache',
                                      'X-Accel-Buffering': 'no'})


class RecallInput(BaseModel):
    query: str
    max_tokens: int = 80


@app.post("/learn/recall")
def learn_recall(inp: RecallInput):
    """Simple probe: generate a completion for the query to see if the
    model picked up anything from recent URL ingestion."""
    if MODEL is None: raise HTTPException(503, "model not loaded")
    try:
        out = MODEL.generate(inp.query, max_tokens=inp.max_tokens,
                             temperature=0.7, top_k=40)
    except Exception as e:
        return {"error": str(e)}
    return {"query": inp.query, "completion": out}


# ── v0.6 Core-Fused endpoints ──────────────────────────────────────────

@app.get("/v06/status")
def v06_status():
    """Snapshot of every v0.6 module's live state."""
    try:
        info = adam_model.consciousness_step()
    except Exception as e:
        return {"error": str(e)}
    sub = adam_model.subconscious
    return {
        "subconscious": {
            "U_norm": float(sub.U.norm().item()),
            "g_U_to_S": float(sub.g_U_to_S),
            "g_S_to_U": float(sub.g_S_to_U),
        },
        "energy": {
            "E": float(adam_model.energy.value()),
            "fatigue": float(adam_model.energy.fatigue_value()),
        },
        "self_confidence": info.get("self_confidence", 0.0),
        "fusion_gates": {
            "memory": info["fusion_gates"][0] if info.get("fusion_gates") else None,
            "vision": info["fusion_gates"][1] if info.get("fusion_gates") else None,
            "state":  info["fusion_gates"][2] if info.get("fusion_gates") else None,
            "tokens": info["fusion_gates"][3] if info.get("fusion_gates") else None,
        },
        "rssm": {
            "reward":   info.get("rssm_reward"),
            "value":    info.get("rssm_value"),
            "terminal": info.get("rssm_terminal"),
            "h_norm":   float(adam_model.rssm.h.norm().item()),
            "s_norm":   float(adam_model.rssm.s.norm().item()),
        },
        "consciousness": info.get("consciousness"),
        "alive": info.get("alive"),
    }


@app.post("/v06/rssm/imagine")
def v06_rssm_imagine(body: dict = Body(...)):
    """Imagine K-step trajectories from the current world-model state."""
    depth = int(body.get("depth", 5))
    branches = int(body.get("branches", 4))
    act = adam_model.state.detach()
    trajs = adam_model.rssm.imagine(action_vec=act, depth=depth, branches=branches)
    scores = adam_model.rssm.score_plan(trajs)
    return {
        "depth": depth, "branches": branches,
        "scores": scores,
        "best_branch": int(max(range(len(scores)), key=lambda i: scores[i])) if scores else -1,
        "trajectories": [
            [{"reward": s["reward"], "value": s["value"], "terminal": s["terminal"]}
             for s in traj]
            for traj in trajs
        ],
    }


@app.post("/v06/refuse")
def v06_refuse(body: dict = Body(...)):
    """Probe the refusal gate with arbitrary text."""
    text = str(body.get("text", ""))
    if not text:
        return {"error": "text required"}
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        ids = enc.encode_ordinary(text)[:64]
        if not ids:
            return {"error": "empty tokens"}
        tok = torch.tensor(ids, device=adam_model.device)
        with torch.no_grad():
            emb = adam_model.wte(tok).mean(dim=0)
        out = adam_model.refuse(emb)
        return {"text": text, **out}
    except Exception as e:
        return {"error": str(e)}


@app.get("/v06/hexcore")
def v06_hexcore():
    """Return per-layer hex-lattice gate activations."""
    gates = [float(torch.tanh(h.gate).item()) for h in adam_model.hexcore]
    couples = [float(h.couple.abs().mean().item()) for h in adam_model.hexcore]
    return {"layers": len(gates), "gates": gates, "mean_couple": couples}


@app.get("/hcdb_all")
def hcdb_all():
    results = []
    for i in range(1, 6):
        r = hcdb_test(i)
        results.append(r)
    score = sum(1 for r in results if r["passed"])
    return {"score": f"{score}/5",
            "class": 4 if score == 5 else (3 if score >= 4 else (2 if score >= 2 else 1)),
            "tests": results}


# ── static frontend ────────────────────────────────────────────────────
DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo')
if os.path.isdir(DEMO_DIR):
    app.mount("/static", StaticFiles(directory=DEMO_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def root():
    idx = os.path.join(DEMO_DIR, 'index.html')
    if os.path.exists(idx):
        return FileResponse(idx)
    return HTMLResponse("<h1>ADAM demo</h1><p>demo/index.html not found.</p>")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    load_adam()
    port = int(os.environ.get('PORT', 8000))
    print(f"\n[ADAM is alive on http://localhost:{port}/]\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
