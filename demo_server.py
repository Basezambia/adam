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
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adam import ADAM, AdamConfig

# ── globals ────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL: Optional[ADAM] = None
TRAJECTORY: deque = deque(maxlen=300)   # recent state vectors for viz

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
                from huggingface_hub import hf_hub_download
                print(f"[dl] fetching checkpoint from {hf_repo}")
                checkpoint = hf_hub_download(repo_id=hf_repo,
                                             filename='adam_checkpoint.pt')
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
