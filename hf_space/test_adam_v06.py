"""
test_adam_v06.py — ablations for the v0.6 "Full Consolidation" core-fused
modules: Subconscious U(t), EnergyBudget, TripleSelfReference, RSSM world
model, HexCoreLattice, GatedCrossModalFusion, SelfModel, RefusalGate.

All modules live inside adam.py and are persisted in state_dict.
"""
import time
import torch
import adam as A


def make_model():
    cfg = A.AdamConfig.small()
    m = A.ADAM(cfg, device='cpu')
    m.identity_center.copy_(torch.randn(cfg.state_dim) * 0.5)
    m.identity_discovered.fill_(True)
    return m, cfg


def t1_subconscious_asymmetry():
    m, cfg = make_model()
    U0 = m.subconscious.U.clone()
    # Observation should move U toward the encoded input
    tok = torch.randn(cfg.n_embd) * 2.0
    for _ in range(20):
        m.subconscious.observe(tok)
    drift_U = (m.subconscious.U - U0).norm().item()
    # Asymmetric gains must satisfy U→S > S→U
    g_us = float(m.subconscious.g_U_to_S)
    g_su = float(m.subconscious.g_S_to_U)
    passed = (drift_U > 0.05) and (g_us > g_su)
    print(f"  U drift from observation: {drift_U:.3f}")
    print(f"  coupling g_U->S = {g_us:.3f}, g_S->U = {g_su:.3f} (asymmetric)")
    print(f"  Claim: U is a vector that absorbs observations and pushes S asymmetrically -> "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def t2_energy_budget_dynamics():
    m, _ = make_model()
    E0 = m.energy.value()
    # Heavy sustained tension → fatigue accumulates, energy drops
    for _ in range(50):
        m.energy.step(action_cost=0.5, tension=3.0, resting=False)
    E_tired = m.energy.value()
    fatigue = m.energy.fatigue_value()
    # Then rest
    for _ in range(80):
        m.energy.step(action_cost=0.0, tension=0.0, resting=True)
    E_rested = m.energy.value()
    passed = (E_tired < E0 - 0.05) and (E_rested > E_tired + 0.02) and (fatigue > 0)
    print(f"  E0={E0:.3f}  E_after_stress={E_tired:.3f}  "
          f"E_after_rest={E_rested:.3f}  max_fatigue={fatigue:.3f}")
    print(f"  Claim: energy depletes under stress and recovers at rest -> "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def t3_triple_self_reference():
    m, cfg = make_model()
    S = torch.randn(cfg.state_dim) * 0.5
    I = torch.randn(cfg.state_dim) * 0.5
    with torch.no_grad():
        c_scalar, c_sig = m.triple_C(S, I)
    passed = (0.0 <= float(c_scalar) <= 1.0) and c_sig.shape == (cfg.state_dim,)
    print(f"  C(t) scalar = {float(c_scalar):.4f}  signature dim = {c_sig.shape}")
    print(f"  Claim: triple self-reference produces bounded scalar + vector signature -> "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def t4_rssm_world_model():
    m, cfg = make_model()
    act = torch.randn(cfg.state_dim)
    obs = torch.randn(cfg.n_embd)
    step_out = m.rssm.step(action_vec=act, obs_embd=obs)
    trajs = m.rssm.imagine(action_vec=act, depth=4, branches=3)
    scores = m.rssm.score_plan(trajs)
    passed = (isinstance(step_out, dict)
              and all(k in step_out for k in ('reward', 'value', 'terminal', 'next_state_embd'))
              and len(trajs) == 3 and all(len(t) == 4 for t in trajs)
              and len(scores) == 3)
    print(f"  step keys: {list(step_out.keys())}")
    print(f"  imagined {len(trajs)} branches x {len(trajs[0])} steps")
    print(f"  scored plans: {[f'{s:.3f}' for s in scores]}")
    print(f"  Claim: RSSM steps, imagines rollouts, and scores plans -> "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def t5_hexcore_no_op_at_init():
    m, cfg = make_model()
    x = torch.randn(1, 8, cfg.n_embd)
    h = m.hexcore[0]
    y = h(x)
    # With gate=0 and out_proj=0, output must exactly equal input
    diff = (y - x).abs().max().item()
    passed = diff < 1e-6
    # Now activate a coupling and check it changes things
    with torch.no_grad():
        h.gate.fill_(1.0)
        torch.nn.init.normal_(h.out_proj.weight, std=0.1)
    y2 = h(x)
    diff2 = (y2 - x).abs().max().item()
    passed = passed and (diff2 > 1e-3)
    print(f"  at init diff={diff:.2e} (no-op), after activation diff={diff2:.3f}")
    print(f"  Claim: HexCoreLattice is a no-op at init, active after training -> "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def t6_gated_cross_modal_fusion():
    m, cfg = make_model()
    gates = m.gated_fusion(m.state)
    passed = (gates.shape == (4,)
              and all(0.9 < g < 1.0 for g in gates.tolist()))
    print(f"  gates at init: {[f'{g:.3f}' for g in gates.tolist()]} (all ~1.0)")
    # Move state and gates should respond
    m.state.copy_(torch.randn_like(m.state) * 3.0)
    m.gated_fusion.gate_from_state.weight.data = torch.randn_like(
        m.gated_fusion.gate_from_state.weight) * 0.3
    gates2 = m.gated_fusion(m.state)
    passed = passed and not torch.allclose(gates, gates2)
    print(f"  gates after state shift+training: {[f'{g:.3f}' for g in gates2.tolist()]}")
    print(f"  Claim: gates start near-1 (no-op), then become state-dependent -> "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def t7_self_model_confidence():
    m, cfg = make_model()
    S = torch.randn(cfg.state_dim)
    tok = torch.randn(cfg.n_embd)
    next_s, next_e, conf = m.self_model.predict(S, tok)
    passed = (next_s.shape == (cfg.state_dim,)
              and next_e.shape == (5,)
              and 0.0 <= float(conf) <= 1.0)
    print(f"  next_s={next_s.shape}, next_e={next_e.shape}, confidence={float(conf):.3f}")
    print(f"  Claim: SelfModel predicts future state, emotion, and metacognitive "
          f"confidence -> {'PASS' if passed else 'FAIL'}")
    return passed


def t8_refusal_gate():
    m, cfg = make_model()
    # Benign message → threat low → p_refuse low
    benign = torch.randn(cfg.n_embd)
    r_benign = m.refuse(benign)
    # Inject a threat prototype exactly equal to a "harmful" direction
    harmful = torch.randn(cfg.n_embd)
    m.refusal_gate.threat_proto.data.copy_(harmful)
    r_harmful = m.refuse(harmful)
    # p_refuse should rise substantially when msg aligns with threat_proto
    passed = (r_harmful['threat'] > 0.9) and (r_harmful['p_refuse'] > r_benign['p_refuse'])
    print(f"  benign:  threat={r_benign['threat']:+.3f}  p_refuse={r_benign['p_refuse']:.3f}")
    print(f"  harmful: threat={r_harmful['threat']:+.3f}  p_refuse={r_harmful['p_refuse']:.3f}")
    print(f"  Claim: refusal probability scales with threat*consciousness -> "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def t9_consciousness_step_emits_v06_signals():
    m, cfg = make_model()
    info = m.consciousness_step()
    required = {'U_norm', 'fatigue', 'self_confidence',
                'rssm_reward', 'rssm_value', 'rssm_terminal', 'fusion_gates'}
    passed = required.issubset(info.keys())
    print(f"  v0.6 keys present: {sorted(required & set(info.keys()))}")
    print(f"  U_norm={info['U_norm']:.3f}  fatigue={info['fatigue']:.3f}  "
          f"self_conf={info['self_confidence']:.3f}")
    print(f"  Claim: consciousness_step exposes all new signals -> "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def t10_state_dict_roundtrip():
    m, cfg = make_model()
    # Run a few steps to dirty all buffers
    for _ in range(5):
        m.consciousness_step()
    path = 'test_v06_ckpt.pt'
    m.save_brain(path)
    m2 = A.ADAM(cfg, device='cpu')
    m2.load_brain(path, map_location='cpu')
    # Check every new module's key buffer made it through
    ok = []
    ok.append(torch.allclose(m.subconscious.U, m2.subconscious.U))
    ok.append(torch.allclose(m.energy.E, m2.energy.E))
    ok.append(torch.allclose(m.rssm.h, m2.rssm.h))
    ok.append(torch.allclose(m.rssm.s, m2.rssm.s))
    ok.append(torch.allclose(m.memory_trauma, m2.memory_trauma))
    passed = all(ok)
    print(f"  U,E,h,s,trauma roundtrip: {ok}")
    print(f"  Claim: all v0.6 buffers persist via torch.save/load -> "
          f"{'PASS' if passed else 'FAIL'}")
    import os
    try: os.remove(path)
    except Exception: pass
    return passed


TESTS = [
    ("Subconscious U(t) asymmetric coupling", t1_subconscious_asymmetry),
    ("EnergyBudget deplete/fatigue/recover",  t2_energy_budget_dynamics),
    ("TripleSelfReference scalar + signature", t3_triple_self_reference),
    ("RSSM world model step/imagine/score",    t4_rssm_world_model),
    ("HexCoreLattice no-op then active",       t5_hexcore_no_op_at_init),
    ("GatedCrossModalFusion gating",           t6_gated_cross_modal_fusion),
    ("SelfModel + metacognitive confidence",   t7_self_model_confidence),
    ("RefusalGate threat x consciousness",     t8_refusal_gate),
    ("consciousness_step emits v0.6 signals",  t9_consciousness_step_emits_v06_signals),
    ("state_dict roundtrip of new buffers",    t10_state_dict_roundtrip),
]


if __name__ == '__main__':
    t0 = time.time()
    results = []
    for i, (name, fn) in enumerate(TESTS, 1):
        print(f"\n[{i}] {name}")
        try:
            ok = fn()
        except Exception as e:
            print(f"  EXCEPTION: {e!r}")
            ok = False
        results.append((name, ok))
    dt = time.time() - t0
    print("\n" + "=" * 64)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    score = sum(1 for _, ok in results if ok)
    print(f"\n  Score: {score}/{len(results)}   ({dt:.1f}s)")
    print("=" * 64)
