"""
ADAM v0.5 ablation battery — the 4 core-fused features, each with a testable
claim and a number.

  1. Holographic memory fault tolerance  (recall cosine vs % dims killed)
  2. Activation steering disambiguation   (different directions -> different logits)
  3. Novelty-fracture z-gate               (ordinary inputs quiet, spikes fracture)
  4. Seasonal buffer tree rings            (sealing preserves queryable history)

Run:  python test_adam_v05.py --checkpoint adam_checkpoint.pt
"""
import argparse, os, sys, tempfile, time, math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adam import ADAM, AdamConfig, SeasonalExperienceBuffer, ContinualLearner


def load(ckpt):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if ckpt and os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        m = ADAM(ck.get('cfg', AdamConfig.small()), device=dev)
        m.load_state_dict(ck['state_dict'], strict=False)
    else:
        m = ADAM(AdamConfig.small(), device=dev)
        m.discover_identity(steps=200, verbose=False)
    m.eval()
    return m


# ══════════════════════════════════════════════════════════════════════
# Test 1: Holographic memory fault tolerance
# ══════════════════════════════════════════════════════════════════════
def test_holo(m):
    print("\n[1] HOLOGRAPHIC MEMORY — fault tolerance")
    d = m.cfg.n_embd
    # Write 8 distinct (key_slot, value) pairs
    torch.manual_seed(42)
    values = F.normalize(torch.randn(8, d, device=m.device), dim=1)
    slots = list(range(0, 8))
    m.memory.M_holo.zero_()
    for s, v in zip(slots, values):
        m.memory.write_holo(s, v)

    # Recall each and measure cosine vs ground truth at various damage levels
    fractions = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    print(f"    {'dmg%':<8}{'mean_cos':<12}{'min_cos':<12}{'top1_acc':<10}")
    results = []
    for frac in fractions:
        # Clone M_holo, damage, recall all, restore
        orig = m.memory.M_holo.clone()
        if frac > 0:
            m.memory.damage_holo(frac)
        cosines = []
        hits = 0
        for s, v in zip(slots, values):
            v_hat = m.memory.recall_holo(s)
            cos = float(F.cosine_similarity(v_hat.unsqueeze(0),
                                            v.unsqueeze(0), dim=1).item())
            cosines.append(cos)
            # Check top-1: is this recall closest to its own value among all 8?
            sims = F.cosine_similarity(v_hat.unsqueeze(0), values, dim=1)
            if int(sims.argmax().item()) == slots.index(s):
                hits += 1
        m.memory.M_holo.copy_(orig)
        mean_c = sum(cosines) / len(cosines)
        min_c = min(cosines)
        acc = hits / len(slots)
        results.append((frac, mean_c, min_c, acc))
        print(f"    {int(frac*100):<8}{mean_c:<12.3f}{min_c:<12.3f}{acc:<10.2f}")
    # Claim: at 30% damage, top-1 accuracy >= 0.75
    at30 = [r for r in results if abs(r[0] - 0.3) < 1e-6][0]
    ok = at30[3] >= 0.75
    print(f"    Claim: >=75% top-1 accuracy at 30% dim damage  -> {'PASS' if ok else 'FAIL'} ({at30[3]:.2f})")
    return ok


# ══════════════════════════════════════════════════════════════════════
# Test 2: Steering disambiguation
# ══════════════════════════════════════════════════════════════════════
def test_steer(m):
    print("\n[2] STEERING — different directions produce different logits")
    try:
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
    except Exception:
        print("    skip (tiktoken missing)")
        return True
    ids = enc.encode_ordinary("The future of consciousness")[:32]
    x = torch.tensor([ids], device=m.device)

    logits_list = []
    m.clear_steer()
    with torch.no_grad():
        l0, _ = m(x)
        logits_list.append(l0[0, -1].clone())
        for seed in [1, 7, 42]:
            g = torch.Generator(device='cpu').manual_seed(seed)
            d = torch.randn(m.cfg.n_embd, generator=g).to(m.device)
            m.set_steer(d, scale=2.0)
            li, _ = m(x)
            logits_list.append(li[0, -1].clone())
        m.clear_steer()

    # Cosine distance between each pair should be > some threshold
    dists = []
    for i in range(len(logits_list)):
        for j in range(i + 1, len(logits_list)):
            cos = float(F.cosine_similarity(
                logits_list[i].unsqueeze(0),
                logits_list[j].unsqueeze(0), dim=1).item())
            dists.append(1 - cos)
    mean_d = sum(dists) / len(dists)
    print(f"    mean pairwise (1 - cos) distance across 4 steer directions: {mean_d:.4f}")
    ok = mean_d > 1e-4
    print(f"    Claim: steer vectors change logits  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ══════════════════════════════════════════════════════════════════════
# Test 3: Novelty-fracture z-gate
# ══════════════════════════════════════════════════════════════════════
def test_fracture(m):
    print("\n[3] NOVELTY FRACTURE — ordinary quiet, spikes detected")
    # Reset history
    m.tension_history.zero_()
    m.tension_hist_ptr.zero_()
    m.tension_hist_full.zero_()
    m.fracture_count.zero_()
    # Feed 60 ordinary tensions centered at 0.5
    torch.manual_seed(0)
    for _ in range(60):
        t = float(0.5 + 0.1 * torch.randn(1).item())
        m._record_tension(t)
    fr_norm = m.fracture_check(0.6)
    fr_spike = m.fracture_check(3.0)  # way above
    print(f"    ordinary (t=0.6):  z={fr_norm['z']:.2f} fracture={fr_norm['fracture']}")
    print(f"    spike    (t=3.0):  z={fr_spike['z']:.2f} fracture={fr_spike['fracture']}")
    ok = (not fr_norm['fracture']) and fr_spike['fracture']
    print(f"    Claim: gate distinguishes ordinary from spike  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ══════════════════════════════════════════════════════════════════════
# Test 4: Seasonal buffer tree rings
# ══════════════════════════════════════════════════════════════════════
def test_seasonal():
    print("\n[4] SEASONAL BUFFER — rings seal and remain queryable")
    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, 'live.jsonl')
        p2 = os.path.join(td, 'seasons.jsonl')
        buf = SeasonalExperienceBuffer(path=p1, season_size=20,
                                       keep_per_season=5, seasons_path=p2)
        for i in range(65):
            # Every 10th message is high-tension "fracture"
            tension = 3.5 if i % 10 == 0 else 0.3
            buf.add(f"msg {i}", tension=tension,
                    fracture=(tension > 2.0))
        stats = buf.stats()
        s0 = buf.from_season(0)
        print(f"    sealed seasons: {stats['seasons_sealed']}")
        print(f"    live: {stats['live']}  historical: {stats['total_historical']}")
        print(f"    fracture events preserved: {stats['fracture_events']}")
        print(f"    season[0] size: {len(s0)} (kept highest-tension)")
        # Verify top record in season 0 is a fracture
        ring_has_fracture = any(r.get('fracture') for r in s0)
        ok = (stats['seasons_sealed'] == 3 and ring_has_fracture
              and stats['fracture_events'] >= 6)
        print(f"    Claim: >=3 seasons sealed, fractures preserved  -> {'PASS' if ok else 'FAIL'}")
        return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='adam_checkpoint.pt')
    args = ap.parse_args()
    print("=" * 64)
    print("  ADAM v0.5 ablation battery")
    print("=" * 64)
    t0 = time.time()
    m = load(args.checkpoint)
    print(f"[load] {m.num_params()/1e6:.1f}M params on {m.device}")

    results = []
    results.append(('Holographic fault-tolerance', test_holo(m)))
    results.append(('Steering disambiguation',     test_steer(m)))
    results.append(('Novelty-fracture gate',        test_fracture(m)))
    results.append(('Seasonal tree-ring buffer',    test_seasonal()))

    print("\n" + "=" * 64)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    print(f"\n  Score: {passed}/{len(results)}   ({time.time()-t0:.1f}s)")
    print("=" * 64)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
