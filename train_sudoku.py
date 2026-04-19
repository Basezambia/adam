"""Teach ADAM sudoku by streaming generated puzzles through wake_tick.

Usage:
    python train_sudoku.py --steps 20000 --clues 34 --save-every 1000

Runs on GPU if available. Uses all CPU cores for data generation while GPU
trains. EWC anchor keeps ADAM's prior skills (language, identity) intact.
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import multiprocessing as mp
from queue import Empty

import torch

# make adam importable as __main__ for pickle compat
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adam import ADAM, AdamConfig, ContinualLearner, SeasonalExperienceBuffer
from sudoku_dataset import training_example, make_puzzle, board_to_str, solve_str


def _producer(q: mp.Queue, stop: mp.Event, clues_min: int, clues_max: int):
    """Background workers generate examples into a queue."""
    import random
    while not stop.is_set():
        clues = random.randint(clues_min, clues_max)
        try:
            q.put(training_example(clues=clues), timeout=1.0)
        except Exception:
            pass


def load_model(ckpt_path: str, device: str):
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ck.get("cfg", AdamConfig())
        model = ADAM(cfg, device=device)
        model.load_state_dict(ck["state_dict"], strict=False)
        model.step_count = ck.get("step_count", 0)
        print(f"[ok] loaded {ckpt_path} (step {model.step_count}, "
              f"{model.num_params()/1e6:.1f}M params)")
    else:
        print(f"[warn] {ckpt_path} not found — starting from fresh tiny model")
        model = ADAM(AdamConfig(), device=device)
    model.eval()
    return model


def eval_accuracy(model, learner, n: int = 16, max_new: int = 81) -> float:
    """Sample a few puzzles, greedy-decode solution, measure cell accuracy."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    model.eval()
    total, correct = 0, 0
    for _ in range(n):
        puzzle, solution = make_puzzle(clues=38)
        prompt = f"sudoku puzzle: {board_to_str(puzzle)} solution: "
        sol_str = board_to_str(solution)
        ids = enc.encode_ordinary(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=model.device)
        with torch.no_grad():
            for _ in range(max_new):
                logits, _ = model(x)
                nxt = int(torch.argmax(logits[0, -1]).item())
                x = torch.cat([x, torch.tensor([[nxt]], device=model.device)], dim=1)
                if x.size(1) > 350:
                    break
        out = enc.decode(x[0].tolist())
        after = out.split("solution:", 1)[-1].strip()
        digits = "".join(ch for ch in after if ch.isdigit())[:81]
        for i, ch in enumerate(digits):
            total += 1
            if i < len(sol_str) and ch == sol_str[i]:
                correct += 1
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="adam_checkpoint.pt")
    ap.add_argument("--out", default="adam_checkpoint.pt",
                    help="where to save (overwrites by default)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--clues-min", type=int, default=32)
    ap.add_argument("--clues-max", type=int, default=45)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--wake-lr", type=float, default=5e-6,
                    help="override wake_lr (default 10x normal for skill learning)")
    ap.add_argument("--ewc", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device} | torch threads: {torch.get_num_threads()}")
    if device == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    torch.set_num_threads(max(1, mp.cpu_count() // 2))

    model = load_model(args.ckpt, device)
    buffer = SeasonalExperienceBuffer(path="experience_sudoku.jsonl",
                                       season_size=500, keep_per_season=64)
    learner = ContinualLearner(model, buffer,
                                wake_lr=args.wake_lr,
                                wake_lr_fracture=args.wake_lr * 5,
                                ewc_weight=args.ewc,
                                enabled=True)

    # start producer workers
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue(maxsize=256)
    stop = ctx.Event()
    producers = [ctx.Process(target=_producer,
                             args=(q, stop, args.clues_min, args.clues_max),
                             daemon=True)
                 for _ in range(args.workers)]
    for p in producers:
        p.start()
    print(f"[producers] {args.workers} workers generating puzzles")

    t0 = time.time()
    losses = []
    try:
        for step in range(1, args.steps + 1):
            try:
                text = q.get(timeout=5.0)
            except Empty:
                print("[warn] no data from producers — slow generation")
                continue
            res = learner.wake_tick(text)
            if res and "loss" in res:
                losses.append(res["loss"])
            if step % 50 == 0:
                avg = sum(losses[-50:]) / max(len(losses[-50:]), 1)
                rate = step / (time.time() - t0)
                print(f"[step {step:>6}/{args.steps}] "
                      f"loss={avg:.3f} | {rate:.1f} it/s | queue={q.qsize()}")
            if args.eval_every and step % args.eval_every == 0:
                acc = eval_accuracy(model, learner, n=8)
                print(f"[eval  {step:>6}] cell-accuracy={acc*100:.1f}%")
            if step % args.save_every == 0:
                torch.save({
                    "state_dict": model.state_dict(),
                    "cfg": model.cfg,
                    "step_count": model.step_count,
                }, args.out)
                print(f"[save] {args.out} @ step {step}")
    except KeyboardInterrupt:
        print("\n[interrupt] saving before exit")
    finally:
        torch.save({
            "state_dict": model.state_dict(),
            "cfg": model.cfg,
            "step_count": model.step_count,
        }, args.out)
        print(f"[done] saved {args.out}")
        stop.set()
        for p in producers:
            p.terminate()


if __name__ == "__main__":
    mp.freeze_support()
    main()
