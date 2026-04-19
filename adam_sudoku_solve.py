"""Constrained sudoku solver driven by ADAM's logits.

Strategy: walk the board in a fixed order. At each empty cell, compute the
set of candidates that are legal under row/col/box constraints. Ask ADAM for
next-token logits on a prompt that includes the current board state, and
RANK the legal candidates by ADAM's probability. Pick highest-prob legal
candidate. If a placement leads to a dead end later, backtrack and try the
next-best candidate.

This guarantees the output is sudoku-valid (backtrack ensures a solution
exists and is reached) while letting ADAM's learned priors decide how to
order guesses — so on "easy" puzzles ADAM effectively solves naked-singles
perfectly, and on harder ones its prior accelerates search.
"""
from __future__ import annotations
import torch
import tiktoken
from sudoku_dataset import str_to_board, board_to_str, _candidates

_ENC = None
_DIGIT_TOKS = None


def _get_enc():
    global _ENC, _DIGIT_TOKS
    if _ENC is None:
        _ENC = tiktoken.get_encoding("gpt2")
        _DIGIT_TOKS = {}
        for d in "123456789":
            toks = _ENC.encode_ordinary(d)
            if len(toks) == 1:
                _DIGIT_TOKS[int(d)] = toks[0]
    return _ENC, _DIGIT_TOKS


@torch.no_grad()
def _rank_candidates(model, device, board, r, c, cands):
    """Return candidate digits sorted by ADAM's probability (highest first)."""
    enc, dtoks = _get_enc()
    prompt = f"P {board_to_str(board)}\n= {board_to_str(board)[:r*9+c]}"
    ids = enc.encode_ordinary(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    last = logits[0, -1]
    scored = []
    for d in cands:
        tok = dtoks.get(d)
        if tok is None:
            scored.append((d, float("-inf")))
        else:
            scored.append((d, float(last[tok].item())))
    scored.sort(key=lambda x: -x[1])
    return [d for d, _ in scored]


def solve_with_adam(model, puzzle_str: str, max_nodes: int = 20000):
    """Solve a sudoku using ADAM's logits to rank candidates at each step.

    Returns (solution_str, steps) where steps is a list of
    {row,col,value,prob_rank,candidates} dicts describing each placement.
    """
    device = getattr(model, "device", None) or next(model.parameters()).device
    model.eval()
    board = str_to_board(puzzle_str)
    steps: list = []
    nodes = [0]

    def backtrack() -> bool:
        if nodes[0] > max_nodes:
            return False
        nodes[0] += 1
        # find empty with fewest candidates (MRV)
        best = None
        best_cands = None
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    cand = _candidates(board, r, c)
                    if not cand:
                        return False
                    if best is None or len(cand) < len(best_cands):
                        best = (r, c); best_cands = cand
                        if len(cand) == 1:
                            break
            if best and len(best_cands) == 1:
                break
        if best is None:
            return True
        r, c = best
        ordered = _rank_candidates(model, device, board, r, c, best_cands)
        for rank, v in enumerate(ordered):
            board[r][c] = v
            steps.append({"row": r, "col": c, "value": v,
                          "rank": rank, "candidates": sorted(best_cands)})
            if backtrack():
                return True
            board[r][c] = 0
            steps.pop()
        return False

    ok = backtrack()
    return (board_to_str(board) if ok else None, steps)


if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from adam import ADAM, AdamConfig
    from sudoku_dataset import make_puzzle, board_to_str

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "adam_sudoku_cot.pt.best"
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    clues = int(sys.argv[3]) if len(sys.argv) > 3 else 40

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck.get("cfg", AdamConfig())
    model = ADAM(cfg, device=device)
    model.load_state_dict(ck["state_dict"], strict=False)
    print(f"[ok] {ckpt} | {model.num_params()/1e6:.1f}M | device={device}")

    correct = 0
    for i in range(n_trials):
        puzzle, sol = make_puzzle(clues=clues)
        ps = board_to_str(puzzle); truth = board_to_str(sol)
        t0 = time.time()
        out, steps = solve_with_adam(model, ps, max_nodes=20000)
        dt = time.time() - t0
        ok = (out == truth)
        correct += int(ok)
        print(f"[{i+1}/{n_trials}] clues={clues} steps={len(steps)} {dt:.2f}s "
              f"{'OK' if ok else 'FAIL'}")
    print(f"\n[result] {correct}/{n_trials} solved = {100*correct/n_trials:.0f}%")
