"""Sudoku generator + backtracking solver + text formatting for ADAM.

Format used for training:
    "sudoku puzzle: <81 chars, . = blank> solution: <81 chars>"

Row-major 9x9.
"""
from __future__ import annotations
import random
from typing import List, Optional, Tuple

N = 9


def _find_empty(board: List[List[int]]) -> Optional[Tuple[int, int]]:
    for r in range(N):
        for c in range(N):
            if board[r][c] == 0:
                return r, c
    return None


def _is_valid(board: List[List[int]], r: int, c: int, v: int) -> bool:
    for i in range(N):
        if board[r][i] == v or board[i][c] == v:
            return False
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if board[i][j] == v:
                return False
    return True


def solve(board: List[List[int]]) -> bool:
    spot = _find_empty(board)
    if spot is None:
        return True
    r, c = spot
    nums = list(range(1, 10))
    random.shuffle(nums)
    for v in nums:
        if _is_valid(board, r, c, v):
            board[r][c] = v
            if solve(board):
                return True
            board[r][c] = 0
    return False


def _count_solutions(board: List[List[int]], limit: int = 2) -> int:
    spot = _find_empty(board)
    if spot is None:
        return 1
    r, c = spot
    total = 0
    for v in range(1, 10):
        if _is_valid(board, r, c, v):
            board[r][c] = v
            total += _count_solutions(board, limit - total)
            board[r][c] = 0
            if total >= limit:
                return total
    return total


def generate_full() -> List[List[int]]:
    board = [[0] * N for _ in range(N)]
    solve(board)
    return board


def make_puzzle(clues: int = 36) -> Tuple[List[List[int]], List[List[int]]]:
    """Return (puzzle, solution). `clues` = number of givens (81 - blanks)."""
    solution = generate_full()
    puzzle = [row[:] for row in solution]
    cells = [(r, c) for r in range(N) for c in range(N)]
    random.shuffle(cells)
    removed = 0
    target_blanks = 81 - clues
    for r, c in cells:
        if removed >= target_blanks:
            break
        saved = puzzle[r][c]
        puzzle[r][c] = 0
        if _count_solutions([row[:] for row in puzzle], limit=2) != 1:
            puzzle[r][c] = saved
        else:
            removed += 1
    return puzzle, solution


def board_to_str(board: List[List[int]], blank: str = ".") -> str:
    return "".join(str(v) if v else blank for row in board for v in row)


def str_to_board(s: str) -> List[List[int]]:
    s = s.strip().replace(" ", "").replace("\n", "")
    assert len(s) == 81, f"need 81 chars, got {len(s)}"
    flat = [0 if ch in ".0_-" else int(ch) for ch in s]
    return [flat[i * 9:(i + 1) * 9] for i in range(9)]


def solve_str(puzzle_str: str) -> Optional[str]:
    board = str_to_board(puzzle_str)
    if solve(board):
        return board_to_str(board)
    return None


def pretty(board_or_str) -> str:
    if isinstance(board_or_str, str):
        b = str_to_board(board_or_str)
    else:
        b = board_or_str
    lines = []
    for r in range(9):
        if r and r % 3 == 0:
            lines.append("------+-------+------")
        row = []
        for c in range(9):
            if c and c % 3 == 0:
                row.append("|")
            row.append(str(b[r][c]) if b[r][c] else ".")
        lines.append(" ".join(row))
    return "\n".join(lines)


def _candidates(board, r, c):
    if board[r][c] != 0:
        return set()
    used = set()
    for i in range(9):
        used.add(board[r][i]); used.add(board[i][c])
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            used.add(board[i][j])
    return set(range(1, 10)) - used


def _row_set(board, r): return {board[r][c] for c in range(9) if board[r][c]}
def _col_set(board, c): return {board[r][c] for r in range(9) if board[r][c]}
def _box_set(board, r, c):
    br, bc = 3 * (r // 3), 3 * (c // 3)
    return {board[i][j] for i in range(br, br+3) for j in range(bc, bc+3) if board[i][j]}


def solve_with_steps(puzzle_str: str, max_steps: int = 81):
    """Step-by-step logical solver. Returns list of step dicts:
      {row, col, value, strategy, row_has, col_has, box_has, why}
    Uses naked-single + hidden-single + fallback backtrack.
    """
    board = str_to_board(puzzle_str)
    steps = []
    made_progress = True
    while made_progress and len(steps) < max_steps:
        made_progress = False
        # naked single: cell has exactly one candidate
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    cand = _candidates(board, r, c)
                    if len(cand) == 1:
                        v = cand.pop()
                        row_has = sorted(_row_set(board, r))
                        col_has = sorted(_col_set(board, c))
                        box_has = sorted(_box_set(board, r, c))
                        board[r][c] = v
                        steps.append({
                            "row": r, "col": c, "value": v,
                            "strategy": "naked single",
                            "row_has": row_has, "col_has": col_has, "box_has": box_has,
                            "why": (f"row {r+1} contains {row_has}, col {c+1} contains {col_has}, "
                                    f"box contains {box_has}. Only {v} is left for ({r+1},{c+1})."),
                        })
                        made_progress = True
                        break
            if made_progress:
                break
        if made_progress:
            continue
        # hidden single: a digit can only go in one cell in a row/col/box
        found = False
        for unit_kind in ("row", "col", "box"):
            for idx in range(9):
                for v in range(1, 10):
                    positions = []
                    if unit_kind == "row":
                        for c in range(9):
                            if board[idx][c] == 0 and v in _candidates(board, idx, c):
                                positions.append((idx, c))
                    elif unit_kind == "col":
                        for r in range(9):
                            if board[r][idx] == 0 and v in _candidates(board, r, idx):
                                positions.append((r, idx))
                    else:
                        br, bc = 3 * (idx // 3), 3 * (idx % 3)
                        for r in range(br, br+3):
                            for c in range(bc, bc+3):
                                if board[r][c] == 0 and v in _candidates(board, r, c):
                                    positions.append((r, c))
                    if len(positions) == 1:
                        r, c = positions[0]
                        row_has = sorted(_row_set(board, r))
                        col_has = sorted(_col_set(board, c))
                        box_has = sorted(_box_set(board, r, c))
                        board[r][c] = v
                        steps.append({
                            "row": r, "col": c, "value": v,
                            "strategy": f"hidden single ({unit_kind})",
                            "row_has": row_has, "col_has": col_has, "box_has": box_has,
                            "why": (f"{v} can only fit at ({r+1},{c+1}) in this {unit_kind} "
                                    f"— every other empty cell already has {v} blocked."),
                        })
                        made_progress = True
                        found = True
                        break
                if found: break
            if found: break
        if found:
            continue
    # backtrack to finish if needed
    if _find_empty(board) is not None:
        solve(board)  # completes remaining without step logging
    return steps, board_to_str(board)


def training_example(clues: int = 36) -> str:
    puzzle, solution = make_puzzle(clues=clues)
    return f"sudoku puzzle: {board_to_str(puzzle)} solution: {board_to_str(solution)}"


def training_example_cot(clues: int = 50, max_steps: int = 20) -> Optional[str]:
    """Chain-of-thought sudoku example. The model learns to emit naked/hidden
    singles one at a time, with the board state it sees implicit in history.

    Format:
        P <81chars>
        S (r,c)=v
        S (r,c)=v
        ...
        = <81chars>

    Easy puzzles (high clue count) only — so the solve is mostly
    naked/hidden singles that a small LM can actually learn.
    Returns None if puzzle cannot be solved with pure logical steps
    within max_steps.
    """
    puzzle, solution = make_puzzle(clues=clues)
    ps = board_to_str(puzzle)
    steps, final = solve_with_steps(ps, max_steps=max_steps)
    if final != board_to_str(solution) or len(steps) == 0:
        return None
    parts = [f"P {ps}"]
    for s in steps:
        parts.append(f"S ({s['row']+1},{s['col']+1})={s['value']}")
    parts.append(f"= {final}")
    return "\n".join(parts)


def stream(n: int = 10000, clues_range: Tuple[int, int] = (30, 45)):
    for _ in range(n):
        clues = random.randint(*clues_range)
        yield training_example(clues=clues)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for ex in stream(n=n):
        p, s = ex.split(" solution: ")
        print(p)
        print("solution:", s)
        print()
