r"""
drills.py -- a mental-maths drill generator and a timed command-line session.

    python -m src.drills --n 20 --seed 1

Problem kinds: 3-digit sums, 2-digit products, percentages, fractions to two
decimals, squares up to 99, and expected-value one-liners. Answers are checked
within a tolerance (exact for integers). The generator is deterministic for a
seed, so a session can be repeated.
"""

import argparse
import time

import numpy as np

KINDS = ("sum", "product", "percent", "fraction", "square", "ev_dice")


def generate_problem(rng, kind):
    """Returns (question, answer, tolerance)."""
    if kind == "sum":
        a, b, c = (int(x) for x in rng.integers(100, 1000, size=3))
        return f"{a} + {b} + {c}", float(a + b + c), 0.0
    if kind == "product":
        a, b = (int(x) for x in rng.integers(11, 100, size=2))
        return f"{a} x {b}", float(a * b), 0.0
    if kind == "percent":
        p = int(rng.choice([5, 8, 12, 15, 17, 23, 35, 45, 65, 85]))
        base = int(rng.choice([40, 60, 80, 120, 150, 240, 360, 480, 640]))
        return f"{p}% of {base}", p * base / 100.0, 1e-9
    if kind == "fraction":
        a = int(rng.integers(1, 12))
        b = int(rng.choice([3, 6, 7, 9, 11, 12, 13]))
        return f"{a}/{b} to 2 decimals", round(a / b, 2), 0.005
    if kind == "square":
        a = int(rng.integers(12, 100))
        return f"{a}^2", float(a * a), 0.0
    if kind == "ev_dice":
        n = int(rng.integers(2, 9))
        return f"Expected sum of {n} fair dice", 3.5 * n, 1e-9
    raise ValueError(f"unknown kind {kind!r}")


def make_session(n, seed):
    rng = np.random.default_rng(seed)
    kinds = [KINDS[int(i)] for i in rng.integers(0, len(KINDS), size=n)]
    return [(kind, *generate_problem(rng, kind)) for kind in kinds]


def is_correct(guess, answer, tol):
    return abs(guess - answer) <= tol


def run_session(n=20, seed=0, input_fn=input, clock=time.perf_counter):
    problems = make_session(n, seed)
    correct, times = 0, []
    for i, (kind, question, answer, tol) in enumerate(problems, 1):
        start = clock()
        try:
            raw = input_fn(f"[{i}/{n}] {question} = ")
        except EOFError:                       # Ctrl-D ends the session early
            print("\nSession ended early.")
            break
        times.append(clock() - start)
        try:
            ok = is_correct(float(raw), answer, tol)
        except ValueError:
            ok = False
        correct += ok
        if not ok:
            print(f"   wrong: answer is {answer:g}")
    answered = len(times)
    print(f"\nScore {correct}/{answered}; mean time per question {np.mean(times) if times else 0:.1f}s")
    return correct, times


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_session(args.n, args.seed)
