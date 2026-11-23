r"""
ruin.py -- gambler's ruin.

Start with i units; each round win 1 with probability p, lose 1 with
probability q = 1 - p; stop on reaching 0 or N.

    P(reach N | start i) = i/N                          if p = 1/2
                         = (1 - r^i) / (1 - r^N),  r = q/p   otherwise
    E[duration]          = i (N - i)                     if p = 1/2
                         = i/(q-p) - (N/(q-p)) (1 - r^i)/(1 - r^N)   otherwise

`solve_linear` computes both by solving the defining linear equations
directly, as an independent check of the closed forms.
"""

import numpy as np


def _validate(i, N, p):
    if not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError("N must be an integer >= 2")
    if not isinstance(i, (int, np.integer)) or not 0 <= i <= N:
        raise ValueError("i must be an integer between 0 and N")
    if not np.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")
    if p in (0.0, 1.0):
        # Deterministic paths have simple boundary cases, but the expected duration
        # is still defined. Handle these explicitly in the public formulas below.
        return


def win_probability(i, N, p):
    _validate(i, N, p)
    if i == 0:
        return 0.0
    if i == N:
        return 1.0
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    if p == 0.5:
        return i / N
    r = (1 - p) / p
    return (1 - r ** i) / (1 - r ** N)


def expected_duration(i, N, p):
    _validate(i, N, p)
    if i in (0, N):
        return 0.0
    if p == 0.0:
        return float(i)
    if p == 1.0:
        return float(N - i)
    if p == 0.5:
        return i * (N - i)
    q = 1 - p
    r = q / p
    return i / (q - p) - (N / (q - p)) * (1 - r ** i) / (1 - r ** N)


def solve_linear(N, p):
    _validate(0, N, p)
    """Solve h_i = p h_{i+1} + q h_{i-1} (h_0=0, h_N=1) and d_i = 1 + p d_{i+1} + q d_{i-1} (d_0=d_N=0)."""
    q = 1 - p
    n = N - 1
    A = np.zeros((n, n))
    for k in range(n):
        A[k, k] = 1.0
        if k > 0:
            A[k, k - 1] = -q
        if k < n - 1:
            A[k, k + 1] = -p
    h = np.linalg.solve(A, np.r_[np.zeros(n - 1), p])      # boundary value h_N = 1 enters the last row
    d = np.linalg.solve(A, np.ones(n))
    return np.r_[0.0, h, 1.0], np.r_[0.0, d, 0.0]


def simulate(i, N, p, n_paths, seed=0):
    """Monte Carlo: returns (fraction reaching N, mean duration). Vectorised over paths."""
    _validate(i, N, p)
    if not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")
    rng = np.random.default_rng(seed)
    pos = np.full(n_paths, i)
    steps = np.zeros(n_paths, dtype=np.int64)
    alive = np.ones(n_paths, dtype=bool)
    while alive.any():
        move = np.where(rng.random(n_paths) < p, 1, -1)
        pos = np.where(alive, pos + move, pos)
        steps = steps + alive
        alive = alive & (pos > 0) & (pos < N)
    return float(np.mean(pos == N)), float(steps.mean())
