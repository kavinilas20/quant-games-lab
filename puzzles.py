r"""
puzzles.py -- classic expectation puzzles, each with an exact answer and a simulation.

Coupon collector: n equally likely coupon types, draw with replacement until
you hold all of them. If T is the number of draws,
    E[T] = n H_n,    Var[T] = n^2 sum_{k=1}^n 1/k^2  -  n H_n,
with H_n the n-th harmonic number. (T is a sum of independent geometric
waiting times, one for each new coupon.)

Maximum of k dice: E[max] = sum_{m=1}^{6} P(max >= m) = sum_{m=1}^{6} (1 - ((m-1)/6)^k).
"""

import numpy as np


def harmonic(n):
    return float(sum(1.0 / k for k in range(1, n + 1)))


def coupon_collector_mean(n):
    return n * harmonic(n)


def coupon_collector_variance(n):
    return n * n * sum(1.0 / (k * k) for k in range(1, n + 1)) - n * harmonic(n)


def coupon_collector_simulate(n, n_trials, seed=0):
    rng = np.random.default_rng(seed)
    draws = np.empty(n_trials)
    for t in range(n_trials):
        seen = np.zeros(n, dtype=bool)
        count = remaining = 0
        remaining = n
        while remaining:
            c = rng.integers(n)
            count += 1
            if not seen[c]:
                seen[c] = True
                remaining -= 1
        draws[t] = count
    return float(draws.mean()), float(draws.var(ddof=1)), float(draws.std(ddof=1) / np.sqrt(n_trials))


def expected_max_of_dice(k):
    return float(sum(1.0 - ((m - 1) / 6.0) ** k for m in range(1, 7)))


def expected_max_of_dice_simulate(k, n_trials, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 7, size=(n_trials, k)).max(axis=1)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(n_trials))
