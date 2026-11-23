r"""
kelly.py -- the Kelly criterion for repeated bets.

A bet wins `b` per unit staked with probability p and loses the stake with
probability q = 1 - p. Staking a fixed fraction f of current wealth, the
expected log-growth per bet is

    g(f) = p ln(1 + b f) + q ln(1 - f),         maximised at  f* = p - q/b.

After n bets with w wins, log-wealth is  w ln(1+bf) + (n-w) ln(1-f), and w is
Binomial(n, p), so wealth paths can be sampled EXACTLY from a binomial draw
(no need to simulate bet by bet).
"""

import math

import numpy as np
from scipy.stats import binom


def _validate_bet_parameters(p, b):
    if not np.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("b must be positive and finite")


def growth_rate(p, b, f):
    _validate_bet_parameters(p, b)
    f = np.asarray(f, dtype=float)
    if np.any(~np.isfinite(f)) or np.any((f < 0.0) | (f > 1.0)):
        raise ValueError("f must be finite and between 0 and 1")
    return p * np.log1p(b * f) + (1 - p) * np.log1p(-f)


def kelly_fraction(p, b):
    """Optimal fraction (0 if the bet has no edge)."""
    _validate_bet_parameters(p, b)
    return max(0.0, p - (1 - p) / b)


def simulate_log_wealth(p, b, f, n_bets, n_paths, seed=0):
    """Final log-wealth for each of n_paths paths (start wealth 1)."""
    _validate_bet_parameters(p, b)
    if not isinstance(n_bets, (int, np.integer)) or n_bets < 1:
        raise ValueError("n_bets must be a positive integer")
    if not isinstance(n_paths, (int, np.integer)) or n_paths < 2:
        raise ValueError("n_paths must be an integer >= 2")
    if not np.isfinite(f) or not 0.0 <= f < 1.0:
        raise ValueError("f must be in [0, 1) for finite log-wealth")
    rng = np.random.default_rng(seed)
    wins = rng.binomial(n_bets, p, size=n_paths)
    return wins * math.log1p(b * f) + (n_bets - wins) * math.log1p(-f)


def prob_end_below_start(p, b, f, n_bets):
    """Exact P(final wealth < 1): wins * ln(1+bf) + (n-wins) * ln(1-f) < 0."""
    _validate_bet_parameters(p, b)
    if not np.isfinite(f) or not 0.0 < f < 1.0:
        raise ValueError("f must be in (0, 1)")
    if not isinstance(n_bets, (int, np.integer)) or n_bets < 1:
        raise ValueError("n_bets must be a positive integer")
    a, c = math.log1p(b * f), -math.log1p(-f)          # gain per win, loss per loss (both > 0)
    # w*a < (n-w)*c  <=>  w < n*c/(a+c)
    threshold = n_bets * c / (a + c)
    k = math.ceil(threshold) - 1                       # largest integer strictly below threshold
    return float(binom.cdf(k, n_bets, p)) if k >= 0 else 0.0


def break_even_fraction(p, b, hi=0.999):
    """Smallest f > 0 with g(f) = 0 (growth turns negative beyond it)."""
    _validate_bet_parameters(p, b)
    if not np.isfinite(hi) or not 0.0 < hi < 1.0:
        raise ValueError("hi must be between 0 and 1")
    from scipy.optimize import brentq
    fs = kelly_fraction(p, b)
    if fs <= 0:
        return 0.0
    return brentq(lambda f: float(growth_rate(p, b, f)), fs, hi)
