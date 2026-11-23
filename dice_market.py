r"""
dice_market.py -- a market-making game with an informed counterparty.

The "asset" pays the SUM of n fair dice. You quote a bid and an ask. A
counterparty arrives and may buy at your ask or sell at your bid:

    * with probability 1 - mu the counterparty is UNINFORMED: buys or sells
      with equal probability, regardless of price;
    * with probability mu the counterparty is INFORMED: it has seen k of the
      dice, and trades only if that makes money in expectation (buys if
      E[sum | seen dice] > ask, sells if E[sum | seen dice] < bid).

You quote symmetrically around the unconditional mean m = 3.5 n with
half-spread w. Your expected profit per arrival is exact by enumeration:

    uninformed:  +w                          (random side, fair value m)
    informed:    w - |E[sum|seen] - m|       when the informed trader trades,
                 0                            otherwise.

So the break-even half-spread is the w that balances the two, the same
adverse-selection logic as Glosten-Milgrom (see the market-making project).
"""

import itertools
import math

import numpy as np

FACE_MEAN = 3.5


def _validate(n, k, mu=None, w=None):
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= n:
        raise ValueError("k must be an integer between 1 and n")
    if mu is not None and (not np.isfinite(mu) or not 0.0 <= mu <= 1.0):
        raise ValueError("mu must be between 0 and 1")
    if w is not None and (not np.isfinite(w) or w < 0.0):
        raise ValueError("w must be non-negative and finite")


def informed_signal_distribution(n, k):
    _validate(n, k)
    """Distribution of |E[sum | k seen dice] - 3.5 n| = |sum of seen - 3.5 k|, as (values, probs)."""
    values = {}
    for seen in itertools.product(range(1, 7), repeat=k):
        dev = abs(sum(seen) - FACE_MEAN * k)
        values[dev] = values.get(dev, 0.0) + 6.0 ** (-k)
    items = sorted(values.items())
    return np.array([v for v, _ in items]), np.array([p for _, p in items])


def expected_profit(n, k, mu, w):
    _validate(n, k, mu, w)
    """Exact expected profit per arrival for half-spread w."""
    dev, prob = informed_signal_distribution(n, k)
    informed = float(np.sum(prob * np.where(dev > w, w - dev, 0.0)))
    return (1 - mu) * w + mu * informed


def break_even_half_spread(n, k, mu, tol=1e-12):
    _validate(n, k, mu)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be positive and finite")
    """Smallest w >= 0 with expected_profit = 0 (profit is non-decreasing in w beyond that point)."""
    lo, hi = 0.0, FACE_MEAN * k + 1.0
    if expected_profit(n, k, mu, 0.0) >= 0:
        return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if expected_profit(n, k, mu, mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def expected_profit_bruteforce(n, k, mu, w):
    _validate(n, k, mu, w)
    """Independent check: enumerate every outcome of all n dice, trader type and side."""
    m = FACE_MEAN * n
    total = 0.0
    for dice in itertools.product(range(1, 7), repeat=n):
        p_dice = 6.0 ** (-n)
        value = sum(dice)
        cond = sum(dice[:k]) + FACE_MEAN * (n - k)            # E[sum | first k dice]
        bid, ask = m - w, m + w
        # uninformed: buy (we sell at ask) or sell (we buy at bid), equally likely
        total += p_dice * (1 - mu) * (0.5 * (ask - value) + 0.5 * (value - bid))
        # informed
        if cond > ask:
            total += p_dice * mu * (ask - value)
        elif cond < bid:
            total += p_dice * mu * (value - bid)
    return total


def simulate(n, k, mu, w, n_trades, seed=0):
    """Monte Carlo of the same game; returns (mean profit per trade, standard error)."""
    _validate(n, k, mu, w)
    if not isinstance(n_trades, (int, np.integer)) or n_trades < 2:
        raise ValueError("n_trades must be an integer >= 2")
    rng = np.random.default_rng(seed)
    dice = rng.integers(1, 7, size=(n_trades, n))
    value = dice.sum(axis=1).astype(float)
    cond = dice[:, :k].sum(axis=1) + FACE_MEAN * (n - k)
    m = FACE_MEAN * n
    bid, ask = m - w, m + w
    informed = rng.random(n_trades) < mu
    uninformed_buys = rng.random(n_trades) < 0.5
    buys = np.where(informed, cond > ask, uninformed_buys)
    sells = np.where(informed, cond < bid, ~uninformed_buys)
    profit = np.where(buys, ask - value, 0.0) + np.where(sells, value - bid, 0.0)
    return float(profit.mean()), float(profit.std(ddof=1) / math.sqrt(n_trades))
