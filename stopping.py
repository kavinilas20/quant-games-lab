r"""
stopping.py -- two optimal-stopping problems.

1. Secretary problem. n candidates arrive in random order and each must be
   accepted or rejected on the spot. Rule: reject the first r-1, then accept
   the first candidate better than all seen so far. Success = you picked the
   overall best.
        P_success(r, n) = ((r-1)/n) * sum_{j=r}^{n} 1/(j-1)      (r >= 2),  1/n for r = 1.
   The best r is about n/e and the best success probability tends to 1/e.

2. Red/black card game. A shuffled deck with R red and B black cards; you
   reveal cards one at a time, gain +1 for red and -1 for black, and may stop
   whenever you like (keeping the running total). Dynamic programme on the
   remaining counts:
        V(r, b) = max( 0,  r/(r+b) (1 + V(r-1, b)) + b/(r+b) (-1 + V(r, b-1)) ).
   The optimal policy continues exactly when the continuation value is positive.
"""

import numpy as np


# ----------------------------- secretary problem -----------------------------
def secretary_success_probability(r, n):
    if r < 1 or r > n:
        raise ValueError("need 1 <= r <= n")
    if r == 1:
        return 1.0 / n
    return (r - 1) / n * sum(1.0 / (j - 1) for j in range(r, n + 1))


def secretary_best_cutoff(n):
    probs = [secretary_success_probability(r, n) for r in range(1, n + 1)]
    r_star = int(np.argmax(probs)) + 1
    return r_star, probs[r_star - 1]


def secretary_simulate(r, n, n_trials, seed=0):
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n_trials):
        ranks = rng.permutation(n)                  # larger = better
        best_seen = ranks[: r - 1].max() if r > 1 else -1
        chosen = ranks[-1]                           # forced to take the last if none qualifies
        for x in ranks[r - 1:]:
            if x > best_seen:
                chosen = x
                break
        wins += chosen == n - 1
    return wins / n_trials


def secretary_brute_force(r, n):
    """Exact success probability by enumerating all n! orderings (tiny n only)."""
    from itertools import permutations
    wins = total = 0
    for perm in permutations(range(n)):
        best_seen = max(perm[: r - 1]) if r > 1 else -1
        chosen = perm[-1]
        for x in perm[r - 1:]:
            if x > best_seen:
                chosen = x
                break
        wins += chosen == n - 1
        total += 1
    return wins / total


# ------------------------------- card game --------------------------------
def card_game_values(R, B):
    """Table V[r][b] for 0 <= r <= R, 0 <= b <= B."""
    V = np.zeros((R + 1, B + 1))
    for r in range(R + 1):
        for b in range(B + 1):
            if r + b == 0:
                continue
            cont = 0.0
            if r > 0:
                cont += r / (r + b) * (1 + V[r - 1, b])
            if b > 0:
                cont += b / (r + b) * (-1 + V[r, b - 1])
            V[r, b] = max(0.0, cont)
    return V


def card_game_simulate(R, B, n_trials, seed=0, V=None):
    """Play the optimal policy on random decks; returns (mean total, standard error)."""
    rng = np.random.default_rng(seed)
    V = card_game_values(R, B) if V is None else V
    totals = np.empty(n_trials)
    deck = np.array([1] * R + [-1] * B)
    for t in range(n_trials):
        rng.shuffle(deck)
        r, b, total = R, B, 0
        for card in deck:
            if r + b == 0:
                break
            cont = 0.0
            if r > 0:
                cont += r / (r + b) * (1 + V[r - 1, b])
            if b > 0:
                cont += b / (r + b) * (-1 + V[r, b - 1])
            if cont <= 1e-12:                       # continuing is no better than stopping
                break
            total += card
            if card == 1:
                r -= 1
            else:
                b -= 1
        totals[t] = total
    return float(totals.mean()), float(totals.std(ddof=1) / np.sqrt(n_trials))
