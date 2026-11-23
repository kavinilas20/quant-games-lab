"""Optimal stopping: secretary problem and the red/black card game."""

import math

import numpy as np
import pytest

from src import stopping


@pytest.mark.parametrize("r", range(1, 7))
def test_secretary_formula_equals_enumeration_of_all_orderings(r):
    assert stopping.secretary_success_probability(r, 6) == pytest.approx(stopping.secretary_brute_force(r, 6))


def test_secretary_edge_cases():
    assert stopping.secretary_success_probability(1, 10) == pytest.approx(0.1)       # take the first: 1/n
    with pytest.raises(ValueError):
        stopping.secretary_success_probability(0, 10)


def test_secretary_best_cutoff_is_about_n_over_e_and_probability_about_one_over_e():
    r, p = stopping.secretary_best_cutoff(100)
    assert r == 38 and p == pytest.approx(0.3710, abs=1e-3)
    assert abs(r - 100 / math.e) < 1.5
    assert p > 1 / math.e                                    # finite-n value is slightly above the limit
    r1000, p1000 = stopping.secretary_best_cutoff(1000)
    assert abs(p1000 - 1 / math.e) < abs(p - 1 / math.e)     # and converges to it


def test_secretary_simulation_matches_exact():
    n_trials = 20_000
    exact = stopping.secretary_success_probability(37, 100)
    sim = stopping.secretary_simulate(37, 100, n_trials, seed=1)
    assert abs(sim - exact) < 4 * math.sqrt(exact * (1 - exact) / n_trials)


# ------------------------------ card game ------------------------------
def test_card_game_hand_computed_cases():
    V = stopping.card_game_values(3, 3)
    assert V[0, 0] == 0
    assert V[1, 0] == 1 and V[3, 0] == 3                    # only red cards left: take them all
    assert V[0, 3] == 0                                  # only black left: stop immediately
    assert V[1, 1] == pytest.approx(0.5)                     # 1/2 (+1 + 0) + 1/2 (-1 + 1)


def test_card_game_value_properties():
    V = stopping.card_game_values(26, 26)
    assert np.all(V >= 0)
    assert np.all(np.diff(V, axis=0) >= -1e-12)              # more red cards never hurts
    assert np.all(np.diff(V, axis=1) <= 1e-12)               # more black cards never helps
    assert V[26, 26] == pytest.approx(2.6245, abs=1e-3)
    assert V[26, 26] > 0                                     # strictly better than the 0 you get by stopping at once


def test_card_game_optimal_policy_simulation_matches_dp_value():
    V = stopping.card_game_values(26, 26)
    mean, se = stopping.card_game_simulate(26, 26, 8_000, seed=2, V=V)
    assert abs(mean - V[26, 26]) < 4 * se


def test_card_game_value_beats_naive_policies():
    """A fixed rule such as 'always draw the whole deck' earns exactly 0 on a balanced deck."""
    V = stopping.card_game_values(10, 10)
    assert V[10, 10] > 0
