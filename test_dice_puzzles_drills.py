"""Dice market game, expectation puzzles, and the drill generator."""

import math

import numpy as np
import pytest

from src import dice_market as dm
from src import drills, puzzles


# ------------------------------- dice market -------------------------------
@pytest.mark.parametrize("n,k", [(2, 1), (3, 1), (3, 2), (4, 2)])
@pytest.mark.parametrize("mu", [0.1, 0.3, 0.7])
@pytest.mark.parametrize("w", [0.0, 0.45, 1.0, 2.2])
def test_exact_profit_equals_full_enumeration(n, k, mu, w):
    assert dm.expected_profit(n, k, mu, w) == pytest.approx(dm.expected_profit_bruteforce(n, k, mu, w), abs=1e-12)


def test_known_break_even_when_every_informed_trade_happens():
    # k=1: |d - 3.5| is 0.5, 1.5 or 2.5 (mean 1.5). For w < 0.5 the informed always trade:
    # (1-mu) w = mu (1.5 - w)  ->  w = 1.5 mu.
    assert dm.break_even_half_spread(3, 1, 0.3) == pytest.approx(1.5 * 0.3)
    assert dm.expected_profit(3, 1, 0.3, 0.45) == pytest.approx(0.0, abs=1e-12)


def test_break_even_depends_on_dice_seen_not_on_dice_in_total():
    assert dm.break_even_half_spread(3, 1, 0.3) == pytest.approx(dm.break_even_half_spread(7, 1, 0.3))


def test_break_even_increases_with_informed_share_and_with_information():
    mus = [0.05, 0.1, 0.2, 0.4, 0.6]
    w = [dm.break_even_half_spread(3, 1, m) for m in mus]
    assert all(a < b for a, b in zip(w, w[1:]))
    assert dm.break_even_half_spread(3, 2, 0.3) > dm.break_even_half_spread(3, 1, 0.3)
    assert dm.break_even_half_spread(3, 1, 0.0) == 0.0


def test_profit_is_zero_at_break_even_and_positive_above():
    w_star = dm.break_even_half_spread(4, 2, 0.4)
    assert dm.expected_profit(4, 2, 0.4, w_star) == pytest.approx(0.0, abs=1e-9)
    assert dm.expected_profit(4, 2, 0.4, w_star + 0.3) > 0
    assert dm.expected_profit(4, 2, 0.4, max(w_star - 0.3, 0.0)) < 0


@pytest.mark.parametrize("w", [0.3, 1.0, 2.0])
def test_simulation_matches_exact(w):
    mean, se = dm.simulate(3, 1, 0.3, w, 400_000, seed=4)
    assert abs(mean - dm.expected_profit(3, 1, 0.3, w)) < 4 * se


# --------------------------------- puzzles ---------------------------------
def test_coupon_collector_small_cases_by_hand():
    assert puzzles.coupon_collector_mean(1) == 1.0
    assert puzzles.coupon_collector_mean(2) == pytest.approx(3.0)           # 1 + 2
    assert puzzles.coupon_collector_variance(2) == pytest.approx(2.0)       # geometric(1/2) has variance 2


def test_coupon_collector_simulation_matches_exact():
    mean, var, se = puzzles.coupon_collector_simulate(20, 20_000, seed=1)
    assert abs(mean - puzzles.coupon_collector_mean(20)) < 4 * se
    assert var == pytest.approx(puzzles.coupon_collector_variance(20), rel=0.06)


@pytest.mark.parametrize("k,expected", [(1, 3.5), (2, 161 / 36), (3, 6 - 225 / 216)])
def test_expected_max_of_dice(k, expected):
    assert puzzles.expected_max_of_dice(k) == pytest.approx(expected)
    mean, se = puzzles.expected_max_of_dice_simulate(k, 400_000, seed=2)
    assert abs(mean - expected) < 4 * se


# --------------------------------- drills ---------------------------------
def test_session_is_deterministic_and_seed_dependent():
    assert drills.make_session(15, 3) == drills.make_session(15, 3)
    assert drills.make_session(15, 3) != drills.make_session(15, 4)


def test_arithmetic_answers_are_correct():
    rng = np.random.default_rng(0)
    for _ in range(300):
        q, a, _ = drills.generate_problem(rng, "sum")
        assert a == eval(q)
        q, a, _ = drills.generate_problem(rng, "product")
        assert a == eval(q.replace("x", "*"))
        q, a, _ = drills.generate_problem(rng, "square")
        assert a == eval(q.replace("^", "**"))
        q, a, _ = drills.generate_problem(rng, "percent")
        p, base = q.split("% of ")
        assert a == pytest.approx(float(p) * float(base) / 100)
        q, a, _ = drills.generate_problem(rng, "fraction")
        num, den = q.split(" ")[0].split("/")
        assert a == round(int(num) / int(den), 2)


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        drills.generate_problem(np.random.default_rng(0), "calculus")


def test_scripted_session_scores_correctly(capsys):
    problems = drills.make_session(10, 7)
    answers = iter(str(a) for _, _, a, _ in problems)
    correct, times = drills.run_session(10, 7, input_fn=lambda prompt: next(answers))
    assert correct == 10 and len(times) == 10
    wrong = iter(["abc"] * 10)
    correct, _ = drills.run_session(10, 7, input_fn=lambda prompt: next(wrong))
    assert correct == 0



def test_dice_market_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        dm.expected_profit(3, 4, 0.3, 0.5)
    with pytest.raises(ValueError):
        dm.expected_profit(3, 1, 1.5, 0.5)
    with pytest.raises(ValueError):
        dm.simulate(3, 1, 0.3, 0.5, 1)
