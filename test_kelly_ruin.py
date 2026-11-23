"""Kelly criterion and gambler's ruin: closed forms vs independent calculations and simulation."""

import math

import numpy as np
import pytest

from src import kelly, ruin


# ---------------------------------- Kelly ----------------------------------
@pytest.mark.parametrize("p,b,expected", [(0.55, 1.0, 0.10), (0.4, 2.0, 0.10), (0.6, 1.0, 0.20), (0.5, 1.0, 0.0), (0.3, 1.0, 0.0)])
def test_kelly_fraction_formula(p, b, expected):
    assert kelly.kelly_fraction(p, b) == pytest.approx(expected)


@pytest.mark.parametrize("p,b", [(0.55, 1.0), (0.4, 2.5), (0.7, 0.5)])
def test_kelly_fraction_maximises_growth_rate(p, b):
    fs = kelly.kelly_fraction(p, b)
    grid = np.linspace(0.0, 0.95, 20001)
    assert grid[np.argmax(kelly.growth_rate(p, b, grid))] == pytest.approx(fs, abs=1e-4)


def test_growth_is_zero_at_break_even_and_negative_beyond():
    p, b = 0.55, 1.0
    fb = kelly.break_even_fraction(p, b)
    assert kelly.growth_rate(p, b, fb) == pytest.approx(0.0, abs=1e-10)
    assert fb > 1.9 * kelly.kelly_fraction(p, b)             # roughly twice Kelly for small edges
    assert kelly.growth_rate(p, b, 1.5 * fb) < 0


@pytest.mark.parametrize("mult", [0.5, 1.0, 2.0, 3.0])
def test_simulated_growth_matches_expected_log_growth(mult):
    p, b, n = 0.55, 1.0, 1000
    f = mult * kelly.kelly_fraction(p, b)
    lw = kelly.simulate_log_wealth(p, b, f, n, 20_000, seed=1)
    expected = n * float(kelly.growth_rate(p, b, f))
    se = lw.std(ddof=1) / math.sqrt(len(lw))
    assert abs(lw.mean() - expected) < 4 * se


@pytest.mark.parametrize("mult", [1.0, 2.0, 3.0])
def test_exact_probability_of_losing_money_matches_simulation(mult):
    p, b, n = 0.55, 1.0, 1000
    f = mult * kelly.kelly_fraction(p, b)
    exact = kelly.prob_end_below_start(p, b, f, n)
    sim = (kelly.simulate_log_wealth(p, b, f, n, 40_000, seed=2) < 0).mean()
    assert abs(sim - exact) < 4 * math.sqrt(exact * (1 - exact) / 40_000)


def test_bet_by_bet_simulation_agrees_with_binomial_shortcut():
    """The binomial shortcut must equal actually betting sequentially."""
    p, b, f, n = 0.55, 1.0, 0.1, 50
    rng = np.random.default_rng(5)
    wins = rng.random((20_000, n)) < p
    log_w = np.where(wins, math.log1p(b * f), math.log1p(-f)).sum(axis=1)
    short = kelly.simulate_log_wealth(p, b, f, n, 20_000, seed=6)
    assert abs(log_w.mean() - short.mean()) < 4 * log_w.std() / math.sqrt(20_000) * math.sqrt(2)


# ---------------------------------- ruin ----------------------------------
@pytest.mark.parametrize("p", [0.5, 0.48, 0.55, 0.3])
@pytest.mark.parametrize("N", [10, 40])
def test_closed_forms_match_linear_system(p, N):
    h, d = ruin.solve_linear(N, p)
    for i in (1, N // 4, N // 2, N - 1):
        assert ruin.win_probability(i, N, p) == pytest.approx(h[i], rel=1e-9, abs=1e-12)
        assert ruin.expected_duration(i, N, p) == pytest.approx(d[i], rel=1e-9)


def test_fair_game_formulas():
    assert ruin.win_probability(30, 100, 0.5) == pytest.approx(0.3)
    assert ruin.expected_duration(30, 100, 0.5) == pytest.approx(30 * 70)


def test_small_unfair_edge_is_costly():
    # classic: 48% game, start 50, target 100 -> ~1.8% chance, versus 50% if fair
    assert ruin.win_probability(50, 100, 0.48) == pytest.approx(0.01795, abs=1e-4)


@pytest.mark.parametrize("p", [0.5, 0.48, 0.6])
def test_simulation_matches_exact(p):
    N, i, n = 20, 8, 20_000
    prob, dur = ruin.simulate(i, N, p, n, seed=3)
    exact_p = ruin.win_probability(i, N, p)
    assert abs(prob - exact_p) < 4 * math.sqrt(exact_p * (1 - exact_p) / n)
    assert dur == pytest.approx(ruin.expected_duration(i, N, p), rel=0.03)



def test_kelly_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        kelly.kelly_fraction(1.2, 1.0)
    with pytest.raises(ValueError):
        kelly.kelly_fraction(0.5, 0.0)
    with pytest.raises(ValueError):
        kelly.simulate_log_wealth(0.5, 1.0, 1.0, 10, 100)
    with pytest.raises(ValueError):
        kelly.simulate_log_wealth(0.5, 1.0, 0.1, 0, 100)


def test_ruin_handles_deterministic_games_and_rejects_invalid_inputs():
    assert ruin.win_probability(4, 10, 1.0) == 1.0
    assert ruin.win_probability(4, 10, 0.0) == 0.0
    assert ruin.expected_duration(4, 10, 1.0) == 6.0
    assert ruin.expected_duration(4, 10, 0.0) == 4.0
    with pytest.raises(ValueError):
        ruin.win_probability(11, 10, 0.5)
    with pytest.raises(ValueError):
        ruin.win_probability(5, 10, 1.1)
