"""
main.py -- solves every game exactly, checks each against simulation, and
writes tables, figures and results/tables/run_summary.json.

    python main.py            # about 1 minute
    python main.py --seed 7

Games: Kelly betting; gambler's ruin; the secretary problem; a red/black card
stopping game; a dice market-making game with an informed trader; and two
expectation puzzles (coupon collector, maximum of dice).
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from src import dice_market as dm
from src import kelly, puzzles, ruin, stopping
from src import visualization as viz

FIG_DIR, TABLE_DIR = "results/figures", "results/tables"


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def save(df, name):
    df.to_csv(os.path.join(TABLE_DIR, name), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    seed = parser.parse_args().seed
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    summary = {"seed": seed}

    # ------------------------------------------------------------------
    section("1. Kelly criterion: 55% coin, even-money bet, 1000 rounds")
    p, b, n_bets, n_paths = 0.55, 1.0, 1000, 40_000
    f_star, f_break = kelly.kelly_fraction(p, b), kelly.break_even_fraction(p, b)
    rows = []
    for mult in (0.5, 1.0, 2.0, 3.0):
        f = mult * f_star
        lw = kelly.simulate_log_wealth(p, b, f, n_bets, n_paths, seed=seed + int(mult * 10))
        g = float(kelly.growth_rate(p, b, f))
        rows.append({"multiple_of_kelly": mult, "fraction": f, "exact_growth_per_bet": g,
                     "sim_growth_per_bet": lw.mean() / n_bets, "sim_se": lw.std(ddof=1) / n_bets / math.sqrt(n_paths),
                     "median_final_wealth": float(np.exp(np.median(lw))),
                     "p_end_below_start_exact": kelly.prob_end_below_start(p, b, f, n_bets),
                     "p_end_below_start_sim": float((lw < 0).mean())})
    kt = pd.DataFrame(rows)
    print(f"Kelly f* = {f_star:.3f}; growth turns negative at f = {f_break:.3f}")
    print(kt.round(5).to_string(index=False))
    save(kt, "kelly.csv")
    grid = np.linspace(0.001, 0.4, 400)
    viz.plot_kelly(grid, kelly.growth_rate(p, b, grid), f_star, f_break, kt["fraction"], kt["sim_growth_per_bet"], kt["sim_se"], FIG_DIR)
    summary["kelly"] = {"f_star": f_star, "break_even_fraction": f_break,
                        "max_abs_z_growth": float(((kt.sim_growth_per_bet - kt.exact_growth_per_bet) / kt.sim_se).abs().max()),
                        "p_end_below_start_exact": dict(zip(kt["multiple_of_kelly"].astype(str), kt["p_end_below_start_exact"]))}

    # ------------------------------------------------------------------
    section("2. Gambler's ruin: p = 0.48, target 100")
    N, p_r = 100, 0.48
    starts = np.arange(1, N)
    exact = np.array([ruin.win_probability(i, N, p_r) for i in starts])
    sim_starts = np.array([20, 40, 60, 80, 90])
    sim_vals, sim_se = [], []
    for i in sim_starts:
        prob, _ = ruin.simulate(int(i), N, p_r, 20_000, seed=seed + int(i))
        sim_vals.append(prob)
        e = ruin.win_probability(int(i), N, p_r)
        sim_se.append(math.sqrt(max(e * (1 - e), 1e-12) / 20_000))
    rr = pd.DataFrame({"start": sim_starts, "exact": [ruin.win_probability(int(i), N, p_r) for i in sim_starts],
                       "simulated": sim_vals, "se": sim_se,
                       "exact_expected_duration": [ruin.expected_duration(int(i), N, p_r) for i in sim_starts]})
    print(rr.round(4).to_string(index=False))
    save(rr, "gamblers_ruin.csv")
    viz.plot_ruin(starts, exact, (sim_starts, sim_vals), sim_se, p_r, N, FIG_DIR)
    summary["ruin"] = {"p_reach_100_from_50": ruin.win_probability(50, N, p_r),
                       "fair_game_p_reach_100_from_50": 0.5}

    # ------------------------------------------------------------------
    section("3. Secretary problem, n = 100")
    n = 100
    rs = np.arange(1, n + 1)
    exact_s = np.array([stopping.secretary_success_probability(int(r), n) for r in rs])
    r_star, p_star = stopping.secretary_best_cutoff(n)
    sim_r = np.array([10, 25, 38, 55, 80])
    sim_p = np.array([stopping.secretary_simulate(int(r), n, 20_000, seed=seed + int(r)) for r in sim_r])
    sim_se_s = np.sqrt(sim_p * (1 - sim_p) / 20_000)
    st = pd.DataFrame({"cutoff_r": sim_r, "exact": exact_s[sim_r - 1], "simulated": sim_p, "se": sim_se_s})
    print(st.round(4).to_string(index=False))
    print(f"best cutoff r* = {r_star} (reject the first {r_star - 1}); success probability {p_star:.4f}; 1/e = {1 / math.e:.4f}")
    save(st, "secretary.csv")
    viz.plot_secretary(rs, exact_s, sim_r, sim_p, sim_se_s, r_star, p_star, n, FIG_DIR)
    summary["secretary"] = {"n": n, "best_cutoff": r_star, "best_success_probability": p_star,
                            "one_over_e": 1 / math.e}

    # ------------------------------------------------------------------
    section("4. Red/black card game with optimal stopping")
    V = stopping.card_game_values(26, 26)
    sizes = np.arange(1, 27)
    sim_sizes = np.array([5, 10, 20, 26])
    sv, sse = [], []
    for s in sim_sizes:
        m, se = stopping.card_game_simulate(int(s), int(s), 10_000, seed=seed + int(s), V=V)
        sv.append(m); sse.append(se)
    cg = pd.DataFrame({"cards_each_colour": sim_sizes, "exact_value": [V[s, s] for s in sim_sizes],
                       "simulated": sv, "se": sse})
    cg["z"] = (cg.simulated - cg.exact_value) / cg.se
    print(cg.round(4).to_string(index=False))
    save(cg, "card_game.csv")
    viz.plot_card_game(sizes, [V[s, s] for s in sizes], sim_sizes, sv, sse, FIG_DIR)
    summary["card_game"] = {"value_26_26": float(V[26, 26]), "max_abs_z": float(cg["z"].abs().max())}

    # ------------------------------------------------------------------
    section("5. Dice market-making with an informed trader (3 dice, 30% informed)")
    n_dice, mu = 3, 0.3
    w_grid = np.linspace(0, 3.5, 200)
    curves, sims, bes, rows = {}, {}, {}, []
    for k in (1, 2):
        label = f"Informed sees {k} die" if k == 1 else f"Informed sees {k} dice"
        curves[label] = np.array([dm.expected_profit(n_dice, k, mu, w) for w in w_grid])
        bes[label] = dm.break_even_half_spread(n_dice, k, mu)
        ws = np.array([0.5, 1.5, 2.5])
        res = [dm.simulate(n_dice, k, mu, float(w), 400_000, seed=seed + k * 100 + int(w * 10)) for w in ws]
        sims[label] = (ws, np.array([r[0] for r in res]), np.array([r[1] for r in res]))
        for w, (m, se) in zip(ws, res):
            rows.append({"k_dice_seen": k, "half_spread": w, "exact": dm.expected_profit(n_dice, k, mu, float(w)),
                         "simulated": m, "se": se, "z": (m - dm.expected_profit(n_dice, k, mu, float(w))) / se})
        print(f"{label}: break-even half-spread w* = {bes[label]:.4f}")
    dt = pd.DataFrame(rows)
    print(dt.round(4).to_string(index=False))
    save(dt, "dice_market.csv")
    viz.plot_dice_market(w_grid, curves, sims, bes, FIG_DIR)
    summary["dice_market"] = {"mu": mu, "break_even_k1": bes["Informed sees 1 die"],
                              "break_even_k2": bes["Informed sees 2 dice"], "max_abs_z": float(dt["z"].abs().max())}

    # ------------------------------------------------------------------
    section("6. Expectation puzzles")
    cm, cv, cse = puzzles.coupon_collector_simulate(50, 20_000, seed=seed)
    mx, mse = puzzles.expected_max_of_dice_simulate(3, 1_000_000, seed=seed)
    pz = pd.DataFrame([
        {"puzzle": "Coupon collector, n=50: mean draws", "exact": puzzles.coupon_collector_mean(50), "simulated": cm, "se": cse},
        {"puzzle": "Coupon collector, n=50: variance", "exact": puzzles.coupon_collector_variance(50), "simulated": cv, "se": float("nan")},
        {"puzzle": "Expected maximum of 3 dice", "exact": puzzles.expected_max_of_dice(3), "simulated": mx, "se": mse},
    ])
    print(pz.round(4).to_string(index=False))
    save(pz, "puzzles.csv")
    summary["puzzles"] = {"coupon_mean_exact": puzzles.coupon_collector_mean(50), "coupon_mean_sim": cm,
                          "max3dice_exact": puzzles.expected_max_of_dice(3), "max3dice_sim": mx}

    with open(os.path.join(TABLE_DIR, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    section("DONE")
    print(f"figures -> {FIG_DIR}/   tables -> {TABLE_DIR}/")


if __name__ == "__main__":
    main()
