"""
visualization.py -- figures for the Quant Games Lab.

One colour per entity (blue = exact solution, orange = comparison series),
grey for reference lines, simulation shown as hollow markers with 95% error
bars over the exact curve. Thin lines, recessive grid, a legend whenever there
are two or more series.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK_2, GRID, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#8a8985"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DPI = 160

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "font.size": 10, "legend.frameon": False, "lines.linewidth": 2.0,
})


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _sim_points(ax, x, y, se, color=ORANGE, label="Simulation (95% CI)"):
    ax.errorbar(x, y, yerr=1.96 * np.asarray(se), fmt="o", color=color, markerfacecolor=SURFACE,
                markeredgewidth=2, markersize=7, capsize=3, linewidth=1.4, label=label, zorder=4)


def plot_kelly(f_grid, g_grid, f_star, f_break, sim_f, sim_g, sim_se, out_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.plot(f_grid, g_grid, color=BLUE, label="Exact $g(f)$")
    ax.axvline(f_star, color=GREY, linestyle=":", linewidth=1.2)
    ax.axvline(f_break, color=GREY, linestyle=":", linewidth=1.2)
    ymax = max(g_grid)
    ax.text(f_star + 0.004, ymax * 1.02, f"Kelly $f^*$={f_star:.2f}", color=INK_2, fontsize=9)
    ax.text(f_break + 0.004, ymax * 0.6, f"growth turns\nnegative at {f_break:.2f}", color=INK_2, fontsize=9)
    _sim_points(ax, sim_f, sim_g, sim_se)
    ax.set_xlabel("Fraction of wealth bet each round $f$")
    ax.set_ylabel("Expected log-growth per bet $g(f)$")
    ax.set_title("Betting more than Kelly lowers growth; about 2x Kelly gives none")
    ax.legend(loc="lower left", fontsize=9)
    return _save(fig, out_dir, "01_kelly_growth.png")


def plot_ruin(starts, exact, sim, sim_se, p, N, out_dir):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(starts, starts / N, color=GREY, linestyle=(0, (4, 3)), linewidth=1.4, label="Fair game ($p=0.5$)")
    ax.plot(starts, exact, color=BLUE, label=f"Exact, $p={p}$")
    _sim_points(ax, sim[0], sim[1], sim_se)
    ax.set_xlabel(f"Starting wealth (target {N}, ruin at 0)")
    ax.set_ylabel("Probability of reaching the target")
    ax.set_title("A small house edge makes the target very unlikely")
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, out_dir, "02_gamblers_ruin.png")


def plot_secretary(rs, exact, sim_r, sim_p, sim_se, r_star, p_star, n, out_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(1 / np.e, color=GREY, linestyle=(0, (4, 3)), linewidth=1.4, label="$1/e \\approx 0.368$")
    ax.plot(rs, exact, color=BLUE, label="Exact")
    _sim_points(ax, sim_r, sim_p, sim_se)
    ax.scatter([r_star], [p_star], s=60, color=BLUE, zorder=5)
    ax.annotate(f"best: reject first {r_star - 1}\nP = {p_star:.3f}", (r_star, p_star), xytext=(r_star + 14, p_star - 0.17),
                fontsize=9, color=INK_2, arrowprops=dict(arrowstyle="-", color=GREY))
    ax.set_xlabel(f"Cutoff $r$ (reject the first $r-1$ of {n} candidates)")
    ax.set_ylabel("Probability of picking the best")
    ax.set_title("Secretary problem: look at about 37%, then leap")
    ax.legend(loc="lower center", fontsize=9)
    return _save(fig, out_dir, "03_secretary.png")


def plot_card_game(sizes, values, sim_sizes, sim_vals, sim_se, out_dir):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(sizes, values, color=BLUE, label="Exact (dynamic programme)")
    _sim_points(ax, sim_sizes, sim_vals, sim_se)
    ax.set_xlabel("Red cards = black cards in the deck")
    ax.set_ylabel("Expected winnings with optimal stopping")
    ax.set_title("Red/black card game: the value of the option to stop")
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, out_dir, "04_card_game_value.png")


def plot_dice_market(w_grid, curves, sims, break_evens, out_dir):
    """curves: {label: exact profit array}; sims: {label: (w, mean, se)}; break_evens: {label: w*}."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0, color=INK_2, linewidth=1)
    colors = [BLUE, ORANGE]
    for (label, y), c in zip(curves.items(), colors):
        ax.plot(w_grid, y, color=c, label=f"{label}: exact")
        ax.scatter([break_evens[label]], [0], s=70, facecolor=SURFACE, edgecolor=c, linewidth=2, zorder=5)
        w, m, se = sims[label]
        ax.errorbar(w, m, yerr=1.96 * np.asarray(se), fmt="o", color=c, markerfacecolor=SURFACE,
                    markeredgewidth=1.6, markersize=5, capsize=2, linewidth=1.0, zorder=4)
    ax.set_xlabel("Half-spread $w$ quoted around the mean")
    ax.set_ylabel("Expected profit per customer")
    ax.set_title("The break-even spread is set by what the informed trader knows")
    ax.text(0.99, 0.04, "hollow circles on zero: break-even $w^*$\nsmall markers: simulation (95% CI)",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=INK_2)
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, out_dir, "05_dice_market.png")
