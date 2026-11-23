# Quant Games Lab

Classic quantitative-interview and betting problems, each solved **exactly** (closed form, dynamic programming, or full enumeration) and then **checked against simulation**. The point is a verified, reproducible implementation of the reasoning that quant interviews test, plus the derivations to explain it.

| Game | File | Exact method | Checked by |
|---|---|---|---|
| Kelly betting | `src/kelly.py` | Closed form $f^*=p-q/b$, exact binomial probabilities | Simulation of 40,000 wealth paths per setting |
| Gambler's ruin | `src/ruin.py` | Closed forms | Direct linear-system solve, Monte Carlo |
| Secretary problem | `src/stopping.py` | Closed form | Enumeration of all $n!$ orderings ($n=6$), Monte Carlo |
| Red/black card game | `src/stopping.py` | Dynamic programme | Hand-computed cases, Monte Carlo of the optimal policy |
| Dice market-making vs an informed trader | `src/dice_market.py` | Exact expectation | Full enumeration of every outcome, Monte Carlo |
| Coupon collector, max of dice | `src/puzzles.py` | Closed forms | Monte Carlo |
| Mental-maths drills | `src/drills.py` | n/a | Answers recomputed independently in tests |

Everything is a textbook problem with a known answer, which is deliberate: the answers can be verified. This is a study of the methods, not new research.

## Run it
```bash
pip install -r requirements.txt
python -m pytest -q                 # 110 tests, ~6 s
python main.py                      # ~25 s, deterministic for a given --seed (default 42)
python -m src.drills --n 20 --seed 1   # timed mental-maths session
```
Two runs with the same seed give a byte-identical `results/tables/run_summary.json`.

## Results (seed 42; numbers from `results/tables/`)

### Kelly criterion: 55% coin, even money, 1,000 bets
| Bet size | Exact growth per bet | Median final wealth | Chance of ending below start |
|---|---|---|---|
| ½ Kelly (5%) | +0.00375 | 42.6 | 0.9% |
| Kelly (10%) | +0.00501 | 149.7 | 6.0% |
| 2× Kelly (20%) | −0.00014 | 0.87 | 51.2% |
| 3× Kelly (30%) | −0.0162 | ≈ 0 | 95.4% |

Simulated growth rates agree with the exact values (largest |z| = 1.3 across the four rows, 40,000 paths each), and the simulated chance of ending below the start agrees with the exact binomial probability. Betting twice Kelly turns a winning game into a losing one.

![Kelly](results/figures/01_kelly_growth.png)

### Gambler's ruin ($p=0.48$, target 100)
Starting from 50, the chance of reaching 100 before 0 is **1.8%** (a fair game would give 50%), and it takes about 1,205 rounds on average. Closed forms match a direct linear solve to $10^{-9}$ and the simulation within error.

![Ruin](results/figures/02_gamblers_ruin.png)

### Secretary problem ($n=100$)
Rejecting the first 37 and then taking the first candidate better than all seen succeeds with probability **0.3710** (the large-$n$ limit is $1/e=0.3679$). The exact formula equals brute-force enumeration for $n=6$.

![Secretary](results/figures/03_secretary.png)

### Red/black card game (26 red, 26 black, stop whenever you like)
The dynamic programme gives an expected win of **2.6245** under optimal stopping; simulating the optimal policy gives 2.624 ± 0.020. A strategy that always draws the whole deck earns exactly 0.

![Card game](results/figures/04_card_game_value.png)

### Dice market-making with an informed counterparty
You quote a bid and ask around the mean of the sum of 3 dice; with probability 30% the customer has seen some of the dice and trades only when it pays. The exact break-even half-spread is **0.45** if the informed trader has seen one die and **0.614** if two. Simulation agrees within error (largest |z| = 1.8 across six settings). With one die seen, break-even is $1.5\mu$ no matter how many dice there are in total (derivation in `PROJECT_NOTES.md`). This is the same adverse-selection logic as Glosten–Milgrom.

![Dice market](results/figures/05_dice_market.png)

### Puzzles
| | Exact | Simulated |
|---|---|---|
| Coupon collector, 50 types: mean draws | 224.96 | 225.10 ± 0.44 |
| Coupon collector, 50 types: variance | 3837.9 | 3803.9 |
| Expected maximum of 3 dice | 4.9583 | 4.9577 ± 0.0011 |

## Continuous integration

GitHub Actions runs the test suite on pushes and pull requests using Python 3.11 and 3.12. The same tests can be run locally with `python -m pytest -q`.

## How it is tested
- Closed forms are compared with **independent** calculations (linear-system solve, full enumeration of outcomes, hand-computed cases), not just with simulation.
- Simulations must land within 4 standard errors of the exact value.
- Properties are tested directly: Kelly maximises growth on a fine grid; the card-game value never decreases when red cards are added; break-even spread rises with the share of informed traders.
- As a spot check of the tests themselves, I injected three deliberate bugs (wrong Kelly formula, wrong card-game sign, wrong informed-trader threshold). The suite failed each time. This was a manual check, not a scripted mutation-testing run.

## Limitations
- Textbook problems: the answers are known, so this shows correct implementation and explanation, not discovery.
- The Kelly model assumes the win probability is known exactly; with an estimated $p$, betting full Kelly is riskier than shown (the 2×-Kelly row hints at why).
- The dice market is a stylised, one-customer game: the informed trader follows a simple rule and the market maker never learns from the order flow.
- The drill session's interactive loop is only tested with scripted input; the problem generator and answers are tested thoroughly.
- Public numerical functions validate common invalid inputs; these checks are not a substitute for reviewing model assumptions.

## Layout
```
src/                 solvers, drills, plots
tests/               110 tests
main.py              all experiments
results/             figures/ and tables/ written by main.py
PROJECT_NOTES.md     derivations and interview-style questions
LEARNING_ROADMAP.md
```
