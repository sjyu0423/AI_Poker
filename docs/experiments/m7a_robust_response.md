# M7A results: Restricted Nash Response and Data-Biased Response

Reproduce (about 85 minutes on 12 CPU cores):

```bash
pokermind robust-study --hero runs/m4-leduc-teacher/strategies/strategy_100000.json
```

Numbers come from `runs/m7a-robust-response/metrics.json`; the plot is
`runs/m7a-robust-response/frontier.png`. The design is in
`docs/design/robust_response.md`; M6 definitions are in
`docs/design/adaptive_strategy.md`.

* **Gain:** hero's exact expected payoff per hand minus fixed CFR's, against
  the true opponent, in chips per hand. Seats alternate.
* **Budget:** exact exploitability (NashConv / 2) ≤ expl(CFR) + δ, with
  expl(CFR) = 0.0030 and δ ∈ {0, 0.02, 0.05, 0.10, 0.25}.
* **Uncertainty:** 95 % CIs are across opponents, after averaging each
  opponent's seeds. Comparisons between families are **paired** on the same
  opponent, seed and observation stream.
* **Opponents (M6 world, unchanged):**
  * 24 held-out opponents from the development families with new parameters,
    3 seeds;
  * 8 out-of-population opponents (station, maniac, trapper, card-blind),
    3 seeds;
  * development opponents (2 seeds) are used only to tune DBR's threshold.

## 1. Answer to the research question

**Yes, RNR beats M6's mixture at every budget, significantly and by a wide
margin. DBR also beats the mixture, but it is significantly worse than RNR.**

Held-out opponents, after 100 hands, gain over CFR:

| δ above CFR | mix (M6) | RNR | DBR | RNR − mix (paired) | oracle mix | oracle RNR |
|---|---|---|---|---|---|---|
| 0.02 | +0.004 | **+0.087** | +0.028 | +0.083 [+0.054, +0.113] | +0.008 | +0.117 |
| 0.05 | +0.009 | **+0.115** | +0.051 | | +0.017 | +0.155 |
| 0.10 | +0.016 | **+0.138** | +0.063 | +0.123 [+0.084, +0.161] | +0.034 | +0.193 |
| 0.25 | +0.037 | **+0.186** | +0.094 | +0.149 [+0.103, +0.194] | +0.079 | +0.261 |

Out-of-population opponents, after 100 hands:

| δ | mix | RNR | DBR | RNR − mix (paired) | oracle RNR |
|---|---|---|---|---|---|
| 0.02 | +0.020 | **+0.281** | +0.101 | +0.262 [+0.176, +0.348] | +0.449 |
| 0.10 | +0.082 | **+0.619** | +0.374 | +0.537 [+0.307, +0.766] | +0.785 |
| 0.25 | +0.190 | **+0.775** | +0.579 | +0.585 [+0.355, +0.815] | +0.995 |

* **Size of the effect.** At δ = 0.10, RNR gets almost **9×** M6's held-out
  gain at the same exact exploitability. It even beats the *oracle* mixture,
  which knows the true policy, by a factor of 4. RNR reaches 72 % (held-out)
  and 79 % (out-of-population) of the oracle RNR.
* **Why.** The mixture is a chord of the (exploitability, payoff) frontier;
  RNR is on the frontier. Small exploitability budgets are where the
  difference is largest: at δ = 0.02 the mixture can only shift 1.6–2 % of
  its probability to the best response, while RNR changes its play wherever
  exploiting is cheap.
* **Budgets are respected.** RNR's largest exact exploitability is 0.0229 /
  0.0528 / 0.1030 / 0.2524 against budgets 0.0230 / 0.0530 / 0.1030 /
  0.2530. RNR never fell back to CFR. Compliance comes from **exact
  verification of every candidate**, not from a guarantee: the exploitability
  of `σ*_p` is not known before it is measured.
* **M6 reproduction.** M6's budget_max, gated and br_map gains were
  recomputed inside this study and match `runs/m6-adaptive-strategy` on all
  576 rows with maximum difference 0.0.

## 2. Personal increment (secondary question)

The increment is gain after `n` hands minus gain with 0 hands (the population
prior alone), paired per opponent.

**RNR (held-out):** significant only with enough data and budget:

| δ | 20 hands | 100 hands | 500 hands |
|---|---|---|---|
| 0.02 | −0.001 [−0.007, +0.006] | +0.002 [−0.007, +0.012] | **+0.014 [+0.004, +0.024]** |
| 0.10 | −0.003 [−0.010, +0.004] | **+0.015 [+0.003, +0.027]** | **+0.041 [+0.026, +0.055]** |
| 0.25 | −0.004 [−0.019, +0.012] | +0.019 [−0.008, +0.045] | **+0.058 [+0.019, +0.097]** |

* On held-out opponents most of RNR's gain is already there at 0 hands:
  +0.124 of +0.138 at δ = 0.10 and 100 hands. RNR against the *population*
  model exploits the population's shared tendencies.
* Personal observations now add a measurable amount: about **+0.04
  chips/hand at 500 hands** (δ = 0.10), a 33 % increase over the zero-hand
  policy. M6 had found no significant increment with the mixture (−0.002 to
  +0.002), and that is reproduced here: the mixture's increment is still not
  significant on held-out opponents.

**RNR (out-of-population):** larger and earlier: +0.195 [+0.062, +0.329] at
100 hands and +0.263 [+0.116, +0.410] at 500 (δ = 0.10). Unfamiliar
opponents are where observations matter.

**DBR's increment must not be read at face value.** DBR's own increment
looks large (held-out +0.066 at 100 hands, δ = 0.10), but DBR's zero-hand
policy is plain CFR: with no observations every information set is
unrestricted. Its "increment" therefore includes re-learning what the
population prior gives RNR for free. Against the right baseline, **RNR with
0 hands**, DBR never comes out ahead:

| DBR(n) − RNR(0), paired | 20 hands | 100 hands | 500 hands |
|---|---|---|---|
| held-out, δ = 0.10 | −0.084 [−0.118, −0.051] | −0.060 [−0.087, −0.034] | −0.047 [−0.067, −0.026] |
| out-of-pop., δ = 0.10 | −0.193 [−0.254, −0.132] | −0.050 [−0.124, +0.024] | −0.008 [−0.117, +0.101] |
| out-of-pop., δ = 0.25 | −0.420 [−0.564, −0.275] | −0.074 [−0.158, +0.010] | +0.018 [−0.072, +0.108] |

So DBR's total gain is **not** evidence that it personalizes better: it
doesn't exceed the zero-hand baseline that the other method already has.

## 3. DBR in detail

* **Threshold tuning** (development opponents, mean gain over all budgets and
  `n > 0`): threshold 3 scores 0.0451, 10 scores 0.0413, 30 scores 0.0291.
  The chosen threshold is 3, the least conservative one.
* DBR beats the mixture (paired, held-out δ = 0.10, 100 hands: +0.048
  [+0.032, +0.063]). It loses to RNR at every budget and hand count in both
  groups (RNR − DBR, held-out δ = 0.10: +0.075 [+0.050, +0.100]).
* **Why DBR underperforms here.** In Leduc, most of the opponent's
  information sets are rarely or never reached in 20–500 hands. In the
  inferred counts, the opponent's hidden card is unknown unless there is a
  showdown, so counts are fractional and spread out. DBR leaves those
  information sets free (worst case), which discards the population prior
  that RNR uses. DBR was designed against overfitting a model built from
  *few observations with no prior*. M6's reconstruction is already a
  posterior that shrinks to the population, so DBR's protection costs more
  than it saves.
* **Small budgets.** At δ = 0 DBR falls back to CFR in 48 % (held-out) and
  69 % (out-of-population) of runs.
* **Approximations, restated:**
  * counts are EM posterior expected counts, not observed frequencies;
  * the DBR game is not bilinear, so CFR+ has no formal convergence
    guarantee;
  * every output was nonetheless verified exactly in the original game.

## 4. Harm and worst cases

Harm is the fraction of (opponent, seed, n) runs with a gain below 0.

| | mix | RNR | DBR |
|---|---|---|---|
| held-out, δ = 0.10 | 18.5 % | 6.7 % | 17.4 % |
| held-out, δ = 0.25 | 18.5 % | 11.6 % | 18.8 % |
| out-of-pop., δ = 0.10 | 2.1 % | 0 % | 17.4 % |

* **Worst single run.** Held-out, δ = 0.10: mix −0.013 (aggressive-0, 100
  hands), RNR −0.023 (balanced-0, 20 hands), DBR −0.029 (balanced-0, 20
  hands). At δ = 0.25: mix −0.028, RNR −0.071, DBR −0.105. RNR loses less
  often than the mixture but, when it misreads a balanced opponent, by
  more. Every loss is bounded by its exploitability budget.
* **Out-of-population.** RNR's worst run at δ ≥ 0.02 is still **positive**
  (+0.063 at δ = 0.02).
* **δ = 0 (budget = CFR's 0.0030).** RNR reports +0.030 held-out because
  500-iteration CFR+ has exploitability 0.0009, below the teacher's 0.0030.
  The leftover 0.002 lets RNR use a small p (mean 0.13–0.18 held-out, 0.02–0.15 out-of-population). These are different
  near-equilibria, not exploitation of the opponent alone, and RNR loses on
  some opponents: harm is 2 % held-out and 35 % out-of-population, worst
  −0.053 (card_blind).

## 5. Changing opponents (δ = 0.10)

Gain after the switch, held-out switching opponents (8, 3 seeds), by hands
since the switch:

| policy | 0 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| mix, cumulative | +0.006 | +0.009 | +0.013 | +0.014 | +0.016 |
| mix, detection (M6) | +0.005 | +0.019 | +0.023 | +0.020 | +0.022 |
| RNR, cumulative | +0.056 | +0.072 | +0.111 | +0.130 | +0.168 |
| **RNR, detection** | +0.058 | **+0.143** | +0.174 | +0.181 | +0.183 |
| DBR, cumulative | +0.033 | +0.036 | +0.049 | +0.067 | +0.084 |
| DBR, detection | +0.033 | +0.050 | +0.079 | +0.099 | +0.099 |

* M6's change detector (ChangeAwareObserver) works unchanged with RNR and
  DBR. With RNR it doubles the gain 20 hands after a switch (+0.143 vs
  +0.072).
* **Bait-and-switch** (2 opponents, 3 seeds): the means favour RNR
  (detection: +0.51 at 20 hands, +0.71 at 100), but with 2 opponents every
  CI includes 0. This is not evidence either way.
* Exploitability stays ≤ 0.1030 in every change run.

## 6. Runtime and memory

| step | cost |
|---|---|
| tree compilation | 0.014 s |
| one RNR solve (both seats, 500 CFR+ iterations) | 1.02 s |
| exact exploitability of one candidate | 0.065 s |
| peak traced memory of one RNR solve | 1.26 MB |
| per snapshot: RNR grid (up to 13 values of p) | about 23 s |
| per snapshot: DBR grid (9 values of P_max) | about 9.4 s per threshold |
| per snapshot: mixture search | 3.6 s |

* The vectorized solver is about 33× faster than the M3 recursive solver. It
  matches M3 (vanilla mode) to 1e-12, OpenSpiel's CFRPlusSolver to 1e-12,
  and generic CFR on the explicit RNR game to 1e-12.
* An RNR selection takes about 23 s per snapshot. That is acceptable between
  sessions but too slow per hand. Ways to cut it:
  * a coarser p grid;
  * warm starts across snapshots;
  * fewer iterations: 500 are within about 0.001 of convergence.

## 7. Failure cases and limits

* **No a-priori bound.** Neither RNR nor DBR can be asked for an
  exploitability. Budgets hold only because every candidate is evaluated
  exactly and the grid is searched. If the grid were too coarse, small
  budgets would fall back to CFR (DBR at δ = 0 does so about half the time).
* **The frontier is approximate.** It is guaranteed only for exact
  equilibria. 500 CFR+ iterations are close to converged, but the selected p
  is "best on the grid, verified", not a frontier optimum.
* **Selection uses the model, not the truth.** Candidates are ranked by their
  value against the reconstruction. RNR's worst losses (balanced opponents
  after 20 hands) come from a model that looks exploitable but isn't.
* **Population prior.** Held-out gains rely on the prior from development
  opponents. Out-of-population opponents show the method still helps when the
  prior is wrong, but every number here is for Leduc and the M6 world only.
* **Bait-and-switch** is underpowered (2 opponents).

## 8. Recommendation for M7B

* **Make RNR the default exploitation method.** Use exact-verified selection
  under the budget, the M6 reconstruction and population prior, and M6's
  change detection.
* **Don't pursue DBR** with inferred counts. If it is revisited, give it the
  population prior as `σ_fix` at unobserved information sets (a
  prior-anchored DBR) and compare it against RNR(0), not CFR.
* **Speed:** warm-start RNR solves and cache across snapshots, so selection
  can run per session.
* **Selection risk:** rank candidates by a posterior-averaged value instead
  of the point model, to reduce RNR's worst cases on balanced opponents.
* **Bait-and-switch:** use more opponents, so its CIs can resolve.
