# M7B results: reliability of the adaptive RNR baseline

Reproduce (Apple M-series, 12 CPU cores):

```bash
H=runs/m4-leduc-teacher/strategies/strategy_100000.json
pokermind reliability convergence --hero $H      # ~27 min
pokermind reliability calibrate   --hero $H      # ~20 min, development opponents only
pokermind research-config --calibration runs/m7b-adaptive-reliability/calibration.json
pokermind reliability evaluate    --hero $H      # frozen config, held-out groups
pokermind reliability benchmark   --hero $H      # single process; run on an idle machine
```

* **Where the numbers come from:** `runs/m7b-adaptive-reliability/{convergence,calibration,evaluation,benchmark}.json`.
* **Frozen configuration:** `configs/pokermind-research-v1.json`.
* **Summary of the architecture and the evidence:** `docs/research/research_freeze.md`.

**Definitions** (as in M7A):
* **Gain:** exact chips per hand above the fixed CFR hero, against the true
  opponent, with seats alternating.
* **Budget:** exact exploitability ≤ expl(CFR) + δ, where expl(CFR) = 0.0030.
* **Uncertainty:** 95 % CIs are across opponents, after averaging each
  opponent's seeds. Comparisons are paired on the same observation streams.

## 1. RNR solver convergence

**Setup:**
* Snapshots: M7A's estimates (M6 reconstruction) for 24 held-out and 8
  out-of-population opponents, seed 0, after 20, 100 and 500 hands, plus the
  zero-hand population model. That is 97 snapshots.
* Each snapshot is solved for all 13 values of p in one continued CFR+ run,
  recorded after 500, 1,000 and 2,000 iterations. CFR+ is deterministic, so a
  continued run equals a fresh run of that length.

**Convergence is measured exactly**, not by eye. For hero's average strategy
`x` and the free opponent's average `y`, the RNR-game gap is:

```
gap = max_x' [p·v(x', σ_fix) + (1 − p)·v(x', y)]  −  [p·v(x, σ_fix) − (1 − p)·expl(x)]
```

It is the NashConv of `(x, y)` in the RNR game and 0 only at equilibrium.
* The first term is hero's best response to the realization mixture of the
  model and `y`.
* A test checks the gap against the NashConv of the explicitly written RNR
  game (`RNRGame`) to 1e-12.

Doing this needed one solver fix. The opponent's average strategy is now
accumulated on the unrestricted copy. Hero's strategies are unchanged bit for
bit.

| RNR-game gap (mean / max over 97 snapshots) | p = 0.1 | p = 0.3 | p = 0.6 | p = 1.0 |
|---|---|---|---|---|
| 500 iterations | 0.0019 / 0.0031 | 0.0016 / 0.0028 | 0.0007 / 0.0014 | 0.00003 / 0.00008 |
| 1,000 iterations | 0.0006 / 0.0009 | 0.0005 / 0.0009 | 0.0002 / 0.0005 | 0.00001 / 0.00003 |
| 2,000 iterations | 0.0002 / 0.0003 | 0.0002 / 0.0003 | 0.0001 / 0.0002 | < 0.00001 |

* **The gap shrinks about 3× per doubling.** At 500 iterations the largest
  gap, 0.0031, is 15 % of the smallest non-zero budget step (0.02) and 3 %
  of the primary one (0.10). It is not 0 at any iteration count: these are
  approximate equilibria, and the frontier property of RNR holds only
  approximately.
* **Strategies change where play rarely goes.** Between 500 and 2,000
  iterations, single information sets change by up to 0.99 in probability.
  Weighted by how often hero reaches them against the model, the mean
  total-variation distance is only 0.001–0.004, and at most 0.025. Similar
  maximum differences therefore do *not* indicate convergence, and
  dissimilar ones do not indicate its absence.
* **Selected strategies (point rule, 500 vs 2,000 iterations):**

  | δ | same p chosen | gain difference vs truth (500 − 2,000) | largest single difference |
  |---|---|---|---|
  | 0 | 73 % | **−0.015 [−0.023, −0.006]** | 0.26 |
  | 0.02 | 97 % | −0.0007 [−0.0019, +0.0005] | 0.048 |
  | 0.05 | 98 % | −0.0017 [−0.0035, +0.0000] | 0.072 |
  | 0.10 | 99 % | −0.0002 [−0.0012, +0.0009] | 0.042 |
  | 0.25 | 100 % | −0.0002 [−0.0006, +0.0001] | 0.009 |

  Every selected strategy at every iteration count stays within its budget.
* **M7A's conclusion doesn't depend on the iteration count.** The paired
  RNR-minus-mixture gains are the same to the third decimal at 500, 1,000
  and 2,000 iterations. For example:
  * held-out, δ = 0.1, 100 hands: +0.124 [+0.087, +0.161] /
    +0.124 [+0.087, +0.162] / +0.124 [+0.087, +0.162];
  * out-of-population: +0.550 / +0.549 / +0.549.
* **δ = 0 is the exception.** There the budget equals CFR's own 0.0030 and
  the choice depends on solver noise, as M7A noted. It is not a regime for
  exploitation.
* **Decision:** keep 500 iterations. It is twice as cheap as 1,000 and changes
  no conclusion at δ ≥ 0.02. Recorded in the frozen configuration's
  provenance.

## 2. Uncertainty calibration (development opponents only)

**Setup:**
* 24 development opponents × 2 seeds × 6 observation counts = 288
  snapshots.
* For every RNR candidate with p > 0 (3,277 candidates), 16 posterior
  samples (data-augmentation Gibbs, M6) predict its gain over CFR. Coverage
  is checked against the true gain.
* Three temperatures τ were sampled: θ is drawn from
  `Dirichlet(τ·counts + s·m)`, so τ < 1 counts the data as fewer hands.
* The intervals are `mean ± z·κ·sd`, with κ an inflation factor.

**Gain-interval coverage (two-sided):**

| τ, κ | 50 % | 80 % | 90 % | 95 % | mean abs. error |
|---|---|---|---|---|---|
| 1, 1 (M6 posterior, uncalibrated) | 40 % | 65 % | **77 %** | 84 % | 0.125 |
| 1, 1.5 | 54 % | 83 % | 92 % | 96 % | 0.024 |
| 0.5, 1 | 41 % | 70 % | 81 % | 88 % | 0.088 |
| **0.5, 1.25 (chosen)** | **51 %** | **80 %** | **89 %** | **93 %** | **0.008** |
| 0.25, 1.5 | 54 % | 82 % | 90 % | 95 % | 0.015 |
| 1, 2 | 67 % | 93 % | 98 % | 99 % | 0.105 |

* **The M6 posterior is overconfident.** Its 90 % interval for the gain
  covers 77 %. The empirical inflation that would make the untempered
  90 % interval exact is κ = 1.38.
* Tempering alone is not enough, and neither is inflation alone at τ = 1.
  The pre-declared rule (smallest mean absolute coverage error) selects
  **τ = 0.5, κ = 1.25**: coverage within 1.7 points of nominal at all four
  levels.
* **One-sided lower bounds** (what decisions use) at the chosen calibration:
  P(true gain ≥ bound) = 80 % / 93 % / 96 % for nominal 75 % / 90 % / 95 %.
  So they are slightly conservative.
* **The opponent model's own intervals are overconfident too.** These are
  intervals for each card-conditional action probability, weighted by how
  often the information set is reached:
  * the untempered 90 % interval covers 81 % (error 0.063 over the four
    levels);
  * τ = 0.5 gives 86 % (error 0.037);
  * τ = 0.5 with κ = 1.25 gives 89 % at the 90 % level.

**Effect on decisions** (offline replay of the same candidates, δ = 0.1, all
observation counts):

| rule | mean gain | harm rate | mean loss | worst | plays CFR | max exploitability |
|---|---|---|---|---|---|---|
| point (M7A) | +0.1350 | 14.9 % | −0.0008 | −0.036 | 0 % | 0.1028 |
| posterior mean (z = 0) | +0.1352 | 14.6 % | −0.0007 | −0.020 | 0 % | 0.1028 |
| calibrated LCB, 75 % | +0.1345 | 11.1 % | −0.0006 | −0.020 | 3 % | 0.1028 |
| calibrated LCB, 90 % | +0.1306 | 5.6 % | −0.0004 | −0.020 | 18 % | 0.1028 |
| **calibrated LCB, 95 % (frozen)** | **+0.1242** | **3.1 %** | **−0.0002** | **−0.014** | 26 % | 0.1028 |
| uncalibrated LCB, 95 % | +0.1333 | 9.0 % | −0.0005 | −0.020 | 8 % | 0.1028 |

* **Pre-declared choice:** among variants with at least 90 % of the best
  development mean gain, take the lowest harm rate. That gives the
  **95 % calibrated lower bound**. It gives up 8 % of the development mean
  gain (0.011 chips/hand) and cuts the share of losing runs from 14.9 % to
  3.1 %.
* **Small stakes.** The losses involved were small in every variant (mean
  loss ≤ 0.0008). The rule mainly removes *frequent small* losses, on
  near-equilibrium opponents.
* **Exploitability is unaffected.** Calibration only chooses *among* budget-
  feasible candidates, or falls back to CFR.
* **Cost:** sampling takes 0.15 s per snapshot on average.
* Everything was frozen into `configs/pokermind-research-v1.json` before the
  held-out evaluation below.

## 3. Selection speed (identical inputs)

**Setup:** 24 snapshots (4 held-out opponents × 6 observation counts),
selected at all five budgets by each procedure, single process, idle machine.
"Same selection" means the same `p` as M7A's procedure; gains are exact
against the true opponent.

| procedure | mean time | speed-up | same selection (δ = 0 / 0.02 / 0.1 / 0.25) | mean gain change at δ = 0.1 | budget violations |
|---|---|---|---|---|---|
| M7A (13 separate CFR+ solves, M3 evaluator per candidate) | 12.5 s | 1.0× | — | — | 0 |
| **batched solve + batched evaluator (adopted)** | **7.5 s** | **1.7×** | **100 % / 100 % / 100 % / 100 %** | **0 (bit-identical)** | 0 |
| batched, 200 iterations | 3.2 s | 3.9× | 0 % / 83 % / 100 % / 100 % | −0.0029 | 0 |
| batched, warm start from the previous snapshot + 200 iterations | 3.9 s | 3.2× | 71 % / 88 % / 100 % / 96 % | −0.0021 | 0 |

* **What was optimised.**
  * All (p, seat, copy) problems are solved as rows of one array program
    (`robust/batch.py`). The accumulation order is unchanged, so hero's
    strategies are bit-identical to M7A's (2,880 selections
    compared in the held-out run, max difference 9e-16).
  * Exploitability and values of all candidates come from one level-wise
    evaluator (`TreeEvaluator`, 37× faster than the M3 recursion and equal
    to it to 4e-16). Only the chosen candidate is re-checked by M3: one
    exact certificate per decision instead of one evaluation per
    candidate.
  * Posterior sampling plus evaluating every candidate against 16 samples
    adds 0.1 s.
* **Where the time goes now.** About 99 % is the 500 CFR+ iterations. The
  remaining speed-ups (fewer iterations, warm starts) change selections at
  small budgets and lose up to 0.05–0.09 chips/hand on single snapshots.
  They were **not** adopted. Correctness of the frozen procedure was not
  traded for speed.
* **Memory.** Peak traced allocation is 28 MB batched vs 12 MB for M7A.
* **Under parallel load** (the 12-process studies) the batched solve takes
  ~45 s per snapshot, because it is memory-bandwidth-bound. M7A's "about
  23 s" was measured that way too. Single-process figures are the relevant
  ones for an interactive application.
* **How often to re-solve.** Re-using the strategy chosen at the previous
  checkpoint instead of re-solving costs on average 0.007–0.022 chips/hand on
  held-out opponents (0.01–0.13 out-of-population, most in the first
  20 hands). Re-solving every 20–100 hands in the background is enough; a
  per-hand re-solve is unnecessary.

## 4. Held-out robustness of the frozen procedure

The frozen configuration (`lcb`, z = 1.645, κ = 1.25, τ = 0.5, 500
iterations) was run once on the held-out groups. "Point" is M7A's rule on the
same streams, "mix" is M7A's best mixing baseline, read from M7A's raw rows.
**M7A reproduction:** all 2,880 point-rule selections match M7A's
(max difference 9e-16, no parameter mismatches).

**Gain over CFR at δ = 0.1** (chips/hand; 95 % CIs in the paired rows):

| group | rule | 0 hands | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|---|
| held-out (24) | point | +0.124 | +0.121 | +0.129 | +0.138 | +0.143 | +0.164 |
| | **frozen** | **+0.099** | **+0.112** | **+0.119** | **+0.131** | **+0.139** | **+0.162** |
| | mix | +0.017 | +0.015 | +0.016 | +0.016 | +0.016 | +0.019 |
| | frozen − mix | +0.082 [+0.051, +0.113] | +0.097 [+0.059, +0.136] | | +0.115 [+0.075, +0.155] | | +0.142 [+0.097, +0.187] |
| | frozen − point | −0.024 [−0.037, −0.012] | −0.009 [−0.016, −0.002] | | −0.007 [−0.012, −0.003] | | −0.003 [−0.005, −0.001] |
| out-of-population (8) | point | +0.424 | +0.493 | +0.561 | +0.619 | +0.677 | +0.686 |
| | **frozen** | **+0.346** | **+0.471** | **+0.554** | **+0.619** | **+0.676** | **+0.686** |
| | frozen − mix | +0.301 [+0.151, +0.451] | +0.417 [+0.232, +0.602] | | +0.537 [+0.307, +0.766] | | +0.610 [+0.356, +0.864] |
| equilibrium (2: teacher, CFR+ 500) | point | −0.014 | −0.001 | −0.005 | −0.003 | −0.002 | −0.004 |
| | **frozen** | **−0.007** | **0.000** | **0.000** | **0.000** | **−0.001** | **0.000** |
| balanced + equilibrium (6) | point | −0.000 | +0.000 | +0.004 | +0.005 | +0.005 | +0.004 |
| | **frozen** | **+0.001** | **+0.000** | **+0.000** | **+0.002** | **+0.001** | **+0.003** |

**Increment from personalisation** (gain at n hands minus the zero-hand RNR
response built from the population model, same opponent, δ = 0.1):

| group | rule | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|
| held-out | point | −0.003 [−0.010, +0.004] | +0.005 [−0.004, +0.014] | +0.015 [+0.003, +0.027] | +0.019 [+0.007, +0.032] | +0.041 [+0.026, +0.055] |
| | frozen | −0.012 [−0.021, −0.002] | −0.005 [−0.016, +0.007] | +0.007 [−0.004, +0.019] | +0.016 [+0.003, +0.029] | +0.038 [+0.023, +0.054] |
| out-of-population | point | +0.069 [−0.008, +0.145] | +0.138 [+0.025, +0.250] | +0.195 [+0.062, +0.329] | +0.253 [+0.099, +0.407] | +0.263 [+0.116, +0.410] |
| | frozen | +0.048 [−0.019, +0.115] | +0.131 [+0.022, +0.240] | +0.195 [+0.062, +0.329] | +0.253 [+0.099, +0.406] | +0.263 [+0.116, +0.410] |

* **Most of the gain against in-population opponents comes before any
  personal data.** The zero-hand RNR response to the population model
  already earns +0.12 (point) / +0.10 (frozen) of the +0.16 available at
  500 hands. Personal observations add a significant +0.016 by 200 hands and
  +0.038 by 500. At 20 hands they add nothing (point) or cost a little
  (frozen, whose zero-hand reference is lower because it often falls back).
* **Against out-of-population opponents personalisation is what matters:**
  +0.13 by 50 hands, +0.26 by 500.
* **Risk** (share of opponent × seed × hands runs with gain < 0; worst run):

  | group, δ = 0.1 | point harm | frozen harm | point worst | frozen worst | frozen falls back to CFR |
  |---|---|---|---|---|---|
  | held-out | 6.7 % | **2.3 %** | −0.023 | −0.019 | 28 % |
  | out-of-population | 0 % | 0 % | +0.17 | 0.000 | 3 % |
  | equilibrium | 94 % | **17 %** | −0.026 | −0.014 | 83 % |
  | balanced + equilibrium | 37 % | **5.6 %** | −0.026 | −0.014 | 75 % |

  The frozen rule's remaining losses are almost all at 0 hands, where the
  population posterior says "exploit" and the opponent happens to be
  balanced. Once the opponent's own hands are observed, the frozen rule
  plays CFR against both equilibrium opponents at 20, 50 and 100 hands.
* **Exploitability.** Every strategy played, by every rule, satisfied its
  budget (max 0.1030 at δ = 0.1, max 0.2524 at δ = 0.25); each was certified
  by the M3 evaluator.
* **Other budgets** give the same picture. At δ = 0.02 the frozen rule is
  +0.03 to +0.09 above mixing on held-out opponents and cuts harm from 4.2 %
  to 0.9 %, at a cost against the point rule that shrinks from 0.05 at
  0 hands to 0.001 by 200 hands.

## 5. Changing opponents

Switching opponents (8 held-out, change after 150 or 250 hands) and a
**larger bait-and-switch population** (24 opponents, 7 exploitable styles,
every switch to a different style after 50, 100 or 200 hands) were played
with change detection (CUSUM, as M6). "pre" is just before the change;
offsets count hands after it. δ = 0.1, 2 seeds.

| scenario | rule | pre | +0 | +20 | +50 | +100 | +200 | harm | worst |
|---|---|---|---|---|---|---|---|---|---|
| switching | point | +0.150 | +0.059 | +0.140 | +0.168 | +0.182 | +0.176 | 6.3 % | −0.024 |
| | **frozen** | +0.144 | +0.051 | +0.110 | +0.161 | +0.172 | +0.165 | **2.5 %** | −0.016 |
| | mix | +0.019 | +0.005 | +0.017 | +0.022 | +0.021 | +0.022 | 8.8 % | −0.018 |
| bait-and-switch (24) | point | +0.495 | +0.192 | +0.347 | +0.404 | +0.432 | +0.454 | 4.6 % | −0.102 |
| | **frozen** | +0.491 | +0.188 | +0.323 | +0.385 | +0.430 | +0.454 | 4.6 % | −0.102 |
| | mix | +0.062 | +0.023 | +0.037 | +0.056 | +0.052 | +0.055 | 7.1 % | −0.026 |

* **Frozen − mix, paired:** switching +0.046 [+0.003, +0.090] at the change
  and +0.151 [+0.102, +0.201] 100 hands later; bait-and-switch +0.164
  [+0.096, +0.233] at the change and +0.378 [+0.270, +0.487] after 100 hands.
* **The failure case is the moment of the switch.** Hero is still playing
  against the old style. The worst run (station → rock after 200 hands)
  loses 0.102 chips/hand at the switch before detection resets the model.
  Detection fires on average 0.9–1.25 times per run. Mixing loses less at
  that moment only because it barely exploits at all.
* A loss *relative to CFR* can slightly exceed δ. The budget bounds hero's
  worst-case value; CFR itself may earn more than the game value against
  the new style.
* The uncertainty rule does not help here. The posterior was confident about
  the *old* style, which is exactly what bait-and-switch exploits. Protection
  comes from the exploitability budget and from change detection, not from
  calibration.
