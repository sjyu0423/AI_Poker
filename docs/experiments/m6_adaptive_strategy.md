# M6 results: adaptive strategy with an exploitability budget

Reproduce (about 43 minutes on 12 CPU cores):

```bash
pokermind adaptive-study --hero runs/m4-leduc-teacher/strategies/strategy_100000.json
```

Numbers come from `runs/m6-adaptive-strategy/metrics.json`. Design and
definitions: `docs/design/adaptive_strategy.md`.

* **Gain:** hero's exact expected payoff per hand minus fixed CFR's, against
  the true opponent, in chips per hand. Seats alternate.
* **Exploitability:** NashConv / 2 of hero's complete policy. CFR: 0.0030.
* **Budget:** exploitability ≤ 0.0030 + 0.1, enforced by exact evaluation.

## 1. Answer to the acceptance question

**Yes, but the improvement is small, and on familiar opponent types it
does not come from learning about the individual opponent.**

* **Held-out opponents** (24, same families as development, new
  parameters):
  * the budgeted adaptive policy (`gated`) gains **+0.016 to +0.020
    chips/hand** over CFR at every observation budget; 95 % CIs exclude 0
    (at 500 hands: +0.0199 [+0.0134, +0.0264]);
  * exploitability is ≤ 0.103 in every run (budget 0.103);
  * that is 59 % of what an oracle that knows the true policy gets under the
    same budget (+0.034), and 5 % of the unconstrained oracle best response
    (+0.39, at exploitability 1.9).
* **Most of that gain is available before observing a single hand.** It
  comes from adapting to the *population* prior (built from development
  opponents). The paired difference "after N hands minus after 0 hands" is
  not significant: −0.002 [−0.005, +0.001] at 20 hands, −0.001 [−0.005,
  +0.003] at 100, +0.002 [−0.002, +0.006] at 500. Within this budget,
  observing these opponents did not measurably add to what the population
  prior gave.
* **Out-of-population opponents** (station, maniac, trapper, card-blind;
  never seen in development): gains +0.045 at 0 hands, rising to **+0.077
  at 500 hands**. The learning effect is significant at 500 hands: +0.032
  [+0.006, +0.057] over 0 hands. That is 67 % of the same-budget oracle
  (+0.114).
* **The budget is the binding constraint.** Gain grows almost linearly with
  the budget. At 500 held-out hands: +0.004 (δ = 0.02), +0.010 (0.05),
  +0.019 (0.1), +0.044 (0.25), +0.081 (0.5). Unbudgeted best responses to
  the estimate gain +0.16 to +0.24, but are exploitable by 1.2–4.6 chips/hand
  and lose money in 15–33 % of runs.

## 2. Opponent-policy reconstruction (validated against the true policies)

Reach-weighted mean total-variation distance between the reconstructed and
the true card-conditional policy (lower is better):

| Hands | 0 | 10 | 20 | 50 | 100 | 200 | 500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Held-out: EM estimate | 0.215 | 0.154 | 0.144 | 0.129 | 0.115 | 0.099 | 0.078 |
| Held-out: population prior alone | 0.215 | | | | | | 0.215 |
| Held-out: CFR prior alone | 0.252 | | | | | | 0.252 |
| Out-of-population: EM estimate | 0.397 | 0.324 | 0.291 | 0.234 | 0.184 | 0.139 | 0.092 |

* The implied public marginals reach a public excess log loss of 0.068 / 0.038
  / 0.018 at 20 / 100 / 500 hands. That is close to M5's direct public model
  (0.060 / 0.030 / 0.014), so the latent-card model fits the observable data
  nearly as well while also producing a card-conditional policy.
* Prior selection by cross-validated likelihood picks the CFR prior about
  ⅓ of the time on held-out opponents (balanced and perturbed families). On
  out-of-population opponents it picks the population prior almost always.
* **Treating public frequencies as card-conditional policies is harmful.**
  The best response to the naive marginal-copy reconstruction loses −0.43
  (0 hands) to −0.19 (500 hands) chips/hand against held-out opponents, in
  74–88 % of runs. Its only success is the `card_blind` family, whose
  marginal *is* its policy (+0.72 to +0.86).

## 3. Payoff gain by observation budget (held-out, δ = 0.1)

Mean gain over CFR [95 % CI over opponents]; "harm" = share of (opponent,
seed) runs with a negative gain.

| Hands | `gated` | harm | `budget_max` | `confidence_budget` | `br_map` (no budget) | its mean expl. | `br_naive` (no budget) |
|---:|---|---:|---|---:|---:|---:|---:|
| 0 | +0.0173 [+0.0104, +0.0242] | 33 % | +0.0173 | +0.0002 | +0.183 | 1.18 | −0.430 |
| 10 | +0.0156 [+0.0096, +0.0216] | 15 % | +0.0155 | −0.0036 | +0.160 | 1.95 | −0.380 |
| 20 | +0.0154 [+0.0104, +0.0205] | 12 % | +0.0154 | −0.0033 | +0.178 | 2.11 | −0.379 |
| 50 | +0.0161 [+0.0109, +0.0213] | 14 % | +0.0159 | −0.0028 | +0.191 | 2.25 | −0.314 |
| 100 | +0.0160 [+0.0107, +0.0213] | 14 % | +0.0158 | −0.0024 | +0.197 | 2.32 | −0.275 |
| 200 | +0.0167 [+0.0112, +0.0223] | 14 % | +0.0167 | +0.0002 | +0.215 | 2.33 | −0.256 |
| 500 | +0.0199 [+0.0134, +0.0264] | 12 % | +0.0195 | +0.0054 | +0.243 | 2.33 | −0.188 |
| oracle, same budget | +0.034 [+0.024, +0.044] | | | | oracle BR +0.391 | 1.92 | |

* `fixed_mix` (α = 0.25) and `evidence_mix` (α = n / (n + 10)) are identical
  to `budget_max` after 10 hands, because the budget caps α at about 0.06–0.09,
  below both rules.
* `gated` beats `budget_max` by a tiny but significant +0.0002 [0.0000,
  +0.0003] at 100 hands and +0.0003 [0.0000, +0.0006] at 500. Its real
  benefit is fewer harmful runs: 15 % vs 24 % at 10 hands, 12 % vs 15 % at
  500.
* `confidence_budget` (trust the model only where it has data, CFR
  elsewhere) is worse than `budget_max` by 0.014–0.019 (paired CIs exclude 0)
  and is negative on average until 200 hands. Mixing a model with CFR per
  information set yields a best response that exploits the CFR part, which
  gains nothing against the real opponent.

### Exploitability

| Policy | Mean / max exploitability |
|---|---|
| CFR | 0.0030 / 0.0030 |
| every budgeted method, every run | ≤ 0.103 (budget 0.1030) |
| `br_map` (unbudgeted) | 1.2–2.3 / 4.63 |
| `br_naive` (unbudgeted) | 2.5–2.8 / 3.95 |
| oracle best response (evaluation only) | 1.92 / – |

The budget search evaluated every accepted mixture exactly. The realized
maximum, 0.1030, sits at the budget because the search bisects to within
the budget, not above it.

## 4. Performance by opponent type

Gain of `gated` at 0 / 20 / 100 / 500 hands (δ = 0.1), with the oracle at
the same budget:

| Family | 0 | 20 | 100 | 500 | Oracle (budget) | Oracle BR (no budget) |
|---|---:|---:|---:|---:|---:|---:|
| tight | +0.009 | +0.012 | +0.018 | +0.024 | +0.030 | +0.556 |
| loose | +0.035 | +0.030 | +0.034 | +0.044 | +0.054 | +0.584 |
| passive | +0.034 | +0.029 | +0.028 | +0.030 | +0.032 | +0.368 |
| aggressive | +0.031 | +0.021 | +0.014 | +0.020 | +0.070 | +0.481 |
| balanced | −0.001 | +0.000 | +0.000 | +0.000 | +0.005 | +0.088 |
| perturbed CFR | −0.004 | +0.001 | +0.001 | +0.002 | +0.011 | +0.271 |
| *out of population* | | | | | | |
| station | +0.038 | +0.041 | +0.048 | +0.062 | +0.094 | +0.803 |
| maniac | +0.065 | +0.080 | +0.179 | +0.153 | +0.247 | +1.967 |
| trapper (deceptive) | +0.050 | +0.047 | +0.043 | +0.057 | +0.062 | +1.571 |
| card-blind | +0.028 | +0.034 | +0.034 | +0.036 | +0.053 | +1.021 |

* **Weak, highly exploitable opponents** (loose, passive, station, maniac)
  give the largest budgeted gains. Tight opponents are the one in-population
  family where learning clearly adds gain (+0.009 → +0.024).
* **Near-balanced opponents give no gain.** The balanced and perturbed-CFR
  families are where adaptation is harmful most often (43 % and 28 % of runs
  negative), but by little: at most −0.006.
* **The deceptive trapper** (bets weak, checks strong) is exploited
  successfully. Showdowns and the full likelihood reveal the inverted
  card-action relationship. The unbudgeted best response to the estimate
  gains +1.20 at 500 hands.

## 5. Changing and misleading opponents

M5's 8 held-out switching opponents × 6 seeds. Gain of `budget_max` against
the *current* policy:

| Hands since switch | 0 | 10 | 20 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| cumulative (no reset) | +0.006 | +0.008 | +0.009 | +0.014 | +0.015 | +0.016 |
| with change detection | +0.006 | **+0.017** | **+0.019** | **+0.022** | +0.021 | +0.023 |
| oracle, same budget | +0.040 | | | | | |

* 47 of 48 switches detected; mean delay 14 hands (p10 6, p90 24); 1.3
  resets per run.
* On a reset, the public model, the reconstruction data and the adaptive
  policy restart from the same retained 10 hands. Tests check this.
* Exploitability stays ≤ 0.103 throughout.

**Bait-and-switch** (station → maniac and maniac → rock at hand 100; 2
opponents × 6 seeds, wide CIs):
* mean gains stay positive with and without detection (+0.03 to +0.11);
  detection was 12 of 12, after 3 hands on average;
* the trap works where expected: for maniac → rock, the cumulative model is
  at −0.023 at +50 hands, while the detecting one is at +0.039 (seed 0).
* Within the budget, the cost of being baited is bounded by the small α. An
  unbudgeted best response tuned to the maniac would call down the rock's
  value raises.

## 6. Cases where adaptation was harmful

1. **Unbudgeted best responses**, even to the principled estimate: 15–33 %
   of runs lose money, and exploitability reaches 4.6 chips/hand.
2. **The naive public-marginal reconstruction**: loses on 74–88 % of
   in-population runs, by up to −0.43 per hand on average.
3. **Budgeted adaptation against near-equilibrium opponents** (balanced,
   perturbed CFR): many slightly negative runs (at most −0.006). There is
   almost nothing to exploit, and estimation noise costs a little.
4. **Budgeted adaptation early against some aggressive or maniac
   opponents**: worst runs −0.023 (aggressive, 10 hands) and −0.035
   (maniac, 20 hands). A few early hands suggested the wrong exploit.
5. **`confidence_budget`**: negative on average until 200 hands.
6. **The posterior gate is not calibrated on in-population opponents.** The
   true gain was at or above the gate's lower bound in only 26–50 % of
   runs (z = 0.5 aims at about 69 %). On out-of-population opponents: 85–94 %.
   The Gibbs posterior's spread (sd 0.006–0.011) understates the real
   uncertainty, just as M5's intervals did. The gate still helped slightly,
   because its threshold was tuned on development data, not because its
   probabilities can be trusted.

## 7. Runtime and memory

| Item | Measurement |
|---|---|
| Full adaptation from 200 hands, one core | 4.9 s: reconstruction and prior selection 0.6 s, three best responses and budget searches 3.6 s, posterior gate 0.7 s |
| …of which one exact exploitability evaluation | ≈ 75 ms; a budget search needs about 10–15 |
| Reconstruction at 500 hands (parallel run) | 2.6 s (CV prior selection runs 4 × 2 EM fits) |
| Peak traced Python memory of one adaptation | 13.8 MB |
| Decision latency of the adaptive policy (a table) | **0.77 µs** per decision |
| Whole study | 2,580 s on 12 cores |

Acting is as fast as fixed CFR, because the adaptive policy is a table. All
the cost is in recomputing it. At about 5 s per update, a real-time agent
should recompute every few dozen hands, or in the background.

## 8. Limitations

* One game (Leduc), one baseline (the M4 CFR), simulated opponents, and a
  population prior built from the development opponents' exact policies.
* Held-out opponents come from the development families. Their gain is
  mostly population-level, and an unfamiliar population would not provide
  it (§1).
* Six seeds per held-out opponent and two bait-and-switch opponents: the
  confidence intervals on the change study are wide.
* The budget bounds exploitability against a best-responding opponent. It
  doesn't model an opponent who adapts to hero's adaptation over time.
* Suit-tying is an approximation for opponents whose policy depends on suits.
