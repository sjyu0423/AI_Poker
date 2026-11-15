# M5 results: how much can we learn about an unfamiliar Leduc opponent?

Reproduce (about 8 minutes on an Apple M-series CPU):

```bash
pokermind opponent-study --hero runs/m4-leduc-teacher/strategies/strategy_100000.json
```

All numbers come from `runs/m5-opponent-modeling/metrics.json`. Design,
definitions and leakage rules: `docs/design/opponent_modeling.md`.

## 1. Setup

* **Hero:** the M4 CFR teacher (100k iterations, digest `3f1136d4…`).
  Hero and opponent alternate seats every hand. The opponent makes about
  2.1 decisions per hand (range 1.9–2.4 across opponents).
* **Opponents:** 6 stationary families × 4 opponents in each split, plus 8
  switching opponents per split. Development and held-out opponents are
  drawn from different seeds; no policy occurs in both splits.

  | Family | Definition (equity `s` = showdown equity of the opponent's own card) |
  |---|---|
  | tight | folds to a bet unless `s` > τ, τ ∈ [0.55, 0.70]; medium aggression |
  | loose | τ ∈ [0, 0.15]: continues with almost anything |
  | passive | aggression α ∈ [0.05, 0.2], no bluffs: mostly checks and calls |
  | aggressive | α ∈ [0.8, 1.0], bluff share β ∈ [0.2, 0.4]: bets and raises a lot |
  | balanced | the CFR teacher mixed with ≤ 5 % uniform play |
  | perturbed_cfr | every information set redrawn from Dirichlet(c·p_CFR + 0.1), c ∈ [5, 30] |
  | switching | one family's policy until hand 150 or 250, then another family's |

  Parametric opponents also tremble (ε ∈ [1 %, 5 %] uniform play).
* **Budgets:** for each held-out stationary opponent, 30 seeds. Each seed has
  an estimation stream of 500 hands and an *independent* evaluation stream
  of 500 future hands.
* **Uncertainty:** 95 % confidence intervals are over the 24 opponents (the
  mean of each opponent's 30 seeds). "p10/p90" are run-level percentiles
  over all 720 (opponent, seed) runs.

### Selected model (tuned on development opponents only)

| Candidate | Mean excess log loss on dev (10–500 hands) |
|---|---:|
| **hierarchical back-off + population prior, α_fine 16, α_coarse 64** | **0.0403** |
| flat per-context Dirichlet + population prior, α 4 | 0.0512 |
| hierarchical back-off + uniform prior, α 32 / 16 | 0.0538 |
| flat per-context Dirichlet + uniform prior, α 2 | 0.0778 |

Both ingredients help: the population prior (learned from development
opponents) and the hierarchical fallback for rarely seen contexts. α_coarse = 64
is the largest value in the grid; a larger value was not tried.

## 2. Prediction quality by number of hands observed (held-out opponents)

**Excess log loss** = expected log loss minus that of the oracle public
model (the best any public-information predictor can do), in nats per
opponent decision. It is computed exactly over future hands.

| Hands | Excess log loss (95 % CI) | p10 – p90 over runs | Accuracy | Sampled future log loss | Sampled Brier |
|---:|---:|---:|---:|---:|---:|
| 0 (prior only) | 0.123 [0.106, 0.140] | 0.088 – 0.207 | 0.600 | 0.746 | 0.488 |
| 10 | 0.080 [0.073, 0.087] | 0.050 – 0.115 | 0.641 | 0.704 | 0.454 |
| **20** | **0.060 [0.056, 0.064]** | 0.037 – 0.086 | 0.657 | 0.684 | 0.439 |
| **50** | **0.039 [0.035, 0.044]** | 0.021 – 0.060 | 0.669 | 0.663 | 0.424 |
| **100** | **0.030 [0.026, 0.033]** | 0.016 – 0.046 | 0.676 | 0.654 | 0.417 |
| 200 | 0.022 [0.019, 0.025] | 0.012 – 0.035 | 0.679 | 0.647 | 0.412 |
| 500 | 0.014 [0.012, 0.016] | 0.008 – 0.023 | 0.683 | 0.638 | 0.407 |
| oracle (public) | 0 | | 0.691 | 0.624 (exact) | |
| oracle that also sees the opponent's card | −0.156 | | | 0.468 (exact) | |

* The sampled future-hand scores agree with the exact expectations to within
  about 0.001 (exact log loss at 100 hands: 0.653; sampled: 0.654).
* The development opponents give the same curve (0.118, 0.078, 0.060, 0.039,
  0.030, 0.022, 0.014), so the model generalizes to the held-out population.
* Share of the learnable gap closed (from the population prior's 0.123 down
  to the oracle's 0): 35 % after 10 hands, **51 % after 20, 68 % after 50,
  76 % after 100**, 88 % after 500.
* **Hidden information dominates.** Even a perfect public model is 0.156
  nats per decision worse than one that sees the opponent's card. That is
  more than the whole learning curve. Prediction from public actions has a
  hard ceiling: best possible accuracy 0.691.

### Compared with simpler estimators (excess log loss)

| Hands | Uniform | Pop. prior only | Raw frequency | Add-one (Laplace) | Flat + pop. prior | Hierarchical, uniform prior | **Selected** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.200 | 0.123 | 3.199 | 0.141 | 0.092 | 0.099 | **0.080** |
| 20 | 0.200 | 0.123 | 2.353 | 0.114 | 0.076 | 0.075 | **0.060** |
| 50 | 0.200 | 0.123 | 1.804 | 0.080 | 0.056 | 0.055 | **0.039** |
| 100 | 0.200 | 0.123 | 1.494 | 0.058 | 0.043 | 0.043 | **0.030** |
| 500 | 0.200 | 0.123 | 0.492 | 0.024 | 0.019 | 0.020 | **0.014** |

* Raw counts are disastrous for probabilities: one unseen action in a
  context gets probability 0, and the log loss explodes. Their accuracy is
  fine (0.627 at 20 hands), but smoothing is what makes the probabilities
  usable.
* Add-one smoothing only beats the population prior after about 25 hands.

### By betting context (selected model)

| Context class | Opponent decisions per hand | Seen after 20 / 100 / 500 hands | Excess log loss at 20 / 100 / 500 |
|---|---:|---:|---|
| R1, check or bet (`cr`) | 0.74 | 15 / 74 / 368 | 0.038 / 0.011 / 0.003 |
| R1, facing a bet (`fcr`) | 0.51 | 10 / 51 / 253 | 0.073 / 0.029 / 0.008 |
| R1, facing a re-raise (`fc`) | 0.07 | 1.3 / 6.5 / 33 | 0.121 / 0.076 / 0.019 |
| R2, check or bet | 0.60 | 12 / 59 / 298 | 0.057 / 0.037 / 0.022 |
| R2, facing a bet | 0.17 | 3.5 / 18 / 87 | 0.087 / 0.053 / 0.039 |
| R2, facing a re-raise | 0.02 | 0.5 / 2.1 / 11 | 0.201 / 0.193 / 0.150 |

Round-1 situations are learned quickly. Round-2 contexts are split by public
card and both rounds' histories (90 contexts), so each gets few observations.
Facing a re-raise in round 2 is still poorly predicted after 500 hands, since
it is seen only about 11 times.

### Summary tendencies (mean absolute error; p90 in brackets)

| Hands | Fold to a bet | Raise when facing a bet | Bet when unopened |
|---:|---:|---:|---:|
| 0 | 0.149 (0.336) | 0.080 (0.170) | 0.158 (0.267) |
| 20 | 0.091 (0.187) | 0.056 (0.109) | 0.071 (0.132) |
| 50 | 0.061 (0.118) | 0.044 (0.094) | 0.042 (0.084) |
| 100 | 0.041 (0.084) | 0.034 (0.077) | 0.029 (0.059) |
| 500 | 0.019 (0.039) | 0.017 (0.039) | 0.013 (0.027) |

(Reach-weighted over the contexts where the action is legal; "fold to a bet"
covers both `fcr` and `fc` contexts.)

### By opponent family (selected model, excess log loss)

| Family | 0 | 20 | 50 | 100 | 500 | p90 at 50 hands |
|---|---:|---:|---:|---:|---:|---:|
| tight | 0.105 | 0.045 | 0.030 | 0.024 | 0.011 | 0.046 |
| loose | 0.128 | 0.068 | 0.040 | 0.027 | 0.012 | 0.056 |
| passive | 0.201 | 0.065 | 0.027 | 0.018 | 0.009 | 0.041 |
| aggressive | 0.114 | 0.056 | 0.037 | 0.027 | 0.012 | 0.051 |
| balanced | 0.090 | 0.057 | 0.045 | 0.036 | 0.019 | 0.062 |
| perturbed CFR | 0.099 | 0.069 | 0.057 | 0.045 | 0.023 | 0.073 |

Simple, strongly skewed styles are learned fastest. Passive opponents start
furthest from the population prior but are the easiest after 50 hands.
Near-equilibrium and randomly perturbed opponents mix their actions and
deviate in context-specific ways, so they are learned slowest.

## 3. Calibration and uncertainty

* **Probabilities are well calibrated.** On the future hands (1.76 million
  (decision, action) pairs per budget), the expected calibration error is
  0.011 / 0.016 / 0.007 / 0.009 / 0.005 at 0 / 20 / 50 / 100 / 500 hands.
  After 20 hands it is slightly underconfident at the extremes (predicted
  0.91 → observed 0.95).
* **The credible intervals are too narrow when there is little data.** The
  nominal 90 % intervals contain the true public-context probability for
  57 % (0 hands), 74 % (20), 82 % (50), 84 % (100) and 87 % (500) of the
  reach-weighted (context, action) pairs. At 20 hands, 10 % of runs have
  coverage below 53 %.
* Why: the interval treats the back-off prior mean as known, and α_fine = 16
  is a strong prior, so the posterior is confident before the data justify
  it. Use the intervals as a relative measure of certainty, not as a
  calibrated 90 %.
* The mean posterior standard deviation per action falls from 0.110 (0
  hands) to 0.096 (20), 0.072 (100) and 0.048 (500).

## 4. Opponents that change

Eight held-out switching opponents (for example passive → tight at hand
150, loose → aggressive at hand 250) × 30 seeds. The forgetting rate and the
detector were tuned on 8 development switching opponents plus the 24
development stationary ones, so false alarms count against them.
Excess log loss is measured against the *current* policy; offset 0 is
"just before the first post-switch hand".

| Hands since switch | Cumulative | Exponential forgetting (γ = 0.98 per hand) | CUSUM + reset (short memory 0.8, threshold 5) |
|---:|---:|---:|---:|
| 0 | 0.315 | 0.258 | 0.311 |
| 10 | 0.283 | 0.187 | 0.183 |
| 20 | 0.255 | 0.140 | **0.093** |
| 50 | 0.197 | 0.068 | **0.036** |
| 100 | 0.142 | 0.035 | **0.026** |
| 200 | 0.088 | 0.023 | **0.021** |

| Detector | Value |
|---|---|
| Switches detected | 235 / 240 (5 missed within 250–350 hands) |
| Detection delay | median **11 hands**; mean 15.2 [13.7, 16.7]; p10 5, p90 30 |
| False alarms before the switch | 18 in 48,000 hands (0.4 per 1,000 hands) |
| False alarms on stationary opponents | 0.44 per 500 hands; **40 % of 500-hand runs have at least one**. Spread over the session (median at hand 278); passive opponents trigger most (0.78 per run) |
| Cost on stationary opponents (excess at 500 hands) | cumulative 0.014; CUSUM 0.020; forgetting 0.031 |

* Cumulative counts adapt very slowly: still 0.09 nats worse than
  necessary 200 hands after the switch.
* Forgetting adapts but permanently forgets useful data. On stationary
  opponents it doubles the error at 500 hands.
* CUSUM gives the best post-switch error from about 20 hands on, at a
  smaller stationary cost. Its false alarms are not rare, but each one only
  restarts the counts from the recent hands.

## 5. Answer: what can be learned after 20, 50 or 100 hands?

Against this population, with hero playing near-equilibrium:

* **20 hands (about 42 opponent decisions):** about half of the learnable
  information beyond the population prior. Top-action accuracy is 0.657
  (ceiling 0.691). Broad tendencies are estimated to within about ±0.07–0.09
  on average, but one run in ten is off by 0.13–0.19. Rare situations
  (facing re-raises, most round-2 lines) are still mostly prior.
* **50 hands:** about two thirds of the learnable gap. Summary tendencies to
  within ±0.04–0.06 on average (p90 ±0.08–0.12). Round-1 behaviour is
  reasonably pinned down; round-2 behaviour is not.
* **100 hands:** about three quarters. Tendencies within ±0.03–0.04 (p90
  ±0.06–0.08). Round-2 facing-bet contexts have only about 18 observations,
  and the rarest are essentially unlearned.

None of these numbers is a guarantee. They are averages over a simulated
population and one hero. The p90 columns show that individual runs are often
much worse, and an opponent outside the simulated families may be learned
more slowly (the perturbed-CFR family already is). Public actions alone can
never reveal what the opponent holds when they act. That gap (0.156 nats
per decision) is larger than everything learned from 0 to 500 hands.

## 6. Limitations

* One hero (near-equilibrium). A more exploitative or a passive hero would
  produce different contexts and frequencies, and the estimates are only
  valid for the hero that produced them.
* Simulated opponents are stylized: equity-threshold policies plus noise
  around CFR. Real players condition on things these opponents don't (for
  example the previous hand's result).
* The population prior is built from the development opponents' **exact**
  policies (an idealization of "experience with many past opponents").
  With a real population it would have to be estimated from data.
* Showdown cards are recorded but not used. They are a biased sample of
  hands, and using them correctly needs a model of which hands reach a
  showdown.
* Credible intervals are too narrow with little data (§3).
* CUSUM false alarms are fairly frequent (40 % of 500-hand sessions).
* The tuning grid edge (α_coarse = 64) was reached.
