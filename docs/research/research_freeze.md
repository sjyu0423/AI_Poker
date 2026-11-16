# PokerMind research freeze (v1.0.0, Leduc Hold'em)

This document fixes the adaptive Leduc baseline that the interactive
application will build on. It states what the system is, why each piece was
chosen, and what the evidence does and does not support.
* **Configuration:** `configs/pokermind-research-v1.json`. It is loaded by
  `pokermind.research.load_research_config`, which recomputes its digest.
* **Evidence:** `docs/experiments/m1…m7b_*.md`.

## 1. Architecture

```
observed public actions ──► opponent model (M5/M6)
   (one opponent)            card-conditional policy, Dirichlet posterior
                             latent-card EM + Gibbs samples, CUSUM change detection
                                   │ estimate θ̂, posterior samples θ₁..θ₁₆
                                   ▼
CFR hero (M4 teacher) ──► RNR candidates (M7A, batched in M7B)
  expl 0.0030                 13 values of p, CFR+ 500 iterations
                                   │ exact exploitability + value of every candidate
                                   ▼
                          selection (M7B): budget expl ≤ 0.0030 + δ,
                          95 % calibrated lower bound of gain > 0, else CFR
                                   │ M3 recursive evaluator certifies the pick
                                   ▼
                              strategy played
```

| layer | module | role |
|---|---|---|
| game | `games/leduc.py`, `games/base.py` | rules, matching OpenSpiel's `leduc_poker` (cross-checked) |
| equilibrium | `learning/game_cfr.py`, M4 teacher run | CFR hero, 100,000 iterations, exploitability 0.0030 |
| exact evaluation | `evaluation/game_exact.py`, `robust/batch.TreeEvaluator` | values, best responses, exploitability; the two implementations are independent and agree to 4e-16 |
| opponent model | `opponents/`, `adaptive/reconstruction.py` | public contexts → card-conditional policy with a population prior; posterior samples |
| response | `robust/batch.BatchedRNR` | Restricted Nash Response for a grid of p |
| selection | `robust/selection.Selector` | budget, uncertainty rule, certification |
| configuration | `research.py` | the frozen settings and their digest |

**Information flow is one-way.** The true opponent strategy is used only to
*score* experiments. The acting path (model → RNR → selection) receives
observations, the population prior and the CFR hero, nothing else. A test
(`test_acting_policy_ignores_the_true_opponent`) checks this.

## 2. Models and algorithms

* **CFR (M1, M3).** Vanilla CFR with exact evaluation. The Leduc teacher is
  the 100,000-iteration average strategy (digest `3f1136d4…`).
* **Opponent model (M5, M6).**
  * The opponent's policy is modelled per (private card, public history).
  * Only public actions are observed, so cards at non-showdown hands are
    latent. EM gives the point estimate; data-augmentation Gibbs gives
    posterior samples.
  * The prior is the population-average policy with strength 2 (8 also
    supported).
  * CUSUM on per-hand log-likelihood ratios detects changes and resets the
    model to the last 10 hands.
* **Restricted Nash Response (M7A).** Hero solves a game in which the
  opponent plays the model with probability p and freely otherwise.
  * Hero's average strategy maximises
    `p·v(x, θ̂) − (1 − p)·expl(x)` approximately.
  * Sweeping p traces the trade-off between exploiting the model and
    staying unexploitable.
  * Solved with alternating CFR+ (regrets floored, linear averaging), all p
    at once.
* **Selection (M7B).**
  * Keep the candidates whose exact exploitability is at most
    expl(CFR) + δ (δ = 0.1 frozen, others reported).
  * Score each by the mean posterior gain over CFR minus 1.645 × 1.25 ×
    its posterior sd, with posterior temperature 0.5.
  * Play the best positive score, or CFR if none is positive.
  * Re-certify the chosen strategy's exploitability with the M3 evaluator.

## 3. Why CFR is the foundation

* **It is the only component with a guarantee.** Its exploitability is
  computed exactly (0.0030 chips/hand) and decreases with iterations, as
  measured in M1/M3.
* **It is the safe fallback and the yardstick.** Every adaptive result is
  measured as gain over CFR. Every budget is an exploitability *increment*
  over CFR. When the model is uninformative, the selector returns CFR itself.
* **RNR is CFR on a modified game,** so the solver, the evaluator and the
  OpenSpiel cross-checks carry over unchanged.

## 4. Why neural distillation was not chosen

M2/M4 trained policy networks on the CFR teacher.
* **They can represent it.** A 64-unit network fits all 288 canonical states
  to within 0.0002 of the teacher's exploitability.
* **They don't generalise to states not seen in training.** On held-out
  states the networks were no better than uniform play in exploitability
  (1.32–1.55 vs 1.41 on the random split). Errors are systematic: for
  example, Q/K behaviour is extrapolated to J.
* **In Leduc the table is complete, exact and about 15× faster per
  decision.**

So a network adds approximation error and gives nothing back at this scale.
Distillation remains the path for games where the table cannot be stored.
That is not this freeze.

## 5. Why RNR beat mixing and DBR (M7A, confirmed in M7B)

**Mixing (M6)** blends CFR with a best response to the model. Exploitability
grows almost linearly with the weight of the best response, so a budget
allows only a small weight.

**RNR** instead *solves* for the most exploiting strategy within an
exploitability level. At equal exact budgets it earned about 9× the mixture
on held-out opponents (δ = 0.1, 100 hands: +0.138 vs +0.016). M7B
reproduces this with the frozen rule:
* held-out: frozen − mix +0.115 [+0.075, +0.155];
* out-of-population: +0.537 [+0.307, +0.766].

The result is unchanged at 1,000 and 2,000 solver iterations.

**DBR** restricts the opponent only where it was observed. In Leduc most
information is inferred (cards are hidden unless there is a showdown), so
DBR's per-set counts are weak, and it discards the population prior that RNR
uses. In M7A it beat the mixture but lost to RNR
(−0.075 [−0.100, −0.050], held-out δ = 0.1), and it never beat zero-hand
RNR.

## 6. How uncertainty is represented

| quantity | representation | calibration evidence |
|---|---|---|
| opponent action probabilities | Dirichlet posterior over card-conditional policies, tempered (τ = 0.5), sampled by Gibbs (16 samples) | 90 % intervals cover 89 % with τ = 0.5, κ = 1.25 (81 % untempered) |
| gain of a candidate over CFR | mean ± κ·sd over posterior samples, κ = 1.25 | 51/80/89/93 % coverage at 50/80/90/95 % (development); 77 % at 90 % untempered |
| worst-case loss | exact exploitability of the strategy played (certified) | exact; not statistical |
| opponent change | CUSUM statistic, threshold 5 | 0.9–1.25 resets per switching run |
| experimental results | 95 % CIs across opponents, paired on identical streams | — |

Two kinds of protection are deliberately kept separate.
* **The exploitability budget** is exact and holds against *any* opponent,
  including one that deceives the model.
* **The calibrated lower bound** is statistical. It only makes losing
  against the *modelled* opponent rarer: harm fell from 6.7 % to 2.3 % on
  held-out opponents.

## 7. Supported vs unproven conclusions

**Supported by exact or paired evidence in Leduc:**
1. RNR with a population-prior opponent model beats M6's mixture at every
   budget, on held-out, out-of-population, switching and bait-and-switch
   opponents (M7A, M7B §4–5).
2. The batched solver gives identical selections to M7A's at 1.7× the
   speed. Exact budgets hold for every strategy played (M7B §3–4).
3. 500 CFR+ iterations are enough for decisions at δ ≥ 0.02. δ = 0 is
   noise-dominated (M7B §1).
4. The M6 posterior is overconfident. Tempering plus inflation, chosen on
   development opponents, calibrates it on development data (M7B §2).
5. The calibrated lower-bound rule cuts the share of losing runs against
   held-out opponents about threefold (6.7 % → 2.3 %; equilibrium opponents
   94 % → 17 %). It costs 0.003–0.024 chips/hand, falling with more hands
   (M7B §4).
6. Against in-population opponents most of the gain comes from the
   population model before any personal data. Personal data adds +0.016 by
   200 hands and +0.038 by 500. Against out-of-population opponents it adds
   +0.13 by 50 hands (M7B §4).

**Not established:**
* **Calibration on held-out opponents.** The calibration was fitted and
  measured on development opponents only. Held-out *coverage* was not
  re-measured, only held-out decisions.
* **Any result for real human opponents.** All opponents are simulated
  parametric styles.
* **Anything about Texas Hold'em.** None of these methods has been run beyond
  Leduc. Exact evaluation does not scale there.
* **Speed of change detection under adversarial switching.** The worst
  bait-and-switch loss at the switch is 0.102 chips/hand. It is bounded by
  the budget, not by detection.
* **That the uncertainty rule helps against deception.** It does not: a
  confident posterior about the old style is exactly what bait-and-switch
  exploits.
* **Exact RNR equilibria.** The solver's RNR-game gap is ≤ 0.0031 at
  500 iterations, not 0. "Frontier" statements are approximate to that
  level.

## 8. Reproducing the freeze

Requirements: Python 3.12, `pip install -e ".[dev]"`; optionally
`".[crosscheck]"` for OpenSpiel. Apple M-series figures; runtimes scale with
cores.

```bash
# 1. CFR hero (M4 teacher); its digest must be 3f1136d4…
pokermind train-cfr --game leduc --checkpoints 1000 10000 20000 50000 100000 \
    --output-dir runs/m4-leduc-teacher

# 2. Verify the frozen configuration (recomputes the digest)
pokermind research-config --check

# 3. M7A study (prerequisite for the M7A reproduction check and mixing baseline)
pokermind robust-study --hero runs/m4-leduc-teacher/strategies/strategy_100000.json

# 4. M7B: convergence, calibration (development only), freeze, held-out, benchmark
H=runs/m4-leduc-teacher/strategies/strategy_100000.json
pokermind reliability convergence --hero $H
pokermind reliability calibrate   --hero $H
pokermind research-config --calibration runs/m7b-adaptive-reliability/calibration.json --output runs/research-check.json
pokermind reliability evaluate    --hero $H
pokermind reliability benchmark   --hero $H
```

* **Step 4's `research-config` call** regenerates the selection fields from
  the calibration run. Its digest must equal the committed file's
  (`a67a2826…`). Without `--output` it would overwrite the committed
  file.
* **Determinism.** All runs are seeded. CFR+ and the evaluators are
  deterministic, and the batched solver is bit-identical to the unbatched
  one.
* **Metadata.** Each result JSON records the git commit, package versions and
  platform. `evaluation.json` also embeds the research configuration and its
  digest.
