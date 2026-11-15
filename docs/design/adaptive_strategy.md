# Design: opponent-adaptive play with an exploitability budget (M6)

Code: `src/pokermind/adaptive/` and `src/pokermind/experiments/adaptive_study.py`.
Results: `docs/experiments/m6_adaptive_strategy.md`.

## 1. Statistical limitations carried over from M5

| M5 finding | Consequence for M6 | What M6 does |
|---|---|---|
| Rare contexts (facing re-raises, most round-2 lines) have few or no observations even after 100 hands | A best response trusts whatever the model says there, including the prior | A confidence-weighted target falls back to CFR where data are thin; every adaptive policy is subject to an exact exploitability budget |
| Nominal 90 % credible intervals covered 57–87 % | Interval-based decisions are overconfident | Posterior uncertainty is used only through a gate whose threshold is *tuned and then checked* against true gains; its calibration is reported |
| The population prior came from the development opponents' exact policies | Optimistic prior for opponents unlike the population | The prior is chosen per opponent by cross-validated likelihood (population vs CFR mean); out-of-population opponents are tested |
| Opponents switch styles | Cumulative estimates lag; a response tuned to the old style can be costly | M5's CUSUM detector; on an alarm the model, the reconstruction data and the adaptive policy all restart from the last 10 hands |
| The public-context marginal is not the opponent's policy | A best response needs `π(a \| card, context)` | §2 |

## 2. Why public frequencies are not a strategy, and what M6 estimates instead

M5's model estimates `p(a | context) = Σ_c P(c | context) π(a | c, context)`.
Many card-conditional policies give the same marginal. For example, an
opponent who bets 50 % with every card and one who always bets K and never
bets J have the same public betting frequency. Their best responses are
opposite, though: call down the first, fold to the second.

Copying the marginal to every card (the **naive** reconstruction, kept as
a baseline) assumes the opponent's actions carry no information about their
card. A best response to it ignores that a raise signals strength.

M6 instead puts a prior on the card-conditional policy and uses the **full
likelihood** of each observed hand (`adaptive/reconstruction.py`):

* parameters `θ[k]`: one action distribution per canonical opponent
  information set `k` (own rank, public rank, betting history); 288
  parameters, with suits tied;
* prior `θ[k] ~ Dirichlet(s·m[k])`, where `m` is the development population's
  average card-conditional policy or the CFR strategy;
* likelihood of a hand: the opponent's card is latent, uniform over the cards
  hero can't see (hero's own card and the public card are excluded, a card-removal
  effect hero is entitled to). It is summed out:
  `Σ_c P(c) Π_t θ[k(c, context_t)][a_t]`. At a showdown `c` is known.

  Hands ending in a fold are therefore informative but ambiguous. Showdown
  hands are informative and exact. Using the full likelihood avoids the
  selection bias of learning only from shown-down hands.
* estimates: penalized EM (posterior-mean smoothing in the M-step) for a
  point estimate; data-augmentation Gibbs sampling for posterior samples.

Assumptions, all checked on simulated opponents whose true policies only the
evaluator sees (`docs/experiments/m6_adaptive_strategy.md` §2):

1. **Stationarity** within the retained window.
2. **Suit symmetry**: tying suits is exact for the parametric families. It is
   approximate for perturbed-CFR opponents, which differ by suit.
3. **Uniform deal**: true by construction.
4. **The prior's shape matters where data can't identify the policy.**
   Directions the marginal doesn't identify are resolved by the prior plus
   whatever showdowns reveal. That is why the prior is selected per opponent
   by 2-fold cross-validated likelihood among {population, CFR} × strength
   {2, 8}, and why exploitability is controlled separately.

## 3. Candidate hero policies

| Method | Definition | Uses the budget? |
|---|---|---|
| `cfr` | the M4 teacher (exploitability 0.0030) | – |
| `br_naive` | best response to the naive marginal-copy opponent | no (diagnostic only) |
| `br_map` | best response to the EM estimate | no (diagnostic only) |
| `budget_max` | `(1 − α)·CFR + α·br_map`, with the largest `α` whose exact exploitability is within budget | yes |
| `fixed_mix` | the same mixture with `α = min(α_fixed, α*)` | yes |
| `evidence_mix` | `α = min(n / (n + k), α*)`, growing with the hands observed | yes |
| `confidence_budget` | the budgeted mixture with a best response to `w·θ̂ + (1 − w)·π_CFR`, `w = N_k / (N_k + k)` | yes |
| `gated` | the better of `budget_max` and `confidence_budget` by posterior lower bound `mean − z·sd` of the gain over CFR (Gibbs samples); CFR if that bound is ≤ 0 | yes |

Evaluation-only benchmarks (they read the true policy, so they are **not**
playable): `oracle_br`, a best response to the true policy; and
`oracle_budget`, the same mixture rule with the oracle response, at the same
budget.

### Mixtures and the budget

Mixtures are **realization-weighted** (`adaptive/response.py`): hero plays the
CFR strategy for the whole hand with probability `1 − α`, otherwise the
response. In behavioural form,

```
π_α(a|I) = ((1−α) x_cfr(I) π_cfr(a|I) + α x_r(I) π_r(a|I)) / ((1−α) x_cfr(I) + α x_r(I))
```

with `x(I)` the strategy's own reach probability. Two properties follow and are
tested exactly:

* payoff against any fixed opponent is linear in `α`;
* exploitability is convex in `α`.

The budget `exploitability(π_α) ≤ exploitability(CFR) + δ` therefore holds on
an interval `[0, α*]`. `α*` is found by ascending a grid and bisecting, with
**every candidate evaluated exactly** by the M3 evaluator. If even `α = 0`
violated the budget, hero would keep CFR.

The mixture is an experimental policy, not a safety guarantee in any wider
sense. The budget bounds what a best-responding opponent can win against
hero; it says nothing about play outside Leduc or against opponents that
adapt to hero. Main budget: **δ = 0.1 chips/hand**, about 33× CFR's
exploitability and 4 % of the uniform policy's 2.37. The study also reports
δ ∈ {0.02, 0.05, 0.1, 0.25, 0.5} and the unconstrained responses.

## 4. Information boundaries

* The adapter's inputs are `ObserverHandRecord`s (public actions, hero's
  card, the public card, showdown cards), the CFR strategy and priors built
  from **development** opponents. Tests check that hiding the opponent's card
  in a folded hand leaves the record unchanged.
* The true opponent policy is read only by `oracle_for` and
  `evaluate_snapshot` in the study, to compute exact payoffs, exploitability,
  reconstruction error and oracle benchmarks.
* Observation hands come from a stream keyed by `(opponent, seed,
  "observe")`. Evaluation is exact (the expectation over all future hands),
  so no evaluation hand is ever observed.

## 5. Experimental protocol

1. **Development:** 24 development opponents (M5 families) × 3 seeds × hands
   {0, 10, 20, 50, 100, 200, 500}. Tunes `α_fixed`, evidence `k`, confidence
   `k` and gate `z` to maximize the mean exact gain over CFR with the budget
   enforced.
2. **Frozen held-out:** the 24 held-out opponents of the same families (new
   parameters) and 8 out-of-population opponents (station, maniac, trapper,
   card-blind; families never seen in development) × 6 seeds.
3. **Changing opponents:** M5's 8 held-out switching opponents and 2
   bait-and-switch opponents (station → maniac, maniac → rock at hand 100)
   × 6 seeds, with and without change detection. Gains are measured
   against the opponent's *current* policy.

Confidence intervals are over opponents (each opponent's seeds are
averaged first). Paired comparisons use the same observation stream for
both methods.
