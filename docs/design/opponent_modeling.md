# Design: modelling an unfamiliar Leduc opponent from public information (M5)

Code: `src/pokermind/opponents/`. Results: `docs/experiments/m5_opponent_modeling.md`.

M5 asks a narrow question: after watching an opponent for N hands, how well
can we predict their **next actions**? Adapting our own strategy to the
prediction is M6. Everything here uses the verified Leduc engine (M3).

## 1. What is observable

From one player's ("hero's") seat, a hand of Leduc reveals:

| Information | Observable by hero? |
|---|---|
| Every betting action of both players (fold, call/check, raise/bet) | **yes**, as it happens |
| Order of play, round boundaries, contributions, pot | **yes** (functions of the public actions) |
| The public card (round 2) | **yes**, once dealt |
| Hero's own private card | yes (to hero only) |
| Opponent's private card | **no**, unless the hand reaches a showdown |
| Opponent's card in a hand that ends with a fold | **never** |
| Opponent's policy (their probabilities) | **never**; only sampled actions |

So the opponent's actions are always public, but the *reason* for each
action (their card) is hidden in every hand that ends with a fold, and is
revealed only at a showdown, which is a biased subset of hands (hands both
players were willing to take to the end).

## 2. Observation opportunity and betting context

An **observation opportunity** is a decision node where the opponent is to
act. It is described by its **public context**: everything public at that
moment, and nothing else:

```
round 1:  R1:<round-1 actions>                 e.g. "R1:cr"   (opponent faces a raise after checking)
round 2:  R2:<public rank>:<round-1>:<round-2>  e.g. "R2:K:rc:r"
```

The public card enters by **rank only** (suits never matter in Leduc). The
seat is implied by the history (player 0 opens each round), so contexts of
seat 0 and seat 1 are distinct strings. Leduc has **96 opponent contexts**:
6 in round 1 and 90 in round 2 (3 public ranks × 5 round-1 lines × 6
round-2 histories), 48 for each seat.

The legal actions are a function of the current-round history (fold only
when facing a bet; raise only below the cap of 2), so every context has a
fixed action set: `cr` (check or bet), `fcr` (facing a bet, may re-raise) or
`fc` (facing the second raise).

## 3. What a public-information model can estimate

The opponent plays `π(a | own card, public context)`. We never see the card,
so the best any public-only model can do is the **public-context marginal**

```
p(a | context) = Σ_card P(card | context) · π(a | card, context)
```

where `P(card | context)` is the probability of each opponent card *given
that this public context was reached*. It depends on the deal, on the
opponent's own earlier actions (a player who raised is more likely to hold a
strong card) and on **hero's policy**, which decides which contexts are
reached. Consequences:

* The prediction target is the public-context marginal, computed exactly by
  walking the game tree with the true policies and chance probabilities
  (`opponents/exact.py`). It is the **oracle public model**: no model that
  sees only public information can beat its expected log loss.
* A model that knew the opponent's card would do better (the **card-aware
  oracle**). The gap between the two is information that is hidden in
  principle, not a modelling failure.
* Estimates are specific to hero's policy. The same opponent faced by a
  different hero visits contexts with different frequencies and different
  card distributions. M5 fixes hero to the M4 CFR teacher and alternates
  seats every hand.
* Betting history matters twice: it determines which actions are legal, and
  it changes `P(card | context)`. After `r r` the opponent's range is
  stronger than after `c`, so the same opponent folds less often there.
  The model therefore keeps the full public history as its finest context.

## 4. The model (`opponents/bayes.py`)

Dirichlet-categorical estimates per context, with hierarchical back-off
through four context levels that share the same legal action set:

| Level | Key | Number |
|---|---|---:|
| 0 (finest) | full public context, e.g. `R2:K:rc:r` | 96 |
| 1 | round + current-round history, e.g. `R2:r` | 12 |
| 2 | round + action set, e.g. `R2:fcr` | 6 |
| 3 | action set, e.g. `fcr` | 3 |

Counts are updated at every level after each observed opponent action. The
prediction is computed coarse to fine:

```
q_3 = (n_3 + α_c · b_3) / (N_3 + α_c)
q_ℓ = (n_ℓ + α_ℓ · b_ℓ) / (N_ℓ + α_ℓ),   b_ℓ = normalize(pop_ℓ · q_{ℓ+1} / pop_{ℓ+1})
```

* `n_ℓ` are the (possibly decayed) action counts of that level's key and
  `N_ℓ` their total.
* `pop_ℓ` is the **population prior**: the average public-context marginal
  of the *development* opponents at that level. With a uniform population
  (`pop = uniform` everywhere) the formula reduces to plain hierarchical
  back-off.
* The ratio `q_{ℓ+1} / pop_{ℓ+1}` carries this opponent's observed deviation
  from the population at the coarser level into the finer context ("folds
  20 % more than average when facing a bet"), so a rarely seen context
  inherits the opponent's general tendency rather than the population's.
* With no observations, `q_0 = pop_0`: the model predicts the population
  average (or uniform).
* Uncertainty: the posterior at the finest level is `Dirichlet(n_0 + α_0 b_0)`,
  whose action marginals are Beta distributions. The model reports the mean,
  standard deviation and a central 90 % credible interval for each action.
  This treats `b_0` as fixed, so it can be overconfident; calibration of the
  interval is measured, not assumed.

Recency: optionally every count is multiplied by `γ` before each new hand
(exponential forgetting; `γ = 1` is cumulative). Change detection: a CUSUM on
the log-score of each new action under a short-memory model versus the
long-memory model; when it crosses a threshold the long-memory counts are
replaced by the short-memory ones.

Hyperparameters (`α`, the prior, `γ`, the detector) are chosen on the
development opponents only.

## 5. Evaluating accuracy

For a model snapshot after N hands:

* **Future hands, sampled.** A separate stream of future hands (independent
  seed) against the same opponent. At each opponent decision, the snapshot's
  prediction is scored against the action actually taken: log loss, Brier
  score, top-1 accuracy. The snapshot never sees these hands.
* **Future hands, exact expectation.** The same quantities in expectation
  over the infinite stream of future hands, computed from the exact
  public-context marginal and the probability of reaching each context. This
  removes sampling noise from the evaluation. The **excess log loss** over
  the oracle public model is the KL divergence from the best achievable
  public prediction, weighted by how often each context occurs.
* **Calibration:** reliability bins of predicted probabilities against
  observed frequencies on the sampled future hands (expected calibration
  error), and the coverage of the 90 % credible intervals against the exact
  marginals.
* **Per context:** metrics by context class (round × action set) and the
  number of opportunities actually observed.
* **Variation:** many seeds per opponent and many opponents per family, with
  95 % confidence intervals over (opponent, seed) runs.

## 6. Avoiding leakage

* The model's only input is a `PublicHandRecord` built by the dealer-side
  function `public_record`. It contains each opponent decision's public
  context, legal actions and action, and the opponent's card **only** if the
  hand reached a showdown. Tests check that hands ending in a fold produce
  identical records whatever the opponent's card was. The M5 model doesn't
  use showdown cards at all.
* Hero's own card is not given to the model either, although it is
  legitimately known (it would add card-removal information).
* Estimation hands and evaluation hands come from independent random
  streams. The exact evaluation uses the true policy, which the model never
  sees.
* Opponents are split into **development** opponents (used to choose
  hyperparameters and to build the population prior) and **held-out**
  opponents (used only for the reported results). Held-out opponents have
  different parameters drawn from different seeds.
