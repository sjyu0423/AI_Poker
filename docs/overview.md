# PokerMind: an overview

PokerMind is a local poker AI that plays heads-up **Leduc Hold'em** against
you. It **learns your tendencies as you play**, and it adjusts its strategy
to exploit them **without ever becoming more exploitable than a fixed,
exactly verified limit**. This page explains the ideas in plain language;
each section links to the technical write-up and measured results.

> **What kind of evidence is on this page.** Three kinds of claims appear,
> and they are kept apart:
> * **Guarantees.** Mathematical facts, checked exactly by computer, such as
>   "this strategy can lose at most 0.103 chips per hand to a perfect
>   counter-strategy".
> * **Measured results.** Experiments against *simulated* opponents, with
>   confidence intervals.
> * **Goals.** Untested (real human players, Texas Hold'em). PokerMind
>   claims nothing about these.

## 1. The problem

A poker bot can aim for one of two things:
* **Unbeatable.** It plays a game-theoretic equilibrium. It cannot be
  exploited, but it also doesn't punish an opponent's mistakes.
* **Exploitative.** It models the opponent and plays the best reply. It wins
  more against a predictable player, but a wrong model, or an opponent who
  changes style, can make it lose badly.

PokerMind's question is **how much you can safely gain by adapting to one
particular opponent, from a few dozen hands of observation, while
guaranteeing that the adapted strategy stays close to unbeatable.**

## 2. Why adapting to an opponent is interesting

* **Hidden information.** You never see what the opponent held when they
  folded, so their tendencies must be *inferred*.
* **Little data.** A model must be useful after tens of hands, not millions.
* **Deception and change.** An opponent can play one way and then switch,
  and a confident model is exactly what such a switch exploits.

Handling all three needs a statistical model of the player, a strategy
optimizer that trades exploitation against safety, and a way to *prove* the
safety part.

## 3. Leduc Hold'em, and why it was chosen

Leduc is a small research poker game:
* a six-card deck (J, Q, K in two suits);
* one private card each and one public card;
* two betting rounds;
* at most 13 chips won or lost in a hand.

It keeps poker's essentials: hidden cards, bluffing, betting rounds, and a
public card that changes hand strength. Its game tree is small enough
(936 decision situations, 9,457 nodes) that strategies can be
**evaluated exactly**, with no sampling noise. That is what makes the
guarantees in this project possible. The engine matches OpenSpiel's
`leduc_poker` and is cross-checked against it.
→ [rules](games/leduc.md)

## 4. CFR and Nash equilibrium

* **Nash equilibrium.** A pair of strategies where neither player can gain
  by changing theirs. In two-player zero-sum poker, playing your half of an
  equilibrium guarantees the game's value on average, whatever the opponent
  does.
* **Counterfactual Regret Minimization (CFR).** A self-play algorithm that
  converges to one. At each decision it tracks how much better each action
  would have done ("regret"), and shifts probability toward the regretted
  actions.
* **Exploitability.** How much a *perfect* counter-strategy would win per
  hand. It is 0 exactly at equilibrium.

**Measured:**
* On Kuhn Poker (a 3-card warm-up game), 50,000 CFR iterations reach
  exploitability 0.0010 chips/hand and reproduce the known game value
  (−1/18) to 4×10⁻⁷.
* On Leduc, 100,000 iterations (31 minutes, deterministic) reach
  **0.0030 chips/hand**. This strategy is PokerMind's baseline: the "hero".

→ [CFR explained with a worked example](concepts/cfr.md)

## 5. Neural policy experiments, and their limits

Can a small neural network replace the strategy table?
* **It can memorise it.** A 64-unit network trained on all 288 canonical
  Leduc states plays within 0.0002 chips/hand of the table's exploitability.
* **It does not generalise.** Trained on some states and asked about unseen
  ones, the networks were *no better than uniform random play* in
  exploitability (1.32–1.55 vs 1.41 chips/hand). Their mistakes were
  systematic; for example, they applied Queen and King behaviour to Jacks.
* **The table wins at this scale.** It is exact, about 15× faster per
  decision, and only 95 KB.

So neural networks were **not** used in the final system. They matter only
when a game is too large for a table, and these results show that
generalisation then needs more structure than a plain network.
→ [M4 results](experiments/m4_leduc_neural.md)

## 6. Bayesian opponent modelling

PokerMind models *how often you fold, call or raise in each betting
situation*. It starts from a **prior**: the average behaviour of a
population of simulated "typical" players. Each decision you make moves the
estimate toward your own frequencies. With little data, the estimate stays
near the prior. Cards you show down help it infer what you held when you
acted; a folded card is never seen.

**Measured** (against simulated opponents, with PokerMind playing
equilibrium):
* About **half** of what is learnable beyond the prior is learned after
  20 hands, and about **three quarters** after 100.
* Predicted probabilities are well calibrated (expected calibration error
  ≤ 0.016).
* **But** the model's credible intervals are too narrow with little data:
  nominal 90 % intervals contain the truth only 74 % of the time after
  20 hands.
* **Hidden cards set a hard limit.** Even a perfect model of public actions
  is 0.156 nats per decision short of knowing the cards, which is more than
  all the learning from 0 to 500 hands.

→ [M5 results](experiments/m5_opponent_modeling.md)

## 7. Restricted Nash Response: exploiting safely

Exploiting a model naively is risky. A pure best response to the estimate
gained +0.16 to +0.24 chips/hand in tests, but was itself exploitable by
1.2–4.6 chips/hand and lost money in 15–33 % of runs.

**Restricted Nash Response (RNR)** solves a modified game:
* with probability *p*, the opponent is forced to play like the model;
* otherwise, the opponent plays its best counter-strategy.

So *p* is a dial between "trust the model" and "assume the worst". PokerMind
solves a grid of *p* values, keeps only strategies within the
exploitability budget, and picks one by a **calibrated lower confidence
bound** on its gain. If no candidate is confidently better than the
baseline, it keeps the baseline.

**Measured** (held-out simulated opponents, budget +0.1 chips/hand, after
100 hands; 95 % CIs):

| method | gain over the equilibrium baseline (chips/hand) |
|---|---|
| Mixing equilibrium with a best response (M6) | +0.016 |
| **RNR, frozen v1 rule** | **+0.131** (vs mixing: +0.115 [+0.075, +0.155]) |
| RNR on out-of-population opponent styles | +0.619 (vs mixing: +0.537 [+0.307, +0.766]) |

The calibrated rule cut the share of losing runs from **6.7 % to 2.3 %**,
at a cost of up to 0.024 chips/hand that shrinks with more hands.
→ [M7A](experiments/m7a_robust_response.md),
[M7B](experiments/m7b_adaptive_reliability.md),
[design](design/robust_response.md)

## 8. Exact exploitability certification

Before any new strategy is played, its exploitability is computed **exactly**
by walking the whole game tree. It must be within the budget:
* the baseline's 0.0030 chips/hand;
* plus 0.1;
* giving **0.1030 chips/hand**.

This is the project's central **guarantee**: whatever the opponent does,
the AI's worst-case loss rate is bounded by that number. It holds even if
the opponent model is completely wrong.

Two independent exact evaluators agree to 4×10⁻¹⁶. The game engine and
solver are cross-checked against OpenSpiel.

What it does *not* guarantee: that adapting actually wins more against a
particular person. That part is statistical, and was measured only against
simulated players.

## 9. Application architecture

```
Browser (Next.js, React, TypeScript)        no game logic: shows server state only
   │  /api/* (same-origin proxy)
FastAPI backend (Python)                    sessions, legality, hidden cards, payouts
   ├─ Leduc engine ─ CFR hero (certified 0.0030)
   ├─ opponent model ◄── what the AI legitimately saw after each hand
   └─ background worker process ── RNR re-solve (~7.5 s) ── exact re-certification
                                     └─► atomic install of the new strategy
```

* **Every setting comes from one frozen, digest-checked configuration**
  (`configs/pokermind-research-v1.json`).
* **The server is the source of truth.** It owns the deck, legality,
  settlement and observations. The browser never sees the AI's card before a
  showdown, and the AI never sees a folded card.
* **Strategy updates run in the background**, so actions stay at about
  0.5 ms. Each new strategy is installed only after it is re-certified.
* **The learning dashboard** separates *observed* counts, *estimated*
  tendencies and *certified* strategy facts.

→ [API contract](design/app_backend.md), [web app](../web/README.md)

## 10. Results you can reproduce, and the limitations

**Reproducible:**
* Every experiment is seeded and deterministic.
* Each milestone's results document lists the exact commands, and
  [research_freeze.md](research/research_freeze.md) shows how to regenerate
  the frozen configuration and check its digest.

**Limitations, stated plainly:**
* **No human-player evidence.** All opponents were simulated. Results
  against real people, and the calibration of the model for them, are
  untested.
* **Leduc only.** Exact evaluation does not scale to Texas Hold'em. None of
  this has been run there.
* **Approximate equilibria.** The RNR solver converges approximately (gap
  ≤ 0.0031 at the frozen setting). The budget check, by contrast, is exact.
* **Deceptive opponents still cost something.** In bait-and-switch tests the
  worst loss at the moment of a switch was 0.102 chips/hand. The budget
  bounds it; detection does not prevent it.
* **Local, in-memory app.** There are no accounts, and sessions are lost
  when the server restarts.
