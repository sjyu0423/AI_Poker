# Design: Restricted Nash Response and Data-Biased Response for Leduc (M7A)

Code: `src/pokermind/robust/`. Results: `docs/experiments/m7a_robust_response.md`.

References:
* M. Johanson, M. Zinkevich, M. Bowling, *Computing Robust Counter-Strategies*,
  NIPS 2007 (Restricted Nash Response, RNR).
* M. Johanson, M. Bowling, *Data Biased Robust Counter Strategies*, AISTATS 2009
  (DBR).

## 1. The problem

Given a model `σ_fix` of the opponent, hero wants to win more than a Nash
strategy does against an opponent like `σ_fix`, without becoming very
exploitable if the model is wrong. The two extremes are:
* **Nash:** about 0 exploitability, and no exploitation of the model;
* **best response to `σ_fix`:** maximal exploitation, and in M6 an
  exploitability of 1–4 chips/hand.

M6 interpolated between them with a mixed strategy, `(1 − α)·CFR + α·BR`.
Its payoff is linear in α and its exploitability convex in α, so the mixture
traces a straight *chord* between the two end points in (exploitability,
payoff) space.

## 2. Restricted Nash Response

**The modified game.** Before the hand, a chance event that only the
opponent observes decides the opponent's mode:
* with probability `p` the opponent is **restricted**: it must play
  `σ_fix`;
* with probability `1 − p` it is **free**: it plays any strategy.

Hero never learns the mode. RNR computes a Nash equilibrium of this game and
keeps hero's part, `σ*_p`.

In this implementation (`robust/modified_game.py` writes the game out;
`robust/solver.py` solves it):
* hero's information sets are its Leduc information sets, so the coin is
  hidden from hero;
* the restricted opponent's decision points become chance nodes with
  `σ_fix`'s probabilities;
* the free opponent has its own information sets.

The game has perfect recall. Hero sits in either seat with probability ½.
Hero's seat-0 and seat-1 information sets never meet, so the problem splits
into two independent solves.

**What it guarantees** (Johanson et al., for an *exact* equilibrium):
* `σ*_p` is a best response to `σ_fix` among all strategies whose
  exploitability is at most that of `σ*_p` itself;
* the family `{σ*_p : p ∈ [0, 1]}` traces the Pareto frontier of
  (exploitability, payoff against `σ_fix`). At `p = 0` it is a Nash
  equilibrium; at `p = 1` it is a best response to `σ_fix`.

What it does **not** guarantee:
* any particular exploitability for a requested `p`. The mapping from `p`
  to exploitability is unknown in advance and is measured here;
* anything about payoff against the *true* opponent, which differs from
  `σ_fix`;
* exactness of an approximate solve. CFR+ with 500 iterations gives an
  approximate equilibrium, so the frontier property holds only
  approximately.

**Why it beats mixing.** The frontier is concave, and the mixture is a chord
under it. RNR doesn't mix two fixed strategies. It changes hero's play at every
information set, exploiting `σ_fix` where that is cheap in exploitability and
staying balanced where it isn't. A mixture exploits everywhere with
probability α and plays Nash everywhere with probability 1 − α.

## 3. Data-Biased Response

**What DBR changes.** RNR trusts the model equally everywhere, including
information sets where it is nothing but a prior or a default, and the
counter-strategy overfits those. DBR instead restricts the opponent **per
information set**:

```
opponent at I = P_conf(I)·σ_fix(I) + (1 − P_conf(I))·σ_free(I),
P_conf(I) = P_max · f(n_I)
```

* `n_I` is the number of observations of `I`;
* `σ_fix` is the frequency model `n_I(a) / n_I`;
* `f` is a confidence function with `f(0) = 0`. Here
  `f(n) = min(1, n / threshold)`, the paper's "0–N linear" function;
* where nothing was observed, the opponent is completely free, so hero
  assumes the worst there rather than exploiting a default.

**How it is computed here.** As in the paper, hero runs CFR(+) against the
opponent's *free* part `σ_free`. The opponent's regret for action `a` at `I`
is `(1 − P_conf(I))·(v(a) − Σ_b σ_free(b) v(b))`.

**Approximations and caveats, stated explicitly:**
1. **Counts are inferred, not observed.** In Leduc the opponent's information
   set includes its hidden card, which hero sees only at showdowns. `n_I`
   are therefore the **posterior expected counts** from M6's latent-card
   reconstruction (fractional and model-dependent), and `σ_fix` is
   `n_I(a) / n_I` from those counts. This is the main departure from the
   paper's frequency counts.
2. **Not a standard game.** With a per-information-set restriction the
   opponent's realized behaviour is a product of mixtures along its path. The
   resulting game isn't bilinear in the opponent's sequence-form strategy, so
   CFR's convergence guarantee doesn't formally apply. The paper uses the same
   procedure; every output here is verified exactly.
3. **No exploitability guarantee.** Like RNR, `P_max` has no a-priori
   exploitability bound.
4. **Suits are tied** (canonical information sets share counts), as in M6.

`P_max = 0`, or no observations at all, gives plain CFR+. `P_max = 1` with
every count above the threshold gives exactly the RNR `p = 1` game. Both are
tested.

## 4. Exact evaluation and fair selection

Every candidate is evaluated in the **original** Leduc game:
* exploitability `= NashConv / 2` of hero's complete seat-free policy,
  computed by the M3 exact best-response evaluator;
* payoffs exactly, by tree walk.

No value from the modified game or the solver is trusted for the
comparisons.

For each opponent snapshot (M6's reconstruction after `n` hands) and
exploitability budget `B = expl(CFR) + δ`:
* **mix:** M6's rule, the largest α with exact exploitability ≤ B;
* **RNR:** solve on `p ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.3, …, 1}`, ascending,
  stopping after two consecutive points above the largest budget. Among the
  points with exact exploitability ≤ B, play the one with the highest exact
  value **against the model**;
* **DBR:** the same over `P_max ∈ {0.05, 0.1, 0.2, …, 1}`, with the count
  threshold tuned on development opponents (candidates 3, 10, 30);
* if no point is within budget, hero keeps CFR.

The true opponent policy is read only to evaluate payoffs and by the
evaluation-only oracles: an RNR and a mixture computed against the true
policy, at the same budgets.

## 5. Solver infrastructure

`robust/tree.py` compiles Leduc once into arrays: 9,457 nodes, 2,184
(information set, action) slots and 11 depth levels. `robust/solver.py` runs
CFR or CFR+ with NumPy over those arrays: about 0.55 ms per iteration,
versus about 18 ms for the M3 recursive solver. It is checked against
existing solvers:
* vanilla mode reproduces the M3 `GameCFRSolver` to 1e-15;
* CFR+ mode reproduces OpenSpiel's `CFRPlusSolver` to 4e-14;
* RNR reproduces generic CFR on the explicit modified game (`RNRGame`)
  to 8e-16.

The M1 and M3 CFR baselines are unchanged.
