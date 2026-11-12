# Counterfactual Regret Minimization (CFR)

This note explains the mathematics behind PokerMind's M1 baseline: tabular,
vanilla CFR on Kuhn Poker, and the exact evaluator used to check it. The
notation follows Zinkevich et al. (2007), "Regret Minimization in Games with
Incomplete Information", and Lanctot (2013), "Monte Carlo Sampling and Regret
Minimization for Equilibrium Computation and Decision-Making in Large Extensive
Form Games". The code lives in `src/pokermind/learning/cfr.py` (training) and
`src/pokermind/evaluation/exact.py` (evaluation). The two share no code apart
from the M0 game engine.

## 1. Extensive-form games with imperfect information

An extensive-form game is a tree:

* **Histories** `h` are nodes: the sequence of everything that has happened.
  In Kuhn Poker a history is the deal plus the betting so far, for example
  `(K, J) · check · bet`.
* **Players** act at decision nodes. A special **chance** player acts at
  chance nodes with fixed, known probabilities `σ_c`. In Kuhn there is one
  chance node, the deal: each of the 6 ordered deals has probability 1/6.
* **Terminal histories** `z` carry a utility `u_i(z)` for each player. M0's
  convention is net chips won (see `docs/kuhn_poker.md`), so `u_0(z) = -u_1(z)`
  and the game is zero-sum.

*Imperfect information* means a player does not observe the whole history:
they do not see the opponent's card.

## 2. Information sets

An **information set** `I` is a set of histories that the acting player cannot
tell apart. Everything a player decides must depend only on `I`, never on the
particular `h ∈ I`.

In Kuhn Poker a player knows their own card and the public betting, so an
information set is "my card + betting history". M0's `InfoState.key` encodes
exactly this, for example `"Q|xb"`: "I hold Q, the betting so far is check,
bet". The two histories in this set are `(Q, J)·x·b` and `(Q, K)·x·b`, which
differ only in the opponent's card. Kuhn Poker has 12 information sets:

| Player 0 (acts at `""`, `"xb"`) | Player 1 (acts at `"x"`, `"b"`) |
|---------------------------------|---------------------------------|
| `J|`, `Q|`, `K|`                | `J|x`, `Q|x`, `K|x`             |
| `J|xb`, `Q|xb`, `K|xb`          | `J|b`, `Q|b`, `K|b`             |

The seat is not part of the key because the players alternate strictly, so
the history length fixes who is acting. Because keys come from `InfoState`,
which has no field for the opponent's card, CFR's tables can't be indexed by
hidden information. Tests check this.

The game has **perfect recall**: players never forget their own earlier
actions or card. CFR's guarantees depend on this.

## 3. Strategies and Nash equilibrium

A **behavioural strategy** `σ_i` gives, for every information set `I` of
player `i`, a probability distribution `σ_i(I, ·)` over the legal actions
`A(I)`. A **strategy profile** `σ = (σ_0, σ_1)` has one for each player. In
the code a single table `key → {action: probability}` covers all 12
information sets, so it holds both players' strategies.

`u_i(σ)` is player `i`'s expected utility when everyone follows `σ`. A profile
is a **Nash equilibrium** if no player can gain by deviating alone:

```
u_i(σ) ≥ u_i(σ'_i, σ_{-i})   for every player i and every strategy σ'_i.
```

In a two-player zero-sum game every equilibrium gives player 0 the same
expected payoff, the **game value**. Under M0's payoffs (ante 1, bet 1), Kuhn
Poker's value is **−1/18 ≈ −0.0556 chips per hand for player 0**: acting first
is a disadvantage.

### The known Kuhn equilibria

Kuhn (1950) found every equilibrium. They form a one-parameter family with
`α ∈ [0, 1/3]`. Written in M0's action names:

| Info set | Meaning                       | Equilibrium                |
|----------|-------------------------------|----------------------------|
| `J|`     | P0 opens with J               | bet `α`, check `1 − α`     |
| `Q|`     | P0 opens with Q               | always check               |
| `K|`     | P0 opens with K               | bet `3α`, check `1 − 3α`   |
| `J|xb`   | P0 checked, faces bet, J      | always fold                |
| `Q|xb`   | P0 checked, faces bet, Q      | call `α + 1/3`             |
| `K|xb`   | P0 checked, faces bet, K      | always call                |
| `J|b`    | P1 faces bet with J           | always fold                |
| `Q|b`    | P1 faces bet with Q           | call `1/3`                 |
| `K|b`    | P1 faces bet with K           | always call                |
| `J|x`    | P1 after a check, J           | bet `1/3` (bluff)          |
| `Q|x`    | P1 after a check, Q           | always check               |
| `K|x`    | P1 after a check, K           | always bet                 |

Player 1's equilibrium strategy is unique. Player 0's is unique only up to
`α`. The tests check every member of this family against the evaluator
independently: exploitability 0 and value −1/18.

## 4. Why poker needs mixed strategies

A deterministic (pure) strategy can be exploited because the opponent can
predict it. Suppose player 1 never bluffs with J after a check. Then a player 0
holding Q who checks and then faces a bet knows player 1 has K, and folds
every time. If player 1 always bluffs with J, player 0 calls with Q every time.
Either way player 1's bet carries information that can be used against them.

In equilibrium player 1 bluffs with J exactly often enough (1/3) that player
0's Q is **indifferent** between calling and folding. Likewise player 1 calls
a bet with Q exactly 1/3 of the time, which makes player 0's J indifferent
between bluffing and checking. Randomizing is what makes a hand's betting
unreadable. No pure strategy profile is an equilibrium of Kuhn Poker.

## 5. Reach probabilities

For a history `h`, the **reach probability** `π^σ(h)` is the product of the
probabilities of every action on the path to `h`. It splits by who chose each
action:

```
π^σ(h) = π_0^σ(h) · π_1^σ(h) · π_c(h)
```

where `π_i(h)` multiplies only player `i`'s action probabilities, and `π_c(h)`
multiplies the chance probabilities (1/6 for the deal).

The **counterfactual reach** of `h` for player `i` is everything except
player `i`'s own contribution:

```
π_{-i}^σ(h) = π_c(h) · Π_{j ≠ i} π_j^σ(h)
```

It is the probability of reaching `h` if player `i` had played to reach it
for sure. That is why it is called *counterfactual*. With perfect recall
`π_i(h)` is the same for every `h ∈ I`, so we write `π_i^σ(I)`.

## 6. Counterfactual value and regret

Let `u_i(σ, h)` be player `i`'s expected utility from history `h` onward, and
`σ|_{I→a}` the profile that is identical to `σ` except that it always plays
`a` at `I`. The **counterfactual value** of `I` is

```
v_i(σ, I)    = Σ_{h ∈ I} π_{-i}^σ(h) · u_i(σ, h)
v_i(σ, I, a) = Σ_{h ∈ I} π_{-i}^σ(h) · u_i(σ, h·a)
```

The **instantaneous counterfactual regret** for not having played `a` at
iteration `t` is

```
r^t(I, a) = v_i(σ^t, I, a) − v_i(σ^t, I)
```

and the **cumulative regret** is `R^T(I, a) = Σ_{t=1..T} r^t(I, a)`.

Zinkevich et al. prove that the overall regret is at most the sum of the
positive parts of these local counterfactual regrets. So making each
information set's regret grow sublinearly makes the whole strategy's regret
grow sublinearly.

## 7. Regret matching

At each information set, **regret matching** turns cumulative regrets into the
next strategy:

```
R^+(I, a) = max(R^T(I, a), 0)

σ^{T+1}(I, a) = R^+(I, a) / Σ_b R^+(I, b)     if Σ_b R^+(I, b) > 0
              = 1 / |A(I)|                     otherwise (uniform fallback)
```

Actions we regret not having played more get more probability. If no action
has positive regret (all regrets zero or negative), the strategy falls back to
uniform. That fallback is essential: without it the strategy would be 0/0.
`regret_matching` in the code implements this, and the tests cover the
all-zero, all-negative and mixed cases.

## 8. The CFR iteration

PokerMind implements **vanilla CFR with simultaneous updates**. This is
Zinkevich et al.'s original form. OpenSpiel's public `CFRSolver` uses
*alternating* updates instead (one player per pass), which usually converges
faster. The optional cross-check in `tests/test_openspiel_crosscheck.py`
therefore compares against OpenSpiel's base `_CFRSolver` configured for
simultaneous updates, and agrees with it to rounding error.

1. Compute the current strategy `σ^t` for **every** information set from the
   cumulative regrets, then freeze it for the whole iteration. Each Kuhn
   information set is visited once per possible opponent card, so building
   the strategy lazily while regrets change within the iteration would be a
   subtle bug.
2. Traverse the full tree: chance first, every deal with probability 1/6,
   carrying the reach vector `(π_0, π_1, π_c)`. At a decision node for player
   `i` in information set `I`:
   * recurse into each legal action `a` with `π_i` multiplied by
     `σ^t(I, a)`, giving `u(h·a)`;
   * `u(h) = Σ_a σ^t(I, a) · u(h·a)`;
   * `R(I, a) += π_{-i}(h) · (u_i(h·a) − u_i(h))` (cumulative regret);
   * `S(I, a) += π_i(h) · σ^t(I, a)` (average-strategy accumulator).
3. After the traversal, `σ^{t+1}` comes from regret matching on the updated
   `R`.

Here `u` is the value *given that `h` was reached*. Reach is applied only once,
as the regret weight `π_{-i}(h)`. Both players' regrets are updated from the
same frozen `σ^t`.

### Current strategy versus average strategy

* The **current strategy** `σ^t` is what regret matching outputs at iteration
  `t`. It does **not** converge in general: it often cycles around the
  equilibrium as the players keep reacting to each other.
* The **average strategy** is

  ```
  σ̄^T(I, a) = Σ_t π_i^{σ^t}(I) · σ^t(I, a)  /  Σ_t π_i^{σ^t}(I)
  ```

  This average is **weighted by the player's own reach**. Iterations in which
  the player rarely reached `I` count for less. This is what accumulator `S`
  stores. In a two-player zero-sum game the average strategy converges to a
  Nash equilibrium. That is the strategy PokerMind exports, evaluates and
  plays. An information set with `Σ_b S(I, b) = 0` was never reached, and its
  average falls back to uniform.

*Implementation note:* the code adds `π_i(h)·σ^t(I, a)` once for each
history `h ∈ I` visited, as OpenSpiel does. Every Kuhn information set holds
exactly 2 histories (one per possible opponent card), and `π_i(h)` is the same
across them, so this multiplies `S(I, ·)` by a constant that disappears when it
is normalized.

### Convergence

If `Δ` is the range of utilities, `|ℐ_i|` the number of information sets and
`|A|` the maximum number of actions, average regret after `T` iterations is
`O(Δ · |ℐ_i| · √|A| / √T)`. The exploitability of the average strategy
therefore goes to zero at worst like `1/√T`. That is a worst-case bound. On
Kuhn Poker vanilla CFR usually does better in practice, and the M1 experiment
reports the measured curve rather than assuming a rate.

## 9. Best responses and exploitability

A **best response** `BR_i(σ_{-i})` is a strategy for player `i` that
maximizes `u_i(·, σ_{-i})` against a fixed opponent. It must still respect
information sets: player `i` picks one action per *information set*, not per
history, because they don't see the opponent's card. The evaluator computes it
exactly.

1. For each of player `i`'s information sets, collect its histories `h` and
   their weights `π_c(h) · π_{-i}(h)`, the opponent's probability of reaching
   them.
2. Process player `i`'s information sets from the deepest to the shallowest.
   At each one, choose the action that maximizes
   `Σ_{h ∈ I} weight(h) · u_i(h·a)`, where `u_i` below `I` uses the best
   response choices already fixed deeper in the tree. Because of perfect
   recall, deeper choices don't depend on how the player reached them.
3. The best response value is the expected utility of the resulting pure
   strategy against `σ_{-i}`.

Ties are broken towards the first legal action (passive first), so results are
deterministic. A pure best response always exists.

**NashConv** adds up how much each player could gain by deviating:

```
NashConv(σ) = Σ_i [ u_i(BR_i(σ_{-i}), σ_{-i}) − u_i(σ) ]
```

PokerMind defines **exploitability** as the average over players, the same
convention as OpenSpiel's `exploitability`:

```
exploitability(σ) = NashConv(σ) / 2
```

Both are in **chips per hand** (ante = bet = 1). Because Kuhn Poker is
zero-sum, `u_0(σ) + u_1(σ) = 0`, so `NashConv = br_0 + br_1`, where `br_i` is
the best response value. Both are ≥ 0, and they are 0 exactly when `σ` is a
Nash equilibrium. The game value gives a useful check: for any `σ`,
`br_0 ≥ −1/18` and `br_1 ≥ +1/18`.

## 10. Worked example: the first CFR iteration at `K|b`

Start with all regrets at zero, so `σ^1` is uniform everywhere. Look at player
1's information set `K|b`: player 1 holds K and faces a bet.

**Histories.** Player 1 holds K in two of the six deals: `(J, K)` and
`(Q, K)`. Each deal has `π_c = 1/6`, and player 0 bets with probability 1/2
under the uniform strategy. So each history has counterfactual reach

```
π_{-1}(h) = π_c · π_0 = 1/6 · 1/2 = 1/12.
```

**Action values for player 1** (net chips). K beats J and Q at showdown:

* `call` → showdown for 2 chips each: `u_1 = +2` in both histories.
* `fold` → player 1 loses the ante: `u_1 = −1` in both histories.
* `u_1(h) = ½·(+2) + ½·(−1) = +½` under the uniform strategy.

**Regrets.** Summing over the two histories:

```
r(K|b, call) = 2 · 1/12 · (2 − ½)   =  +1/4
r(K|b, fold) = 2 · 1/12 · (−1 − ½)  =  −1/4
```

**Next strategy.** Regret matching: only `call` has positive regret, so
`σ^2(K|b) = (fold 0, call 1)`. One iteration already shows that a K facing a
bet should always call.

**Average strategy.** Player 1's own reach at `K|b` is 1, because they made no
earlier choice. So `S(K|b) += 1 · (½, ½)` for each of the two histories. After
iteration 1 the average strategy is still uniform, and it moves toward always
calling as more iterations are added. `tests/test_cfr.py` checks these exact
numbers.

## 11. What M1 does not do

* No sampling (MCCFR), CFR+, or linear averaging: these are follow-ups.
* No function approximation or neural networks (M2+).
* No opponent modelling: an equilibrium strategy doesn't exploit anyone's
  mistakes. It only guarantees it can't be exploited. Adapting to a specific
  opponent is a later milestone.
