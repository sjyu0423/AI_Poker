# Design: the smallest game interface that supports Leduc Hold'em

Status: **implemented in M3** (proposed in M2B). The implementation follows
this design with the deviations listed in §7. Rules and conventions:
`docs/games/leduc.md`.
The goal is to run M1's CFR and exact evaluator, and M2's neural pipeline, on
Leduc Hold'em by changing the *game boundary*, not by rewriting the
algorithms.

## 1. What has to change, and why

Everything above the engine is currently Kuhn-specific in a few places:

| Where | Kuhn assumption | Location |
|---|---|---|
| Chance | One deal at the root, iterated as `all_deals()` with weight `1/6` | `learning/cfr.py` (`iterate`), `evaluation/exact.py` (`expected_value`, `best_response`), `neural/dataset.py` |
| Rounds | One betting round; the seat to act is `len(history) % 2` | `games/kuhn.py`, documented in `InfoState.key` |
| Actions | A global `Action` enum {CHECK, BET, CALL, FOLD} | `games/kuhn.py`, `strategy.py` (JSON codes), `neural/encoding.py` (4 outputs) |
| Information | Private card only; no public cards | `InfoState` fields, `neural/encoding.py` |
| Key format | `"K|xb"`; strategy files tagged `"game": "kuhn"` | `InfoState.key`, `strategy.load_strategy` |
| Encoding | A fixed list of 4 decision histories | `neural/encoding.py` (`DECISION_HISTORIES`) |
| Simulation | `deal(rng)` then betting; seats alternate | `simulation/runner.py` |
| References | The α-family equilibrium, value −1/18 | `evaluation/reference.py` (stays Kuhn-only) |

The CFR update itself (regret matching, frozen `σ^t`, the reach-weighted
average), the best-response recursion and NashConv don't depend on Kuhn.
They only need explicit chance nodes and an information-set key.

## 2. Leduc Hold'em facts to design against

These were measured with OpenSpiel 2.0.2's `leduc_poker` on this Mac. They
will become test oracles in M3.

| Property | Value |
|---|---|
| Deck | 6 cards: J, Q, K in two suits |
| Antes | 1 each |
| Round 1 | private card each; fixed raise 2; at most 2 raises |
| Round 2 | one public card; fixed raise 4; at most 2 raises |
| Showdown | a pair with the public card wins, otherwise the higher card; equal ranks split |
| Distinct actions | 3 (OpenSpiel: Fold, Call/Check, Raise/Bet) |
| Max chance outcomes at a node | 6 |
| Max game length | 8 actions |
| Utility range | −13 … +13 chips |
| Information states (both players) | **936** |
| Full tree | 157 chance, 3,780 decision, 5,520 terminal nodes |
| NashConv of the uniform policy | **4.747222…** (a reference value for our evaluator) |

The tree is about 70× Kuhn's decision nodes. At M1's measured cost of roughly
2.6 µs per decision-node visit in pure Python, one vanilla CFR iteration
would take on the order of 10 ms. 10⁴–10⁵ iterations then take minutes.
That's slow but workable for a baseline, so no compiled tree is needed in the
first step.

## 3. Proposed interface

Two protocols, one dataclass and one constant. Every game, Kuhn included,
implements them. They live in a new `pokermind/games/base.py`.

```python
CHANCE: Final = -1  # pseudo-player id for chance nodes


@dataclass(frozen=True, slots=True)
class Observation:
    """What one player may know. Replaces Kuhn's InfoState as the generic type."""

    player: int
    private: tuple[int, ...]  # own private card(s), game-specific ids
    public: tuple[int, ...]  # public cards revealed so far
    actions: tuple[tuple[int, ...], ...]  # public betting, one tuple per round

    @property
    def key(self) -> str: ...  # game-defined, stable, opponent-free


class GameState(Protocol):
    @property
    def current_player(self) -> int: ...  # 0, 1, or CHANCE
    @property
    def is_terminal(self) -> bool: ...
    def chance_outcomes(
        self,
    ) -> tuple[tuple[int, float], ...]: ...  # (outcome, prob); chance nodes only
    def legal_actions(self) -> tuple[int, ...]: ...  # decision nodes only, fixed order
    def apply(self, action: int) -> GameState: ...  # chance outcome or action
    def returns(self) -> tuple[float, float]: ...  # terminal only, net chips
    def observation(self, player: int) -> Observation: ...


class Game(Protocol):
    name: str  # "kuhn", "leduc"; written into strategy files
    num_actions: int  # size of the global action space (Leduc: 3)
    action_names: tuple[str, ...]

    def initial_state(self) -> GameState: ...  # a chance node (the deal)
    def encode(self, obs: Observation) -> NDArray[np.float32]: ...  # for neural policies
```

Design choices, and the reasons for them:

* **Explicit chance nodes**, with outcomes and probabilities, instead of a
  `deal(rng)` helper. CFR and the evaluator multiply chance probabilities into
  the counterfactual reach; the simulator samples
  `rng.choice(outcomes, p=probs)`. One mechanism covers the Kuhn deal and
  Leduc's private and public cards. Dealing two private cards one at a time
  makes later outcome probabilities depend on earlier ones, which
  `chance_outcomes()` expresses naturally.
* **Integer actions per game**, with a fixed legal order. The global M0
  `Action` enum stays for Kuhn, where `KuhnGame` maps it to 0..3. Leduc gets
  its own 3-action space. Strategy tables keep `{action: prob}` mappings, keyed
  by `int`, with game-specific codes in JSON.
* **Observation splits private, public and per-round betting**, so it can
  stay opponent-free *by construction*, exactly like `InfoState` today. The
  M0 tests (a fixed field set, invariance when the opponent's card is swapped)
  carry over directly.
* **`key` is defined by the game.** Kuhn keeps `"K|xb"`, so M1's strategy files
  and digests stay valid. Leduc uses something like `"Q|K|rc/r"` (private |
  public | round-1/round-2 actions).
* **`returns()` instead of `payoffs()`**, as floats, because Leduc splits the
  pot on ties.
* **No player-count generality** beyond two seats. Both games are heads-up,
  and the evaluator's `NashConv = br0 + br1` identity relies on zero-sum
  two-player play.

What is deliberately *not* in the interface: abstraction, bet sizing,
vectorized tree arrays, observation tensors for the opponent, and
multi-player support. Each can be added later without changing it.

## 4. Migration plan (M3)

1. **Add `games/base.py` and a `KuhnGame` adapter** over the existing
   `KuhnState`. The root becomes a chance node with the 6 deals, in
   `all_deals()` order, each with probability 1/6. `KuhnState` itself doesn't
   change.
2. **Port CFR and the evaluator** to recurse through chance nodes using the
   interface. Acceptance test: on `KuhnGame` the new code reproduces M1
   *bit for bit*, including the 50k strategy digest `b8217a0c…` and every
   exploitability value. M1's tests keep running unchanged on the old path
   until the new path matches.
3. **Implement `LeducGame`** as an immutable state like `KuhnState`, with
   validation in `__post_init__`.
4. **Verify Leduc against OpenSpiel:**
   * 936 information states;
   * the node counts in §2;
   * uniform NashConv 4.747222;
   * a CFR average policy that matches OpenSpiel's simultaneous-update
     `_CFRSolver` to rounding error, as M1 did.
5. **Generalize the strategy files** to `"game": <name>` with game-specific
   action codes. Kuhn files keep loading.
6. **Neural pipeline:** `Game.encode` replaces `neural/encoding.py`'s
   fixed tables. The dataset builder enumerates information sets through the
   interface.

## 5. Migration risks

| Risk | Consequence | Mitigation |
|---|---|---|
| Floating-point summation order changes when the deal moves from a loop into a chance node | M1 digests differ although the maths is the same | Keep the same outcome order and the same accumulation order. Treat "bit-identical on Kuhn" as an acceptance test, not a hope. If exact identity proves impossible, record the new digest together with a tolerance-based equivalence test. |
| Leduc's two cards of each rank mean the second deal's probabilities depend on the first | Wrong chance weights; plausible-looking but wrong exploitability | Chance probabilities come only from `chance_outcomes()`. Check against OpenSpiel's uniform NashConv and its node counts. |
| CHECK/BET vs CALL/RAISE naming differences (OpenSpiel merges check into call and bet into raise) | Mapping errors in the cross-check | Use an explicit mapping table plus tests, as M1's OpenSpiel mapping did. |
| Information-set keys that accidentally depend on hidden data, such as the order cards were dealt | Hidden-information leakage | `Observation` has no opponent field. Port the M0 swap-invariance test to every game. |
| Tie handling (split pots) | Zero-sum checks with integer payoffs break | `returns()` as floats; zero-sum checked with a tolerance. |
| Pure-Python speed on 3,780 decision nodes | Slow CFR and test suite | Small iteration counts in unit tests. If needed later, cache the tree, which needs no interface change. |
| Kuhn-specific references (`reference.py`) used for Leduc by mistake | False "equilibrium" claims | Keep references keyed by game. Leduc has no closed-form equilibrium, so exploitability is the only correctness metric. |
| The neural "generalization" question becomes testable and tempting | Overclaiming from in-sample fits | Make held-out information-set splits part of the Leduc dataset design from the start. |

## 6. Recommended M3 scope

M3 is the **game interface + Leduc + verified CFR/evaluation on Leduc**. It
should stop *before* neural work on Leduc:

1. `Game`/`GameState`/`Observation` and the `KuhnGame` adapter, reproducing M1
   bit for bit.
2. A `LeducGame` engine, with rules tests (every terminal payoff class,
   splits, raise caps) and information-hiding tests.
3. CFR and the exact evaluator through the interface, cross-checked against
   OpenSpiel on Leduc.
4. A `train-cfr --game leduc` experiment with exploitability checkpoints.

Neural distillation on Leduc, with a held-out information-set split, is the
natural M4. It's the first experiment in this project where generalization
can actually be measured.

## 7. What M3 implemented, and where it deviates

| Proposal | Implemented | Why |
|---|---|---|
| One shared `Observation` dataclass with `private`, `public` and `actions` fields | `Observation` is a **protocol** (`player`, `key`). Kuhn returns its existing `InfoState`; Leduc returns `LeducObservation` | Keeps Kuhn's observation type and keys unchanged; each game's observation lists only its own observable fields |
| Keys derived from the observation only | Also exposed as `GameState.information_state_key(player)` | The CFR and evaluator hot paths need only the string |
| `Game.encode` and `num_actions` | Left out | No neural work in M3; `action_codes` covers strategy files |
| The Kuhn deal as chance nodes | **One** root chance node with the 6 ordered deals (1/6 each), in `all_deals()` order | Makes the generic CFR perform M1's exact floating-point operations: the full 50k digest `b8217a0c74657df799530f51315c1f8fa3d4c299a07f196af8474e65ffc0becc` is reproduced |
| Port CFR and the evaluator to the interface | Added **beside** M1's code (`learning/game_cfr.py`, `evaluation/game_exact.py`). M1's modules are unchanged and serve as the regression reference | The brief asks not to rewrite the original CFR without justification. Tests compare the two bit for bit on Kuhn. |
| Generic simulator | `simulation/game_runner.py`, sampling each chance node by inverting the CDF with one draw | M0's runner (`rng.permutation` deal) is kept for Kuhn, so M0 digests don't change |

Measured risk outcomes:

* **Summation order:** CFR on Kuhn is bit-identical. The generic evaluator's
  best responses and NashConv are bit-identical too. The on-policy value
  differs by at most about 6e-17, because the chance node computes
  `Σ (1/6)·v` where M1 computed `Σ v / 6`.
* **Conditional chance:** checked against OpenSpiel at every chance node, and
  the exact deal distribution is uniform over all 120 ordered triples.
* **Speed:** 18 ms per Leduc iteration after removing an immutability-guard
  cost found by profiling, about 4.8 µs per decision-node visit. Full
  traversal is practical, so no tree compilation was needed.
