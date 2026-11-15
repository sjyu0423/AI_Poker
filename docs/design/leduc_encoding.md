# Design: neural encoding of Leduc information states (M4)

Code: `src/pokermind/neural/leduc_encoding.py` (features),
`src/pokermind/neural/leduc_dataset.py` (examples and splits).

## 1. What the encoder sees

`encode_observation` accepts only a `LeducObservation`: the acting player,
their own card, the public card (or none) and the public betting of each
round. That type has **no field for the opponent's card**, so hidden
information can't reach a model through the encoder. A test replaces the
opponent's card with every other unseen card at all 3,780 decision nodes and
checks that the feature vector doesn't change. Passing a full `LeducState`
raises `TypeError`.

## 2. Features (30, float32)

| Index | Feature | Values |
|---:|---|---|
| 0–2 | own rank J, Q, K | one-hot |
| 3–5 | public rank J, Q, K | one-hot; all 0 in round 1 |
| 6 | public card not yet revealed | 1 in round 1 |
| 7 | pair (own rank = public rank) | 0/1 |
| 8 | round 2 | 0/1 |
| 9–16 | round-1 betting: position 0–3 × {call, raise} | 0/1 |
| 17–24 | round-2 betting: position 0–3 × {call, raise} | 0/1 |
| 25 | acting seat is 1 | 0/1 |
| 26 | facing a bet (fold is legal) | 0/1 |
| 27 | raises made this round / 2 | 0, 0.5, 1 |
| 28 | own contribution / 13 | |
| 29 | opponent's contribution / 13 | |

The legal-action mask over (fold, call, raise) is separate and applied to
the logits before the softmax, so illegal actions get exactly zero
probability.

Design decisions:

* **The betting history is positional**, one slot per (round, position,
  action), rather than one one-hot per history string. A model can then
  share what it learns between histories with common prefixes or positions,
  which is what the held-out-history split tests. Fold never appears in a
  non-terminal history, so the slots need only call and raise.
* **Derived features are explicit.** Pair, seat, facing-bet, raise count and
  contributions are functions of the other features. They are included so a
  linear model or a 16-unit MLP doesn't spend capacity rediscovering the
  rules (pair vs no pair is an interaction a linear model can't form).
* **Contributions are replayed from the public history only.** Tests compare
  them, the seat, facing-bet and raise availability with the engine at every
  decision node.

## 3. Suits are dropped: 936 information sets become 288 states

Leduc's rules depend only on ranks, so information sets that differ only in
suits (for example `K0|J1|rc|` and `K1|J0|rc|`) are strategically identical.
The encoding drops suits. That maps the 936 information sets onto **288
canonical states**: 18 in round 1 and 270 in round 2. Each canonical state
stands for 2 or 4 information sets. The encoding is injective on canonical
states, which a test checks: different canonical states never share a feature
vector.

This is lossless only if the teacher is suit-symmetric. Vanilla CFR on Leduc
is: the M3 and M4 teachers give **bit-identical** probabilities on suit twins
(deviation exactly 0). `build_leduc_dataset` checks this and refuses a
teacher whose twins differ by more than 1e-12.

Consequences:

* There is **one training example per canonical state**: 288 examples,
  weighted equally. The 2 or 4 suit twins are not duplicated in the loss.
* **Splits are made over canonical states**, so two information sets with
  the same feature vector can never be on different sides of a split. Tests
  check that no test feature vector, and no test information-set key, occurs
  in the training or validation data.
* A predicted policy is expanded back to all 936 keys
  (`LeducDataset.expand`) before exact evaluation.

## 4. Splits

| Split | Test states | What is new at test time |
|---|---|---|
| `full` | none (all 288 used for training) | nothing; measures representation only |
| `random` | 58 (20 %, seeded) | the exact (cards, history) combination; every test card combination and betting history appears in training with other cards |
| `public` | 60: round-2 states with (own, public) = (Q, Q) or (J, K) | the card combination. Q-pairs and J-under-K never appear in training, though Q, J, K, pairs and every betting history do |
| `history` | 57: all states whose round-1 betting starts `rr` (`rr` itself, and round 2 after `rrc`) | the betting line. No training state has round-1 history `rr` or `rrc`, though each position slot appears in some other history |

In every split, 10 % of the non-test states (seeded) form a validation set.
It is the only data used to select checkpoints. No split activates a binary
feature that never occurs in training. The `public` and `history` splits are
out of distribution only in the *combination* of features.
`split_coverage` reports all of this per split.

## 5. Limitations

* **Uniform weighting.** Every canonical state counts equally in the loss,
  although some are reached far more often in play than others. A
  reach-weighted loss would match the teacher better where it matters most
  for payoff, but would weight the test metrics differently.
* **Only the canonical (suit-free) structure is handled.** A game where suits
  matter (flushes) would need suit features.
* **Positional slots assume at most 4 actions per round**, which holds for
  Leduc's raise cap of 2. Other caps need a different layout.
* **Scaled contributions assume the default antes and raise sizes.** A
  different `LeducConfig` changes the scale, so a model trained on one
  configuration doesn't transfer to another.
* **The teacher's mixed strategies aren't unique.** Leduc has many
  equilibria, so a held-out state's "correct" distribution is the one this
  CFR run produced, not a unique answer. Imitation error on held-out states
  therefore overstates how bad a prediction is for play, and exact
  exploitability is the deciding metric.
