# Leduc Hold'em in PokerMind

Leduc Hold'em (Southey et al., 2005) is the next step up from Kuhn Poker. It
has two betting rounds, a public card and raises. It is still small enough to
solve and evaluate exactly. PokerMind's implementation
(`src/pokermind/games/leduc.py`) matches OpenSpiel's `leduc_poker` with
default parameters. `tests/test_openspiel_leduc.py` walks both game trees in
lockstep and checks every node.

## Rules and conventions

| Element | Value |
|---|---|
| Players | 2 (seat 0 and seat 1) |
| Deck | 6 cards, ids 0–5: `rank = id // 2` (0 = J, 1 = Q, 2 = K), `suit = id % 2`. Names: `J0 J1 Q0 Q1 K0 K1` |
| Initial contributions | each player antes **1** before the deal |
| Deal | chance: player 0's private card (1/6 each), then player 1's (1/5 each of the remaining 5) |
| Round 1 | betting; raise size **2** |
| Public card | chance after round 1: one of the 4 remaining cards, 1/4 each |
| Round 2 | betting; raise size **4** |
| Raise cap | at most **2 raises per round** (a bet counts as the first raise) |
| Who acts first | **player 0 in both rounds**; players alternate within a round |
| Showdown | a player whose card matches the public card's rank wins; otherwise the higher rank wins; equal ranks **split** the pot |
| Utility | net chips. The winner gains the loser's contribution; a split gives 0 to both. Range: −13 to +13. |

### Actions and aliases

| Id | Code | Name | Meaning |
|---:|:---:|---|---|
| 0 | `f` | fold | give up the pot. **Legal only when facing a bet.** |
| 1 | `c` | call | match the opponent's contribution. A **check** when nothing is owed. |
| 2 | `r` | raise | match, then add the round's raise size. A **bet** when nothing is owed. Legal while the round has had fewer than 2 raises. |

These are OpenSpiel's ids and names, so policies and cross-checks line up.

### Turn and round rules

* Player 0 opens each round. After every action the other player acts.
* A round ends at the first **call that isn't the round's first action**:
  check–check (`cc`), or a call of a raise (`rc`, `crc`, `rrc`, `crrc`).
  These five sequences continue to the public card (round 1) or to
  showdown (round 2).
* A fold ends the hand immediately: `rf`, `crf`, `rrf`, `crrf`.
* After two raises in a round the next player can only fold or call.

Example: in `K1` vs `J0` with public `J1`, the sequence `r r c` / `r c` puts
in 1 + 4 (round 1) + 4 (round 2) = **9** chips each. Player 1's J pairs the
public card, so the returns are (−9, +9).

## Information sets

The acting player's key contains only what they observe:

```
round 1:  <own card>|<round-1 actions>                    e.g.  "K1|cr"
round 2:  <own card>|<public card>|<round-1>|<round-2>    e.g.  "K1|J0|rrc|c"
```

* The opponent's card never appears. Tests replace it with every other
  unseen card at all 3,780 decision nodes and check that the key and the
  observation don't change.
* The key records the full public action history, which includes the
  player's own actions. That gives **perfect recall**, which a test checks.
* Suits are part of the key (`K0` vs `K1`), as in OpenSpiel without suit
  isomorphism. This keeps the count at **936 information sets**: 36 in round 1
  and 900 in round 2, 468 per player.
* As in Kuhn, the key omits the seat: the public history determines who
  acts. `information_sets()` checks that every key always has the same acting
  player.

## Size

| | Kuhn | Leduc |
|---|---:|---:|
| Chance nodes | 1 (adapter) | 157 |
| Decision nodes | 24 | 3,780 |
| Terminal nodes | 30 | 5,520 |
| Information sets | 12 | 936 |
| Histories per information set | 2 | 5 (round 1), 4 (round 2) |
| Uniform-policy NashConv | 11/12 | 4.747222… |
