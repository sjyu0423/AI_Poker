# Kuhn Poker in PokerMind

Kuhn Poker (H. W. Kuhn, 1950) is the smallest interesting poker game. It has
hidden information, bluffing and a known equilibrium, yet its game tree is
small enough to write down by hand. That makes it the right place to check the
engine, the agent interface and, later, learning algorithms, before moving to
larger games.

## Rules implemented

| Element         | Value                                                     |
|-----------------|-----------------------------------------------------------|
| Players         | 2 (seat 0 and seat 1)                                     |
| Deck            | 3 cards: J < Q < K (no suits)                             |
| Ante            | `ante` chips each (default 1)                             |
| Deal            | one private card per player; the third card is unseen     |
| Betting rounds  | 1                                                         |
| Bet size        | `bet_size` chips (default 1), at most one bet, no raises  |
| First to act    | seat 0                                                    |

Actions:

* **Not facing a bet:** `CHECK` (`x`) or `BET` (`b`).
* **Facing a bet:** `FOLD` (`f`) or `CALL` (`c`).

Kuhn's original paper uses just two actions, "pass" and "bet", and many
implementations encode the game that way. PokerMind names all four actions
instead, because that matches No-Limit Hold'em terminology and makes it clear
which actions are legal at each point. The game tree is the same either way.

### Game tree

```
seat 0:  x ──► seat 1:  x  → showdown           (xx)
         │              b ──► seat 0:  f → seat 1 wins  (xbf)
         │                             c → showdown      (xbc)
         b ──► seat 1:  f  → seat 0 wins         (bf)
                        c  → showdown            (bc)
```

There are 4 decision histories (`""`, `"x"`, `"b"`, `"xb"`), 5 terminal
histories and 6 deals, so 30 terminal states in total. Every one of them is
checked in `tests/test_kuhn_rules.py`.

### Payoffs

Payoffs are net chips. With the defaults (ante 1, bet 1), seat 0's payoff is:

| History | Seat 0 holds the higher card | Seat 0 holds the lower card |
|---------|-----------------------------|-----------------------------|
| `xx`    | +1                          | −1                          |
| `bf`    | +1                          | +1                          |
| `bc`    | +2                          | −2                          |
| `xbf`   | −1                          | −1                          |
| `xbc`   | +2                          | −2                          |

Seat 1 always receives exactly the negative of this, so the game is zero-sum.
In general, the loser loses what they put into the pot: `ante`, or
`ante + bet_size` if they bet or called.

## State representation

`pokermind.games.kuhn` defines three frozen (immutable) dataclasses.

* **`KuhnConfig(ante, bet_size)`**: the game parameters, validated as
  positive integers.
* **`KuhnState(cards, history, config)`**: the full, omniscient state.
  `cards[p]` is seat `p`'s private card and `history` is a tuple of `Action`s.
  `apply(action)` returns a new state, or raises `IllegalActionError` if the
  action isn't legal. Building a state checks that the cards are distinct and
  that the history is reachable, so an invalid state can't be represented.
  The rules are stored as an explicit table from history to legal actions,
  plus the set of terminal histories, rather than spread across conditionals.
* **`InfoState(player, private_card, history)`**: what one player may know:
  their seat, their own card and the public action history. `key` gives a
  compact string such as `"K|xb"`, which is useful for tabular algorithms.
  The key leaves out the seat on purpose: players alternate strictly, so the
  seat to act is always `len(history) % 2` and each key identifies exactly one
  information set (12 in total). Games where the seat can't be derived from
  the public history must include it in their keys.

### Information hiding

`InfoState` has no field for the opponent's card, so leaking it would need a
deliberate change to its definition, and a test asserts the exact set of
fields. A second test enumerates every reachable state and both seats. For
each one it replaces the opponent's card with every other possible card and
checks that the observing player's `InfoState`, its key and its `repr` stay
the same. The simulation runner only ever passes `InfoState` objects to
agents. A spy-agent test checks this at runtime.

## Simulation and reproducibility

`run_simulation` uses `numpy.random.SeedSequence(seed).spawn(3)` to create
independent random streams for the deck, agent 0 and agent 1. Agents must draw
all their randomness from the generator they are given. As a result:

* the same seed and configuration produce the same hands, actions and payoffs.
  The result's SHA-256 `digest`, computed over every hand record, makes this
  easy to compare;
* the cards dealt do not depend on how much randomness the agents use (this
  is tested).

Seats alternate each hand by default, because seat 0 acts first and the game
is not symmetric. Returns are reported per agent, with a standard error, and
per (agent, seat). If an agent returns an illegal action, it is counted and
replaced with the passive legal action (CHECK or FOLD). `--strict` raises
instead.

Reproducibility holds for a fixed NumPy version. NumPy only promises a stable
bit stream for `Generator` methods within compatible releases, so the NumPy
version is recorded in each run's metadata.
