# M8A: interactive application backend

A local API for a human to play heads-up Leduc Hold'em against PokerMind's
adaptive AI. The AI is the frozen research baseline
(`configs/pokermind-research-v1.json`, `docs/research/research_freeze.md`).
This layer only *connects* existing components:
* it declares no research hyperparameters;
* it adds no solving algorithm;
* it does not change the research core.

```bash
pip install -e ".[app]"
pokermind serve                      # http://127.0.0.1:8000/docs (OpenAPI UI)
pokermind serve --port 8001 --update-workers 2
```

## 1. Architecture

```
HTTP (FastAPI, app/api.py) ── validation, routing, error mapping only
        │
SessionManager (app/manager.py) ── session store, random ids, shared update backend
        │
GameSession (app/session.py) ── one human vs the AI
  ├─ hand engine: games/leduc.py (rules, legality, settlement)
  ├─ AI action:   sample from the active StrategyVersion at the AI's own info-set key
  ├─ observations: adaptive.reconstruction.observer_record → ChangeAwareObserver (M5/M6)
  ├─ views:       public state, history, metrics (hidden-information filtered)
  └─ UpdateScheduler (app/updates.py)
        │ UpdateJob (observer records only)
        ▼
  backend: worker process (default) | thread | inline (tests)
        │ compute_policy_update (app/policy_update.py) = the frozen M7B procedure
        ▼
  UpdateResult ──► certify_policy (M3 exact evaluator, serving process) ──► atomic install
```

| component | reuses |
|---|---|
| rules, legal actions, payouts | `games/leduc.py` |
| AI observations of the human | `opponents/context.public_record`, `adaptive/reconstruction.observer_record` |
| opponent model, change detection | `adaptive/adapter.ChangeAwareObserver`, `opponents/bayes` |
| priors | `adaptive.build_priors` over the development population (the same as `build_world`) |
| reconstruction, posterior | `adaptive/reconstruction.CardConditionalModel` |
| RNR candidates, selection | `robust/batch.BatchedRNR`, `robust/selection.Selector` |
| certification | `adaptive/response.exploitability` (M3), `evaluation/game_exact.validate_game_strategy` |

**Loading the context** (`app/context.py`) verifies:
* the configuration digest;
* the hero file's strategy digest (`3f1136d4…`);
* the hero's exact exploitability, recomputed and compared with the frozen
  value to 1e-9.

Any mismatch stops the server.

## 2. Gameplay rules

* **Seats** alternate: the human is in seat `hand % 2`, as the opponent was
  in M5–M7B. Seat 0 acts first in both rounds. Each hand has two betting
  rounds; the public card is dealt between them. A hand ends by a fold or a
  showdown.
* **The server owns** the deck, both private cards, the betting state,
  settlement and observations. The client sends only an action name and the
  `state_version` it acted on. Every action is checked against the engine's
  legal actions.
* **Deals and AI randomness** come from `SeedSequence((session seed, hand))`.
  * Seeds come from `secrets` and are never served, because a seed reveals
    every future deal.
  * Fixed seeds are accepted only by managers created with
    `allow_client_seed=True` (tests and simulations).
* **Determinism.** With the inline backend a session is a pure function of
  its seed and the human's actions (tested).

### Hidden information

* **The AI's card** is served only in a completed hand's `result.ai_card`,
  and only after a showdown. During a hand and after any fold it appears in
  no response: not the state, history, metrics or strategy status (tested
  over HTTP).
* **The AI acts from its own information-set key**, which never contains the
  human's card.
* **The AI's model receives `observer_record`**. That record holds the
  human's public actions, the AI's own card, the public card, and the
  human's card **only if shown down**. A folded human card is never
  observed, although the server knows it.

## 3. Adaptive strategy updates

* **Start:** the certified CFR baseline (version 1, exploitability 0.0030).
* **Recompute schedule:**
  * every `recompute_every` completed hands (default **50**). M7B found that
    re-solving every 20–100 hands is enough: re-using the previous strategy
    for that long costs about 0.01 chips/hand on held-out opponents.
  * immediately after the change detector resets the model;
  * on `POST …/strategy/recompute`.
* **The job** is the frozen M7B acting procedure: prior selection, EM
  reconstruction, batched RNR (13 values of p, 500 CFR+ iterations),
  16 tempered posterior samples, and the calibrated lower-bound rule within
  the budget. If no lower bound is positive, it falls back to CFR. A test
  checks that it picks the same strategy as `reliability evaluate` on the
  same observations and seed.
* **Off the request path.** By default the job runs in a spawned worker
  process that loads the context once. A solve takes about 7.5 s;
  gameplay requests take under 4 ms meanwhile.
* **Verified, atomic install.** The serving process re-certifies the
  returned table:
  * every information set has a valid distribution;
  * the exact exploitability is ≤ CFR + δ (0.1030 chips/hand);
  * it matches the worker's own M3 certificate to 1e-9.

  Only then does the session's `active` reference switch, under the session
  lock. Hands already dealt keep the strategy they were dealt with.
* **Failures and races:**

  | situation | behaviour |
  |---|---|
  | solver exception | recorded in `last_failure`; active strategy kept; retried at the next trigger |
  | certification fails (over budget, invalid table, certificate mismatch, wrong job) | rejected as above |
  | job exceeds `update_timeout` (600 s) | marked timed out; its late result is discarded |
  | request while a job runs | coalesced into one follow-up job built from the newest observations |
  | result from before a change reset | discarded; a new job is started |
  | result older than the active strategy, or not the current job | discarded (`discarded_stale`) |
  | session closed | running job cancelled or its result discarded; no follow-up |

## 4. API contract (version 1)

All bodies are JSON. Request bodies **reject unknown fields** (422). Errors
are `{"detail": "..."}` with these status codes:
* **400:** illegal action, or invalid settings;
* **404:** unknown or closed session;
* **409:** a stale `state_version`, a hand already in progress, no hand in
  progress, or not the human's turn;
* **422:** a schema violation.

| method, path | body | returns |
|---|---|---|
| `GET /health` | – | `{status, api_version}` |
| `GET /config` | – | research version and digest, hero digest, baseline exploitability, δ, exploitability limit, rule, RNR iterations |
| `POST /sessions` → 201 | `{recompute_every?: 1..10000}` | session state (below) |
| `GET /sessions` | – | `[{session_id, hands_played, created_at}]` |
| `GET /sessions/{id}` | – | session state |
| `DELETE /sessions/{id}` → 204 | – | – |
| `POST /sessions/{id}/hands` → 201 | – | session state with a new hand (the AI may already have acted) |
| `POST /sessions/{id}/actions` | `{action: "fold"\|"call"\|"raise", state_version: int}` | session state after the human's action and the AI's replies |
| `GET /sessions/{id}/history?offset=0&limit=50` | – | `{total, offset, hands: [hand]}` (completed hands, limit ≤ 500) |
| `GET /sessions/{id}/metrics` | – | research metrics (below) |
| `GET /sessions/{id}/strategy` | – | `{active, versions, updates}` |
| `POST /sessions/{id}/strategy/recompute` → 202 | – | update status |

**Session state:**

```json
{
  "session_id": "32 hex chars",
  "state_version": 7,
  "hands_played": 3,
  "hand": {
    "hand_number": 4, "status": "awaiting_human" | "complete",
    "human_seat": 0, "first_to_act": "human" | "ai",
    "human_card": "K1", "public_card": null | "Q0", "round": 1 | 2,
    "actions": [{"actor": "human" | "ai", "round": 1, "action": "raise"}],
    "contributions": {"human": 3, "ai": 1}, "pot": 4,
    "to_act": "human" | null,
    "legal_actions": [{"action": "fold" | "call" | "raise", "label": "fold" | "check" | "call" | "bet" | "raise"}],
    "strategy_version": 2,
    "result": null | {"ended_by": "fold" | "showdown", "winner": "human" | "ai" | null,
                      "human_net": -3.0, "ai_net": 3.0, "pot": 4, "ai_card": null | "J0"}
  },
  "stats": {"hands_played", "human_net_chips", "ai_net_chips", "human_mean_per_hand",
            "human_mean_standard_error", "human_wins", "ai_wins", "splits", "showdowns",
            "human_folds", "ai_folds"},
  "strategy": {"active_version", "source", "certified_exploitability",
               "exploitability_limit", "recomputation_running"}
}
```

**Further notes:**
* Cards are rank (J, Q, K) plus suit digit. `winner: null` is a split pot.
* `state_version` increases with every state change. The client must echo
  the version it acted on; a stale or duplicated submission gets a 409 and
  changes nothing.

**Metrics** keep exact facts and model estimates in separate sections:

| section | contents | nature |
|---|---|---|
| `verified` | active strategy (version, source, RNR p, certified exploitability, digest, hands used), exploitability limit, baseline exploitability, δ, observation counts (hands observed / in model, human decisions and showdowns in the model, change resets), results, research and hero digests | exact: counted or computed by the exact evaluator |
| `estimates` | the human's action tendencies in six betting situations (posterior mean, 90 % credible interval, observed count); the active strategy's model estimates (prior chosen, lower-bound score, posterior mean and sd of gain over CFR); the change-detector statistic and threshold | probabilistic; carries a disclaimer that calibration was measured only on simulated development opponents, **not for human players** |
| `strategy_updates` | running job, pending follow-up, interval, next scheduled hand count, last success, last failure, counters | operational |

## 5. Limits

* **Sessions live in memory** (at most 32) and are lost on restart. There is
  no authentication, and the server binds to 127.0.0.1 by default. It is
  meant for one local user.
* **A timed-out job keeps its worker process busy** until it finishes. Later
  jobs queue behind it, and its result is discarded.
* **Posterior intervals are not calibrated for human players.** The
  exploitability limit is the only guarantee that holds against any player.
