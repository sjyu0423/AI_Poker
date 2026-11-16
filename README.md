# PokerMind

PokerMind is a local-first poker AI research and training project. The long-term
goal is an agent that learns an opponent's tendencies from the actions it
observes and adjusts its strategy to them, while staying reasonably hard to
exploit. Heads-up Texas Hold'em is the eventual target, but every algorithm is
validated in small games first.

**Status: M8B, playable web interface** (`web/`, Next.js): play heads-up
Leduc against the adaptive AI in the browser; see `web/README.md` (M8A: a
FastAPI server, `pokermind serve`, with background, certified strategy
updates, `docs/design/app_backend.md`; M7B: research
freeze of the adaptive Leduc baseline: a frozen,
versioned configuration (`configs/pokermind-research-v1.json`) with a
batched RNR solver, calibrated uncertainty-aware selection and held-out
robustness results, `docs/research/research_freeze.md`; M7A: robust
exploitation with Restricted Nash Response and Data-Biased Response; M6: opponent-adaptive play with an exploitability
budget; M5: Bayesian opponent modelling; M4: neural generalization; M3: Leduc and
generalized CFR/evaluation; M2B: training stability; M2A: neural
distillation; M1: vanilla CFR with exact evaluation; M0: the Kuhn Poker
engine).

## Layout

```
src/pokermind/
  games/kuhn.py          immutable config/state/info-state, rules, payoffs
  games/base.py          M3 game interface: Game, GameState, Observation, chance nodes
  games/kuhn_game.py     Kuhn behind the interface (adapter over games/kuhn.py)
  games/leduc.py         Leduc Hold'em, matching OpenSpiel's leduc_poker
  agents/                Agent protocol, random/rule baselines, StrategyAgent (CFR)
  strategy.py            tabular strategy type and JSON file format
  learning/cfr.py        vanilla CFR for Kuhn (M1, kept as regression reference)
  learning/game_cfr.py   vanilla CFR for any Game (M3)
  evaluation/exact.py    exact EV, best response, NashConv, exploitability (Kuhn, M1)
  evaluation/game_exact.py  the same for any Game (M3)
  evaluation/reference.py known Kuhn equilibrium family, game value -1/18
  neural/                M2A: encoding, teacher dataset, masked MLP, training, agent
  neural/leduc_*.py      M4: Leduc encoding, canonical dataset and splits, neural agent
  neural/game_*.py       M4: size-agnostic policy network; training with validation
  opponents/             M5: public contexts/records, simulated opponent population,
                         exact public-context marginals, Dirichlet opponent model, CUSUM
  adaptive/              M6: card-conditional policy reconstruction (latent-card EM, Gibbs),
                         hero best responses, realization mixtures, exploitability budgets
  robust/                M7A: compiled game tree, vectorized restricted CFR(+), Restricted
                         Nash Response and Data-Biased Response, explicit RNR game;
                         M7B: batched RNR solver and evaluator (batch.py), budgeted
                         uncertainty-aware selection (selection.py)
  research.py            M7B: versioned research configuration and digest
  app/                   M8A: verified context, Leduc sessions, background certified
                         strategy updates, session store, FastAPI adapter (api.py)
  experiments/           CFR and neural-distillation runners, metrics and plots
  simulation/runner.py   seeded batch runner, metrics, result digest
  simulation/metadata.py reproducibility metadata (versions, platform, git)
  cli.py                 `pokermind` command
tests/                   rules, payoffs, leakage, agents, simulation, CLI,
                         CFR, evaluation oracle, OpenSpiel cross-check (optional)
docs/kuhn_poker.md       rules and state representation
docs/concepts/cfr.md     CFR, best response and exploitability, worked example
docs/concepts/neural_policy.md  layers, backprop, Adam, softmax, KL, distillation
docs/experiments/        frozen M2A results, M2B stability/capacity/benchmark results
docs/design/leduc_interface.md  game interface design and what M3 implemented
docs/games/leduc.md      Leduc rules, action aliases, turn order, information sets
docs/design/leduc_encoding.md  M4 features, suit canonicalization, splits
docs/experiments/m4_leduc_neural.md  M4 results
docs/design/opponent_modeling.md  M5 observability, contexts, model, evaluation, leakage
docs/experiments/m5_opponent_modeling.md  M5 results
docs/design/adaptive_strategy.md  M6 reconstruction, mixtures, budget, protocol
docs/experiments/m6_adaptive_strategy.md  M6 results
docs/design/robust_response.md  M7A RNR/DBR, guarantees, approximations, selection
docs/experiments/m7a_robust_response.md  M7A results
docs/experiments/m7b_adaptive_reliability.md  M7B convergence, calibration, speed, robustness
docs/research/research_freeze.md  M7B frozen architecture, evidence, reproduction
configs/pokermind-research-v1.json  frozen research configuration (digest-checked)
docs/design/app_backend.md  M8A architecture, gameplay rules, update safety, API contract
research/                experiment notebooks/scripts (kept apart from the package)
web/                     M8B: Next.js web interface (client of the M8A API), see web/README.md
```

Dependency direction: `games` ← `strategy` ← {`agents`, `learning`,
`evaluation`} ← `neural` ← `simulation` ← `experiments` ← `cli`. Only `neural`
and its experiment import PyTorch, lazily from the CLI. Games know nothing about
agents or learning. Agents only see `InfoState`. The evaluator never imports
the learning code, so it is an independent check on CFR.

## Setup (Apple Silicon, Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
pokermind agents
pokermind simulate --agents random rule --hands 100000 --seed 42
pokermind simulate --agents rule always-bet --hands 10000 --seed 7 --output runs/rule_vs_bet.json
python -m pokermind simulate --json --hands 1000

# M1: train CFR, evaluate exactly at checkpoints, export strategies and plots
pokermind train-cfr --output-dir runs/m1-cfr --seed 0
pokermind simulate --agents cfr rule --strategy runs/m1-cfr/strategies/strategy_050000.json

# M3: CFR on Leduc Hold'em (or Kuhn) through the generic game interface
pokermind train-cfr --game leduc --output-dir runs/m3-leduc-cfr
pokermind train-cfr --game kuhn --engine generic   # reproduces M1 bit for bit

# M4: a stronger Leduc teacher, then neural distillation and generalization splits
pokermind train-cfr --game leduc --checkpoints 1000 10000 20000 50000 100000 \
    --output-dir runs/m4-leduc-teacher
pokermind leduc-neural --teacher runs/m4-leduc-teacher/strategies/strategy_100000.json \
    --comparison-teacher runs/m3-leduc-cfr/strategies/strategy_010000.json

# M5: learn simulated opponents' tendencies from public actions (hero = M4 teacher)
pokermind opponent-study --hero runs/m4-leduc-teacher/strategies/strategy_100000.json

# M6: adapt to observed opponents within an exact exploitability budget (multi-process)
pokermind adaptive-study --hero runs/m4-leduc-teacher/strategies/strategy_100000.json

# M7A: RNR and DBR vs the M6 mixture at equal exact exploitability budgets (multi-process)
pokermind robust-study --hero runs/m4-leduc-teacher/strategies/strategy_100000.json

# M7B: solver convergence, calibration (development opponents), frozen held-out
# evaluation and selection benchmark; validate the frozen configuration
pokermind reliability {convergence,calibrate,evaluate,benchmark} \
    --hero runs/m4-leduc-teacher/strategies/strategy_100000.json
pokermind research-config --check

# M8A: play against the adaptive AI through a local API (pip install -e ".[app]")
pokermind serve          # http://127.0.0.1:8000/docs

# M8B: the web interface (in a second terminal; Node.js >= 20.9)
cd web && npm install && npm run build && npm run start   # http://localhost:3000
```

`train-cfr` writes `metrics.json`, `metrics.csv`, `convergence.png` and one
strategy file per checkpoint into the output directory. `runs/` is
git-ignored. Exploitability is NashConv / 2 in chips per hand; the
conventions are documented in `docs/concepts/cfr.md`.

M2A: distill the CFR strategy into tiny networks (PyTorch) and evaluate them exactly:

```bash
pokermind train-neural --output-dir runs/m2a-neural
pokermind simulate --agents neural rule --model runs/m2a-neural/models/mlp16_lr0.01_seed2.pt
```

`train-neural` trains linear, 16 and 16×16 policies over several learning rates
and seeds, then reports imitation error, exact exploitability, payoffs, model
size and latency against the 12-entry table (`docs/concepts/neural_policy.md`).
Kuhn Poker has only 12 information sets, so these runs measure
representation, not generalization.

M2B: stability of training procedures, capacity sweep and latency benchmarks:

```bash
pokermind neural-study --output-dir runs/m2b
```

Results and methodology: `docs/experiments/m2b_results.md`.

Optional cross-check against OpenSpiel (not a project dependency):

```bash
pip install -e ".[crosscheck]"
pytest tests/test_openspiel_crosscheck.py tests/test_openspiel_leduc.py
```

The report includes the mean return per hand (± standard error) for each
agent and each seat, invalid-action counts, a zero-sum check, counts for each
terminal history, elapsed time, and a SHA-256 digest of every hand record.
The engine raises on any illegal transition, so a completed run has none. Use `--output` to write a JSON file that also records
the full configuration and environment metadata.

## Quality checks

```bash
pytest
ruff check . && ruff format --check .
mypy
```

## Engineering principles

* Environments, agents, learning algorithms and evaluation stay independent.
* Agents never see an opponent's private cards.
* Every experiment is seeded and reproducible.
* No claim that a model improved without measured results.
