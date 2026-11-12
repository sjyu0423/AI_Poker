# PokerMind

PokerMind is a local-first poker AI research and training project. The long-term
goal is an agent that learns an opponent's tendencies from the actions it
observes and adjusts its strategy to them, while staying reasonably hard to
exploit. Heads-up Texas Hold'em is the eventual target, but every algorithm is
validated in small games first.

**Status: M2B, training stability, capacity study and Leduc readiness** (M2A:
neural distillation of the CFR strategy; M1: vanilla CFR with exact
evaluation; M0: the Kuhn Poker engine).

## Layout

```
src/pokermind/
  games/kuhn.py          immutable config/state/info-state, rules, payoffs
  agents/                Agent protocol, random/rule baselines, StrategyAgent (CFR)
  strategy.py            tabular strategy type and JSON file format
  learning/cfr.py        vanilla CFR (simultaneous updates, full traversal)
  evaluation/exact.py    exact EV, best response, NashConv, exploitability
  evaluation/reference.py known Kuhn equilibrium family, game value -1/18
  neural/                M2A: encoding, teacher dataset, masked MLP, training, agent
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
docs/design/leduc_interface.md  proposed game interface for Leduc (not implemented)
research/                experiment notebooks/scripts (kept apart from the package)
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
pytest tests/test_openspiel_crosscheck.py
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
