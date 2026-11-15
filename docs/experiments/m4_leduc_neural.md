# M4 results: neural distillation and generalization on Leduc Hold'em

Reproduce (Apple M-series Mac, CPU, about 31 min + 6 min):

```bash
pokermind train-cfr --game leduc --checkpoints 1000 10000 20000 50000 100000 \
    --output-dir runs/m4-leduc-teacher
pokermind leduc-neural --teacher runs/m4-leduc-teacher/strategies/strategy_100000.json \
    --comparison-teacher runs/m3-leduc-cfr/strategies/strategy_010000.json
```

All values below come from `runs/m4-leduc-neural/metrics.json` and
`runs/m4-leduc-teacher/metrics.json`. Everything except timings is
deterministic. Exploitability is NashConv / 2 in chips per hand; the largest
possible loss in one hand is 13 chips. Encoding and splits:
`docs/design/leduc_encoding.md`.

## 1. Teacher

Vanilla CFR, one deterministic run, evaluated exactly at each checkpoint:

| Iterations | Exploitability | Cumulative time | Strategy digest |
|---:|---:|---:|---|
| 10,000 (= the M3 teacher) | 0.010530 | 188 s | `131e09cb…` (identical to M3's file) |
| 20,000 | 0.007011 | 378 s | `3d8ee78a…` |
| 50,000 | 0.004183 | 940 s | `f918f84c…` |
| **100,000 (M4 teacher)** | **0.002997** | 1,858 s | `3f1136d492159d56eab73efc2c9a365dfc930eb20d7ae13b105cc994e17874bf` |

* 53.8 iterations/s, 4.9 µs per decision-node visit, all 936 information
  sets visited, process peak RSS 37 MiB. Each exact evaluation takes 0.1 s.
* The teacher is exactly suit-symmetric: suit-twin information sets have
  bit-identical probabilities.
* Exact EV vs uniform +0.7166 per hand. Simulated over 100k hands:
  +0.7440 ± 0.0139, z = +1.97. That is just inside two standard errors, so it
  is consistent with the exact value but at the edge.
* The 10k and 100k teachers differ by a mean total variation of 0.024 per
  state, but by up to 0.61 at one state, and by more than 0.05 at 42 of the
  288 canonical states. Both are close to equilibrium, so some of each
  state's mixing is not pinned down by equilibrium play.

## 2. Setup

* 288 canonical states, one example each. Full-batch Adam with the
  soft-target cross-entropy masked to the legal actions, learning rate 0.01,
  cosine decay to 1 % over 5,000 epochs (the M2B procedure). 10 seeds per
  configuration.
* Checkpoint rules:
  * `train_best`: lowest training KL, on the full split;
  * `validation`: lowest KL on 23 validation states, never the test states;
  * `final`: the last epoch.
* Learning rate and epochs were chosen in a pilot on the validation sets of
  the random and history splits (0.003, 0.01 and 0.03 tried). They were
  not tuned on test data.
* A weight-decay ablation (`mlp32_wd0.001`, Adam L2 1e-3) uses the same
  protocol.

## 3. Representation: fitting all 288 states (`full` split)

| Model | Params | Train KL | Exploitability (mean ± sd, 10 seeds) | EV vs teacher | EV vs uniform |
|---|---:|---:|---:|---:|---:|
| teacher (table) | – | – | **0.0030** | 0 | +0.7166 |
| linear | 93 | 0.2135 | 0.960 ± 0.003 | −0.172 | +0.680 |
| mlp16 | 547 | 0.0007 | 0.0080 ± 0.0023 | −0.0005 | +0.7160 |
| mlp32 | 1,091 | 0.0001 | **0.0032 ± 0.0001** | −0.0001 | +0.7165 |
| mlp64 | 2,179 | 0.0001 | **0.0031 ± 0.00004** | −0.0001 | +0.7165 |
| mlp32 + weight decay 1e-3 | 1,091 | 0.0220 | 0.202 ± 0.015 | −0.027 | +0.711 |

* With 32 or 64 hidden units, the network *represents* the teacher almost
  exactly: within 0.0002 chips/hand of the teacher's exploitability, on
  every seed.
* 16 units is close (2.7× the teacher's exploitability, still 0.008).
* The linear model cannot represent the strategy. It needs feature
  interactions (card × history) that the explicit pair and contribution
  features don't supply, and it is exploitable by almost a chip per hand.
* Weight decay prevents the exact fit and costs 65× in exploitability.
* **The student is only as good as the teacher.** The same mlp64 trained on
  the M3 10k teacher reaches 0.0106 (teacher 0.0105), and on the 100k teacher
  0.0031 (5 seeds each).
* Simulation check (mlp64 as a sampling agent vs uniform, 50k hands):
  +0.7125 ± 0.0197 vs exact +0.7165, z = −0.20.

## 4. Generalization to unseen states

Test sets, from `split_coverage`:

| Split | Test states (info sets) | Round 1 / 2 | Out of distribution in | Mean L1-nearest Hamming distance |
|---|---|---|---|---:|
| random | 58 (196) | 6 / 52 | nothing: every card combination and history occurs in training | 2.2 |
| public | 60 (180) | 0 / 60 | the (own, public) rank combination | 2.5 |
| history | 57 (186) | 3 / 54 | the round-1 betting line (`rr…`) | 4.0 |

In every split:
* no test feature vector and no test information-set key is in the
  training or validation data;
* no test state switches on a feature that is never on in training.

### Imitation on test states (mean over 10 seeds)

KL(p ‖ q) with q floored at 1e-12; TV = total variation; "argmax" = share of
states where the most likely action matches the teacher's.

| Split | Model / selection | Train KL | Test KL | Test TV | Argmax |
|---|---|---:|---:|---:|---:|
| random | linear / final | 0.193 | 0.445 | 0.217 | 0.79 |
| random | mlp64 / validation | 0.280 | 0.416 | 0.313 | 0.77 |
| random | mlp64 / final | 0.000 | 0.948 | 0.238 | 0.76 |
| random | uniform baseline | – | 0.611 | 0.458 | 0.60 |
| random | nearest baseline | – | 1.391 | 0.397 | 0.62 |
| public | linear / final | 0.215 | **0.330** | **0.193** | **0.87** |
| public | mlp64 / validation | 0.247 | 1.164 | 0.611 | 0.25 |
| public | mlp64 / final | 0.000 | 0.699 | 0.254 | 0.75 |
| public | uniform baseline | – | 0.660 | 0.468 | 0.40 |
| history | mlp64 / validation | 0.046 | 0.392 | 0.198 | 0.79 |
| history | mlp64 / final | 0.000 | 0.512 | 0.199 | 0.77 |
| history | nearest baseline | – | 0.432 | **0.164** | **0.81** |
| history | uniform baseline | – | 0.581 | 0.426 | 0.42 |

### Exact exploitability

* **Full:** the network's policy on all 936 information sets.
* **Hybrid:** the teacher on fitted states and the model only on the test
  states. This isolates the cost of the held-out predictions.

| Split | Model / selection | Full expl. | Hybrid expl. (mean ± sd) | EV vs teacher (full) |
|---|---|---:|---:|---:|
| random | linear / final | 1.161 | 1.233 ± 0.000 | −0.156 |
| random | mlp16 / validation | 1.687 | 1.357 ± 0.060 | −0.296 |
| random | mlp32 / validation | 1.749 | 1.371 ± 0.028 | −0.311 |
| random | mlp64 / validation | 1.665 | 1.322 ± 0.049 | −0.326 |
| random | mlp64 / final | 1.630 | 1.547 ± 0.063 | −0.021 |
| random | uniform / nearest / history-mean | – | 1.414 / 1.968 / 1.829 | |
| public | linear / final | 0.809 | **0.248 ± 0.000** | −0.196 |
| public | mlp64 / validation | 1.549 | 1.352 ± 0.059 | −0.318 |
| public | mlp64 / final | 0.936 | 0.813 ± 0.384 (best 0.204) | −0.021 |
| public | uniform / nearest / history-mean | – | 0.903 / 0.991 / 0.860 | |
| history | linear / final | 1.138 | 0.752 ± 0.022 | −0.197 |
| history | mlp64 / validation | 0.830 | 0.717 ± 0.178 | −0.053 |
| history | mlp64 / final | 0.735 | 0.693 ± 0.307 | −0.023 |
| history | uniform / nearest / history-mean | – | 1.102 / **0.349** / 1.102 | |

Every configuration and seed is in `runs.csv`; `metrics.json` adds the
min, median and max per group.

### What the numbers say

1. **No model generalizes to unseen states well enough to play them.**
   Replacing only about 20 % of the states (the test states) with model
   predictions raises exploitability from 0.003 to between 0.2 and 1.6
   chips/hand. Uniform play on the same states gives 0.9–1.4.
   * On the in-distribution random split, the networks are *no better than
     uniform* in exploitability (1.32–1.55 vs 1.41). They are better in
     imitation metrics (KL 0.42 vs 0.61, argmax 0.77 vs 0.60).
2. **Round-1 mistakes dominate, and they are systematic, not noise.** The
   random split holds out 6 of the 18 round-1 states, including `J|`, `J|r`
   and `J|cr`. In round 1 the only card feature is the own rank, so a model
   that never saw J at these histories extrapolates Q/K behaviour. For
   example, mlp64 (final, seed 0):
   * at `J|` it bets 97 % (teacher 8 %);
   * at `J|r` it re-raises 99.7 % (teacher folds 82 %).

   Putting the model on just the 6 round-1 test states gives hybrid
   exploitability 1.40. The 52 round-2 states give 0.86. Uniform play at a
   single round-1 state costs 0.07–0.48; at a single round-2 state, a median
   of 0.03.
3. **Imitation metrics disagree with each other and with exploitability.**
   * Validation-selected checkpoints stop after 20–130 epochs. They have the
     lowest test KL, but they underfit the training states (train KL about
     0.25) and give the worst full policies.
   * Final-epoch models fit the training states exactly. Their test KL is
     2–4× higher, because confident mistakes are heavily penalized, but
     their TV and argmax are often better.
   * A final-epoch mlp64 loses only 0.02 chips/hand to the teacher, yet is
     exploitable by 1.6. **Near-teacher payoff against the teacher says
     almost nothing about robustness.**
4. **Inductive bias matters more than capacity.**
   * On the `public` split the linear model generalizes best: hybrid 0.25,
     argmax 0.87. Its additive structure carries over what it learned about
     "pair" and "own rank" to the unseen (Q, Q) and (J, K) combinations.
   * The MLPs memorize interactions instead. Validation-selected MLPs agree
     with the teacher's top action on only 19–25 % of the public-split test
     states, worse than uniform's 40 %.
   * On the `history` split a plain nearest-neighbour lookup (hybrid 0.35)
     beats every network (0.69–0.92).
5. **Seed sensitivity is large on held-out states.** For example, public /
   mlp64 / final ranges from 0.20 to 1.64 across seeds. On the full split
   the seed spread is negligible (sd 0.00004).
6. **Weight decay (1e-3) does not fix generalization.** It gives a modest
   test-KL improvement on `public` and `history` (final epoch), the same or
   worse exploitability, and much worse fitting.

## 5. Performance (CPU, 1 thread, `torch.inference_mode`)

Medians of 9 timing runs, net of loop overhead.

| | Table (teacher) | linear | mlp16 | mlp32 | mlp64 |
|---|---:|---:|---:|---:|---:|
| Parameters | 2,184 probabilities over 936 keys | 93 | 547 | 1,091 | 2,179 |
| Parameter bytes (float32) | 17 KB as float64 values | 372 B | 2.2 KB | 4.4 KB | 8.7 KB |
| File | 95.5 KB JSON | 2.5 KB | 4.7 KB | 6.9 KB | 11.2 KB |
| In-memory | ~398 KB of Python dicts | – | – | – | – |
| Bare lookup / forward, 1 state | **6 ns** | 3.4 µs | 5.5 µs | 5.6 µs | 5.7 µs |
| Complete decision (encode, infer, sample) | **1.0 µs** | 12.3 µs | 14.8 µs | 14.6 µs | 14.8 µs |
| Batched forward, 288 states, per state | – | 22 ns | 83 ns | 95 ns | 158 ns |
| Batched forward, 4,096 states, per state | – | 8 ns | 39 ns | 70 ns | 132 ns |
| Apple GPU (MPS), 4,096 batch + copy, per state | – | 35 ns | 46 ns | 47 ns | 49 ns |

* Encoding one observation costs 1.4 µs. Process peak RSS for the study is
  305 MiB, mostly PyTorch itself; CFR training alone peaks at 37 MiB.
* MPS logits differ from the CPU's by at most 5e-6.
* In float32, a single-row forward pass and a batched one differ by about
  1e-8 in probability, because they use different matrix kernels. Tests use
  1e-6 for that comparison.

**The table is better for Leduc in every practical respect.** It is exact,
about 15× faster per decision, needs no PyTorch, and 95 KB is trivial. The
network's only advantages are file size (11 KB vs 95 KB, though a binary
table of 288 canonical states would be ~7 KB) and batched throughput on a
GPU. Neither matters at this scale. A network is worth considering only when
the state space is too large to tabulate, and then its value depends on
generalization, which these results show is weak without more structure.

## 6. Limitations

* One CFR teacher, one game, uniform weighting of canonical states. Held-out
  errors are weighted equally even though round-1 states are reached in
  every hand.
* Splits use one deterministic seed each, with 57–60 test states. The 23-state
  validation sets are small, so validation selection is noisy.
* The teacher's mixed strategy is one of many near-equilibria (the 10k vs
  100k comparison in §1). Imitation error on held-out states is measured
  against that particular solution, not a unique target.
* Only small one-hidden-layer tanh MLPs, one optimizer and one learning-rate
  setting were studied. Other architectures, losses or data augmentation
  were not explored, deliberately, since the evidence doesn't single out
  architecture as the bottleneck.
* Latency figures are single-machine CPython measurements.
