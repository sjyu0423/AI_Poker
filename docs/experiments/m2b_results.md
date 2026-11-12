# M2B results: training stability, capacity, benchmark reliability

Reproduce with `pokermind neural-study --output-dir runs/m2b` (about 2.5 min
on an Apple M6). Full per-run data is in `runs/m2b/{stability,capacity,benchmarks}.json`,
which are git-ignored. The frozen M2A numbers used for comparison are in
`m2a_baseline_results.json`.

Teacher: M1 vanilla CFR, 50,000 iterations, digest `b8217a0c…`, exploitability
**0.000999**. Python 3.12.16, NumPy 2.5.4, PyTorch 2.15.0. All runs are
3,000 full-batch Adam epochs.

**Scope.** All 12 Kuhn information sets are the training set. "Best" and
"early stopping" are therefore judged on the training KL, and nothing here
measures generalization.

## 1. Training stability

2 architectures (16, 16×16) × 2 learning rates (0.01, 0.03) × 10 seeds × 5
procedures = 200 runs.

* A **spike** is an epoch where the KL first rises above 10× its running
  minimum.
* A run is **degraded** if its exploitability is more than 1e-4 above the
  teacher's.

| Procedure | Median exploitability | Worst | Median max error | Worst max error | Spikes per run (median / max) | Epochs used (median / min) | Degraded |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` (M2A) | 0.000999 | **0.011406** | 1.7e-4 | **4.4e-2** | 44 / 90 | 3000 / 3000 | **12 / 40** |
| `best_checkpoint` | 0.000998 | 0.001041 | 2.3e-5 | 3.5e-4 | 44 / 90 | 3000 / 3000 | 0 / 40 |
| `cosine` | 0.000999 | 0.001004 | 3.1e-5 | 7.5e-5 | **4 / 19** | 3000 / 3000 | 0 / 40 |
| `cosine_best` | 0.000999 | 0.001004 | 3.1e-5 | 7.5e-5 | 4 / 19 | 3000 / 3000 | 0 / 40 |
| `early_stop_best` | 0.000998 | 0.001011 | 3.8e-5 | 1.8e-4 | 32 / 82 | 2673 / 944 | 0 / 40 |

Baseline degradation by configuration (out of 10 seeds):

| | lr 0.01 | lr 0.03 |
|---|---:|---:|
| mlp16 | 1 | 2 |
| mlp16x16 | 3 | **6** |

Findings:

* **Cosine decay addresses the cause.** Spikes per run drop from a median of
  44 to 4, and the worst run's max error falls from 4.4e-2 to 7.5e-5.
* **Best-checkpoint selection hides the symptom.** It returns a good epoch, but
  the same number of spikes happens during training, and its worst max error
  is about 5× cosine's.
* **Early stopping** (patience 200, ≥5% improvement) saves compute: median
  2,673 epochs, as few as 944. It doesn't prevent spikes, and its quality
  depends on the best-checkpoint step that follows it.
* **Initialization sensitivity:** under the baseline, outcomes depend strongly
  on the seed (exploitability ranges from 0.00095 to 0.0114). Under cosine
  decay every one of the 40 seeds lands within 0.000998–0.001004.
* **Recommended default for future distillation:** cosine decay with
  best-checkpoint selection. It is as stable as cosine alone and protects
  against a late spike. I left the M2A defaults unchanged so M2A stays
  reproducible; a regression test checks its weight digests.

## 2. Capacity

Linear, and one tanh hidden layer of width 2, 4, 8 and 16, trained with
`cosine_best`. 10 seeds × 2 learning rates = 100 runs. Latency comes from the
benchmark in §3, using seed 0 and learning rate 0.03.

| Model | Parameters | Checkpoint bytes | Max error (median, seed range), lr 0.03 | Exploitability (seed range), lr 0.03 | Degraded (lr 0.01 / 0.03) | Neural-agent decision (median) |
|---|---:|---:|---|---|---:|---:|
| linear | 32 | 2,149 | 8.1e-2 (8.1e-2 – 8.1e-2) | 0.032701 – 0.032739 | 10 / 10 | 9.6 µs |
| mlp2 | **28** | 2,653 | 1.7e-4 (1.2e-4 – 6.2e-4) | 0.000904 – 0.001035 | 6 / 0 | 12.0 µs |
| mlp4 | 52 | 2,717 | 6.2e-5 (4.5e-5 – 1.2e-4) | 0.000980 – 0.001005 | 0 / 0 | 11.9 µs |
| mlp8 | 100 | 2,909 | 2.9e-5 (2.0e-5 – 5.1e-5) | 0.000998 – 0.001003 | 0 / 0 | 12.9 µs ± 22%¹ |
| mlp16 | 196 | 3,229 | 2.2e-5 (1.5e-5 – 3.7e-5) | 0.000998 – 0.001000 | 0 / 0 | 12.2 µs |
| CFR table | 24 probabilities | 1,263 (JSON) | 0 | 0.000999 | n/a | 0.94 µs |

¹ Relative interquartile range over the 15 repeats. This was a noisy
measurement; every other row was within 4%.

Findings:

* **Two hidden units are enough** to represent this teacher closely. The
  width-2 model has *fewer* parameters than the linear model (28 vs 32) but
  reduces the max error by about 500×. The linear model's limit isn't about
  size: it can't represent the interaction between card and history
  (`docs/concepts/neural_policy.md` §9).
* Error falls steadily with width. Above width 4, exploitability stops
  depending on the model, because the teacher's own exploitability (0.000999)
  dominates.
* Exploitability *below* the teacher (as low as 0.000904 for mlp2) happens when
  the student's errors move it closer to the equilibrium. That isn't the
  training objective, so it isn't evidence of a better model.
* At lr 0.01 the width-2 model was undertrained in 6 of 10 seeds after 3,000
  epochs. Small models need the larger learning rate here.

## 3. Benchmark reliability

### Review of the M2A method

| Issue in M2A | Effect | M2B change |
|---|---|---|
| Each operation was wrapped in `lambda: ...` | About 12 ns of function-call overhead in every number. The bare dictionary lookup showed as 20 ns. | `timeit` with source-string statements; empty-statement overhead measured and subtracted |
| No consumption or correctness check | Couldn't show that the timed code did the intended work | Each statement assigns its result. Every operation is run and checked once outside the timed region. |
| Only the median of 7 repeats | No uncertainty | min, Q1, median, Q3, max and relative IQR over 15 repeats |
| Forward pass under `no_grad` | Slightly more autograd bookkeeping than needed | `torch.inference_mode` for every timed CPU and GPU statement |
| GPU time measured only with a `.cpu()` copy | Mixed launch, compute and copy | Two GPU variants: with copy, and with explicit `torch.mps.synchronize()` |

Setup (building agents, tensors, moving the model to the GPU) was already
outside the timed region in M2A, and still is.

### Results (mlp16, one CPU thread, 15 × 20,000 calls; GPU 15 × 1,000)

| Operation | M2A method | M2B median | M2B relative IQR |
|---|---:|---:|---:|
| Empty statement (overhead) | – | 2.1 ns | 1.7% |
| Bare dictionary lookup | 20 ns | **8.4 ns** (6.3 net) | 1.9% |
| Complete table-agent decision | 906 ns | 956 ns | 3.5% |
| Bare forward pass, CPU | 5,422 ns | 5,830 ns | 8.2% |
| Complete neural-agent decision, CPU | 12,371 ns | 12,206 ns | 2.0% |
| Forward on the Apple GPU, with copy to CPU | 132,754 ns | 139,204 ns | 11.2% |
| Forward on the Apple GPU, then sync | – | 128,334 ns | – |

How to read these:

* The ranking is robust: table lookup ≪ table decision < forward pass <
  neural decision ≪ GPU. On this Kuhn policy, the table decision is about
  **13× faster** than the neural decision, and a bare lookup is about **700×**
  faster than a bare forward pass.
* Nanosecond values are specific to CPython 3.12 on this machine. They change
  with the interpreter version, CPU frequency and background load. Use them
  for comparisons *within one run*, not as universal numbers.
* Python overhead dominates the cost of a decision for both agents: the
  checks, the random draw, NumPy and Torch dispatch. The arithmetic itself is
  a tiny fraction.
* The Apple GPU is slower than the CPU at every size here, because each call
  pays a fixed launch-and-sync cost of about 0.13 ms. That says nothing about
  larger models or batched inference.

## 4. Determinism

* The stability and capacity studies were run twice. All 300 rows are
  identical apart from wall-clock time.
* Re-running M2A's `train-neural` grid reproduces all 18 weight digests and
  exploitabilities in `m2a_baseline_results.json`.
* Timings are not deterministic and are reported with their spread.
