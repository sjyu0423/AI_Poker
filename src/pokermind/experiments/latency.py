"""Latency benchmarks with explicit methodology (M2B).

Changes from the M2A benchmark (``neural_experiment.benchmark_latency``, kept
unchanged so M2A stays reproducible):

* Statements are timed with :mod:`timeit` as source strings (``"table[key]"``),
  not as ``lambda: ...`` calls. The M2A numbers included one Python function
  call (~20-40 ns) in every measurement, which dominated the dictionary lookup.
* Each statement assigns its result (``_ = ...``), and every operation's output
  is computed once outside the timed region and checked against the expected
  value, so timings correspond to work that is actually done and correct.
* An empty statement (``pass``) is timed the same way. Its cost, the loop
  overhead, is reported, and also subtracted in ``net_median_ns``.
* Uncertainty is reported as min, median, interquartile range and max over
  ``repeats`` independent timing runs, plus their spread relative to the median.
* All setup (building agents, tensors, moving the model to the GPU) happens
  before timing.

These are single-machine, single-process, CPython measurements. Nanosecond
results depend on the interpreter version, CPU frequency scaling, thermal
state and background load. They are useful for comparing operations
measured together, not as absolute or universal numbers.
"""

from __future__ import annotations

import platform
import statistics
import sys
import timeit
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from pokermind.agents import StrategyAgent
from pokermind.games.kuhn import Action, Card, InfoState
from pokermind.neural.agent import NeuralPolicyAgent
from pokermind.neural.encoding import encode_info_state
from pokermind.neural.model import PolicyNetwork
from pokermind.strategy import Strategy


@dataclass(frozen=True, slots=True)
class Timing:
    name: str
    description: str
    number: int
    """Executions per timing run."""
    repeats: int
    min_ns: float
    median_ns: float
    q1_ns: float
    q3_ns: float
    max_ns: float
    net_median_ns: float
    """Median minus the median cost of an empty statement (loop overhead)."""

    @property
    def relative_iqr(self) -> float:
        return (self.q3_ns - self.q1_ns) / self.median_ns if self.median_ns else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "relative_iqr": self.relative_iqr}


def _quartiles(samples: list[float]) -> tuple[float, float, float]:
    q = statistics.quantiles(samples, n=4, method="inclusive")
    return q[0], q[1], q[2]


def time_statement(
    name: str,
    description: str,
    stmt: str,
    namespace: dict[str, Any],
    *,
    number: int,
    repeats: int,
    overhead_ns: float = 0.0,
    sync: Callable[[], None] | None = None,
) -> Timing:
    """Time ``stmt`` (executed ``number`` times per run, ``repeats`` runs)."""
    timer = timeit.Timer(stmt, globals=namespace)
    timer.timeit(number=max(number // 10, 10))  # warm-up
    samples = []
    for _ in range(repeats):
        if sync is not None:
            sync()
        samples.append(timer.timeit(number=number) * 1e9 / number)
    q1, median, q3 = _quartiles(samples)
    return Timing(
        name=name,
        description=description,
        number=number,
        repeats=repeats,
        min_ns=min(samples),
        median_ns=median,
        q1_ns=q1,
        q3_ns=q3,
        max_ns=max(samples),
        net_median_ns=median - overhead_ns,
    )


def benchmark_policies(
    model: PolicyNetwork,
    table: Strategy,
    *,
    number: int = 20_000,
    repeats: int = 15,
    include_mps: bool = True,
) -> dict[str, Any]:
    """Compare the four operations the M2B brief asks to separate, plus GPU variants.

    1. bare dictionary lookup: ``table[key]``
    2. complete table-agent decision: ``StrategyAgent.act`` (lookup, checks, one
       uniform draw, CDF inversion)
    3. bare network forward pass: ``model(x)`` on a prepared 1x7 tensor, under
       ``torch.inference_mode``
    4. complete neural-agent decision: ``NeuralPolicyAgent.act`` (encode, forward,
       float64 masked softmax, checks, one uniform draw, CDF inversion)
    """
    info = InfoState(player=1, private_card=Card.Q, history=(Action.BET,))
    legal = (Action.FOLD, Action.CALL)
    key = info.key
    table_agent = StrategyAgent(table)
    neural_agent = NeuralPolicyAgent(model)
    x1 = torch.from_numpy(encode_info_state(info)).unsqueeze(0)

    # Validate every timed operation once, outside the timed region.
    assert table[key] == table_agent.strategy[key]
    with torch.inference_mode():
        expected_logits = model(x1)
    assert expected_logits.shape == (1, 4) and bool(torch.isfinite(expected_logits).all())
    rng_check = np.random.default_rng(0)
    assert table_agent.act(info, legal, rng_check) in legal
    assert neural_agent.act(info, legal, rng_check) in legal

    namespace: dict[str, Any] = {
        "table": table,
        "key": key,
        "table_agent": table_agent,
        "neural_agent": neural_agent,
        "model": model,
        "x1": x1,
        "info": info,
        "legal": legal,
        "rng": np.random.default_rng(1),
    }
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode():
            empty = time_statement(
                "empty_statement",
                "loop overhead: `pass`",
                "pass",
                namespace,
                number=number,
                repeats=repeats,
            )
            overhead = empty.median_ns
            spec = [
                ("dict_lookup", "bare dictionary lookup", "_ = table[key]"),
                (
                    "table_agent_decision",
                    "complete StrategyAgent.act",
                    "_ = table_agent.act(info, legal, rng)",
                ),
                ("nn_forward_cpu", "bare model(x) on a 1x7 CPU tensor", "_ = model(x1)"),
                (
                    "neural_agent_decision",
                    "complete NeuralPolicyAgent.act (CPU)",
                    "_ = neural_agent.act(info, legal, rng)",
                ),
            ]
            timings = [empty] + [
                time_statement(
                    n, d, s, namespace, number=number, repeats=repeats, overhead_ns=overhead
                )
                for n, d, s in spec
            ]
            gpu = (
                _benchmark_mps(model, x1, expected_logits, number, repeats, overhead)
                if include_mps
                else {"available": False, "reason": "disabled"}
            )
    finally:
        torch.set_num_threads(previous_threads)

    return {
        "conditions": {
            "timer": "timeit with source-string statements (no lambda call)",
            "number_per_repeat": number,
            "repeats": repeats,
            "torch_threads": 1,
            "torch_mode": "inference_mode",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "caveat": "single machine and process; compare operations measured together",
        },
        "timings": {t.name: t.to_dict() for t in timings},
        "mps": gpu,
    }


def _benchmark_mps(
    model: PolicyNetwork,
    x1: torch.Tensor,
    expected_logits: torch.Tensor,
    number: int,
    repeats: int,
    overhead_ns: float,
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {"available": False, "reason": "torch.backends.mps.is_available() is False"}
    device = torch.device("mps")
    gpu_model = PolicyNetwork(model.config).to(device).eval()
    gpu_model.load_state_dict(model.state_dict())
    gx1 = x1.to(device)
    out = gpu_model(gx1).cpu()
    diff = float((out - expected_logits).abs().max())
    namespace = {"gpu_model": gpu_model, "gx1": gx1}
    # The GPU runs asynchronously; .cpu() forces completion and is part of the cost
    # whenever a CPU-side agent needs the probabilities.
    n = max(number // 20, 200)
    sync = torch.mps.synchronize
    with_copy = time_statement(
        "nn_forward_mps_with_copy",
        "model(x) on the Apple GPU + copy of the logits to the CPU",
        "_ = gpu_model(gx1).cpu()",
        namespace,
        number=n,
        repeats=repeats,
        overhead_ns=overhead_ns,
        sync=sync,
    )
    enqueue = time_statement(
        "nn_forward_mps_enqueue_then_sync",
        "model(x) on the Apple GPU, then torch.mps.synchronize()",
        "_ = gpu_model(gx1); sync()",
        {**namespace, "sync": sync},
        number=n,
        repeats=repeats,
        overhead_ns=overhead_ns,
        sync=sync,
    )
    return {
        "available": True,
        "max_abs_logit_diff_vs_cpu": diff,
        "timings": {t.name: t.to_dict() for t in (with_copy, enqueue)},
    }
