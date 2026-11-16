"use client";

import { useMemo, useState } from "react";
import { trainingCopy } from "@/content/cfrPage";
import {
  formatExploitability,
  getM1Checkpoint,
  M1_FOCUS_INFOSETS,
  M1_SOURCE,
  M1_TIMELINE_ITERATIONS,
} from "@/content/m1CfrCheckpoints";

export function TrainingTimeline() {
  const [iteration, setIteration] = useState<(typeof M1_TIMELINE_ITERATIONS)[number]>(1000);
  const checkpoint = useMemo(() => getM1Checkpoint(iteration), [iteration]);
  const maxExpl = getM1Checkpoint(1).exploitability;

  return (
    <div className="demo-panel">
      <div className="demo-panel__toolbar">
        <p className="mono demo-panel__badge">Measured · {M1_SOURCE.artifact}</p>
      </div>

      <div className="timeline-controls" role="group" aria-label="CFR training checkpoints">
        {M1_TIMELINE_ITERATIONS.map((value) => {
          const selected = value === iteration;
          return (
            <button
              key={value}
              type="button"
              className={`timeline-chip ${selected ? "is-selected" : ""}`}
              aria-pressed={selected}
              onClick={() => setIteration(value)}
            >
              {value.toLocaleString()}
            </button>
          );
        })}
      </div>

      <div className="timeline-metrics">
        <div>
          <p className="mono research-banner__label">Iterations</p>
          <p className="timeline-metrics__value mono">{checkpoint.iteration.toLocaleString()}</p>
        </div>
        <div>
          <p className="mono research-banner__label">Exploitability</p>
          <p className="timeline-metrics__value mono">
            {formatExploitability(checkpoint.exploitability)}
          </p>
          <p className="muted">chips/hand (NashConv / 2)</p>
        </div>
        <div>
          <p className="mono research-banner__label">Game-value error</p>
          <p className="timeline-metrics__value mono">
            {checkpoint.gameValueError.toExponential(2)}
          </p>
          <p className="muted">vs −1/18</p>
        </div>
        <div>
          <p className="mono research-banner__label">Strategy digest</p>
          <p className="timeline-metrics__value mono">{checkpoint.strategyDigestPrefix}…</p>
        </div>
      </div>

      <div
        className="expl-track"
        role="img"
        aria-label={`Exploitability ${formatExploitability(checkpoint.exploitability)} chips per hand`}
      >
        <div
          className="expl-track__fill"
          style={{ width: `${Math.max(2, (checkpoint.exploitability / maxExpl) * 100)}%` }}
        />
      </div>
      <p className="muted timeline-metric-note">{trainingCopy.metric}</p>

      <h3 className="timeline-strategy-title">Selected average-strategy probabilities</h3>
      <div className="strategy-grid">
        {M1_FOCUS_INFOSETS.map((info) => {
          const probs = checkpoint.selectedAverageStrategy[info.key];
          return (
            <div key={info.key} className="strategy-card">
              <p className="mono strategy-card__key">{info.key}</p>
              <p className="strategy-card__label">{info.label}</p>
              <ul>
                {info.actions.map((action) => {
                  const probability = probs[action.code] ?? 0;
                  return (
                    <li key={action.code}>
                      <div className="strategy-card__row">
                        <span>{action.name}</span>
                        <span className="mono">{(probability * 100).toFixed(1)}%</span>
                      </div>
                      <div className="prob-bar">
                        <i style={{ width: `${probability * 100}%` }} />
                      </div>
                    </li>
                  );
                })}
              </ul>
            </div>
          );
        })}
      </div>

      <p className="muted timeline-source">{trainingCopy.sourceNote}</p>
    </div>
  );
}
