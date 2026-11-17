"use client";

import { useMemo, useState } from "react";
import { regretMatchingCopy } from "@/content/cfrPage";
import { formatProbability, regretMatching } from "@/lib/regretMatching";

const MIN = -5;
const MAX = 8;
const STEP = 0.1;

export function RegretMatchingDemo() {
  const [regretA, setRegretA] = useState<number>(regretMatchingCopy.defaults.regretA);
  const [regretB, setRegretB] = useState<number>(regretMatchingCopy.defaults.regretB);

  const result = useMemo(() => regretMatching([regretA, regretB]), [regretA, regretB]);

  function reset() {
    setRegretA(regretMatchingCopy.defaults.regretA);
    setRegretB(regretMatchingCopy.defaults.regretB);
  }

  const rows = [
    {
      id: "a",
      label: regretMatchingCopy.labels.actionA,
      regret: regretA,
      setRegret: setRegretA,
      positive: result.positive[0],
      probability: result.probabilities[0],
    },
    {
      id: "b",
      label: regretMatchingCopy.labels.actionB,
      regret: regretB,
      setRegret: setRegretB,
      positive: result.positive[1],
      probability: result.probabilities[1],
    },
  ];

  return (
    <div className="demo-panel">
      <div className="demo-panel__toolbar">
        <p className="mono demo-panel__badge">Illustrative · 2-action regret matching</p>
        <button type="button" className="btn btn--secondary demo-reset" onClick={reset}>
          Reset
        </button>
      </div>

      <p className="demo-formula mono">{regretMatchingCopy.formula}</p>

      <div className="regret-grid">
        {rows.map((row) => (
          <div key={row.id} className="regret-card">
            <div className="regret-card__head">
              <h3>{row.label}</h3>
              <span className="mono">{row.regret.toFixed(1)}</span>
            </div>
            <label className="regret-slider">
              <span className="sr-only">Cumulative regret for {row.label}</span>
              <input
                type="range"
                min={MIN}
                max={MAX}
                step={STEP}
                value={row.regret}
                onChange={(event) => row.setRegret(Number(event.target.value))}
                aria-valuemin={MIN}
                aria-valuemax={MAX}
                aria-valuenow={row.regret}
                aria-valuetext={`${row.regret.toFixed(1)} cumulative regret`}
              />
            </label>
            <label className="regret-number">
              <span>R</span>
              <input
                type="number"
                min={MIN}
                max={MAX}
                step={STEP}
                value={row.regret}
                onChange={(event) => {
                  const next = Number(event.target.value);
                  if (Number.isFinite(next)) {
                    row.setRegret(Math.min(MAX, Math.max(MIN, next)));
                  }
                }}
              />
            </label>
            <dl className="regret-stats">
              <div>
                <dt>R⁺</dt>
                <dd className="mono">{row.positive.toFixed(2)}</dd>
              </div>
              <div>
                <dt>σ</dt>
                <dd className="mono">{formatProbability(row.probability)}</dd>
              </div>
            </dl>
            <div
              className="prob-bar"
              role="img"
              aria-label={`${row.label} probability ${formatProbability(row.probability)}`}
            >
              <i style={{ width: `${row.probability * 100}%` }} />
            </div>
          </div>
        ))}
      </div>

      <div className="regret-summary" role="status" aria-live="polite">
        {result.usedUniformFallback ? (
          <p>
            All regrets are ≤ 0 (positive sum {result.positiveSum.toFixed(2)}). Falling back to{" "}
            <strong>uniform</strong> play.
          </p>
        ) : (
          <p>
            Positive regret sum {result.positiveSum.toFixed(2)}. Probabilities are proportional to
            R⁺.
          </p>
        )}
      </div>

      <ul className="demo-notes">
        {regretMatchingCopy.notes.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>
    </div>
  );
}
