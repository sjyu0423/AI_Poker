"use client";

import { useMemo, useState } from "react";
import { bayesCopy, uncertaintyCopy } from "@/content/opponentModelingPage";
import { betaBinomialPosterior } from "@/lib/betaBinomial";

function clampStrength(value: number): number {
  return Math.min(40, Math.max(1, value));
}

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function BayesianLearner() {
  const defaults = bayesCopy.presets[0];
  const [priorMean, setPriorMean] = useState<number>(defaults.priorMean);
  const [priorStrength, setPriorStrength] = useState<number>(defaults.priorStrength);
  const [successes, setSuccesses] = useState<number>(defaults.successes);
  const [trials, setTrials] = useState<number>(defaults.trials);

  const posterior = useMemo(
    () =>
      betaBinomialPosterior({
        priorMean,
        priorStrength,
        successes,
        trials,
      }),
    [priorMean, priorStrength, successes, trials],
  );

  function applyPreset(id: string) {
    const preset = bayesCopy.presets.find((row) => row.id === id);
    if (!preset) {
      return;
    }
    setPriorMean(preset.priorMean);
    setPriorStrength(preset.priorStrength);
    setSuccesses(preset.successes);
    setTrials(preset.trials);
  }

  function reset() {
    applyPreset("prior");
  }

  function observe(success: boolean) {
    setTrials((value) => value + 1);
    if (success) {
      setSuccesses((value) => value + 1);
    }
  }

  const intervalWidth = posterior.interval90.upper - posterior.interval90.lower;
  const meanPct = posterior.mean * 100;
  const lowerPct = posterior.interval90.lower * 100;
  const upperPct = posterior.interval90.upper * 100;

  return (
    <div className="demo-panel">
      <div className="demo-panel__toolbar">
        <p className="mono demo-panel__badge">Illustrative · Beta–Binomial</p>
        <button type="button" className="btn btn--secondary demo-reset" onClick={reset}>
          Reset
        </button>
      </div>

      <p className="demo-formula mono">{bayesCopy.formula}</p>
      <p className="research-disclaimer" role="note">
        {bayesCopy.disclaimer}
      </p>

      <div className="timeline-controls" role="group" aria-label="Example presets">
        {bayesCopy.presets.map((preset) => (
          <button
            key={preset.id}
            type="button"
            className="timeline-chip"
            onClick={() => applyPreset(preset.id)}
          >
            {preset.label}
          </button>
        ))}
      </div>

      <div className="bayes-controls">
        <label className="bayes-field">
          <span>Population prior mean (μ)</span>
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={priorMean}
            onChange={(event) => setPriorMean(Number(event.target.value))}
            aria-valuetext={`Prior mean ${formatPct(priorMean)}`}
          />
          <span className="mono">{formatPct(priorMean)}</span>
        </label>
        <label className="bayes-field">
          <span>Prior strength (s)</span>
          <input
            type="range"
            min={1}
            max={40}
            step={1}
            value={priorStrength}
            onChange={(event) => setPriorStrength(clampStrength(Number(event.target.value)))}
            aria-valuetext={`Prior strength ${priorStrength}`}
          />
          <span className="mono">{priorStrength.toFixed(0)}</span>
        </label>
      </div>

      <div className="hero__actions">
        <button type="button" className="btn btn--primary" onClick={() => observe(true)}>
          Observe {bayesCopy.successLabel}
        </button>
        <button type="button" className="btn btn--secondary" onClick={() => observe(false)}>
          Observe {bayesCopy.failureLabel}
        </button>
      </div>

      <div className="timeline-metrics bayes-metrics">
        <div>
          <p className="mono research-banner__label">Observations</p>
          <p className="timeline-metrics__value mono">
            {successes} / {trials}
          </p>
          <p className="muted">{bayesCopy.tendencyLabel}</p>
        </div>
        <div>
          <p className="mono research-banner__label">Observed frequency</p>
          <p className="timeline-metrics__value mono">
            {posterior.observedFrequency === null
              ? "—"
              : formatPct(posterior.observedFrequency)}
          </p>
        </div>
        <div>
          <p className="mono research-banner__label">Posterior mean</p>
          <p className="timeline-metrics__value mono">{formatPct(posterior.mean)}</p>
        </div>
        <div>
          <p className="mono research-banner__label">90% credible interval</p>
          <p className="timeline-metrics__value mono">
            {formatPct(posterior.interval90.lower)} – {formatPct(posterior.interval90.upper)}
          </p>
          <p className="muted">width {formatPct(intervalWidth)}</p>
        </div>
      </div>

      <div
        className="bayes-track"
        role="img"
        aria-label={`Posterior mean ${formatPct(posterior.mean)}, 90 percent interval from ${formatPct(posterior.interval90.lower)} to ${formatPct(posterior.interval90.upper)}`}
      >
        <div className="bayes-track__prior" style={{ left: `${priorMean * 100}%` }} title="Prior mean" />
        <div
          className="bayes-track__interval"
          style={{ left: `${lowerPct}%`, width: `${Math.max(0.5, upperPct - lowerPct)}%` }}
        />
        <div className="bayes-track__mean" style={{ left: `${meanPct}%` }} />
      </div>
      <div className="bayes-track__legend muted">
        <span>Prior</span>
        <span>90% interval</span>
        <span>Posterior mean</span>
      </div>

      <div id="uncertainty" className="bayes-uncertainty">
        <h3>{uncertaintyCopy.title}</h3>
        <p>{uncertaintyCopy.summary}</p>
        <p className="muted">{uncertaintyCopy.intervalNote}</p>
        <p className="mono muted">
          Posterior Beta(α={posterior.alpha.toFixed(2)}, β={posterior.beta.toFixed(2)}); σ ≈{" "}
          {formatPct(posterior.stdDev)}
        </p>
      </div>
    </div>
  );
}
