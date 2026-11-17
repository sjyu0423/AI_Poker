import { M7B_RELIABILITY_HELD_OUT_D01, M7_SOURCE } from "@/content/m7RobustResponse";
import { reliabilityCopy } from "@/content/robustResponsePage";

function formatGain(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

function formatPct(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`;
}

export function ReliabilityPanel() {
  const data = M7B_RELIABILITY_HELD_OUT_D01;
  const maxGain = Math.max(data.pointGain, data.frozenGain, data.mixGain);
  const maxHarm = Math.max(data.pointHarmRate, data.frozenHarmRate);

  return (
    <div className="demo-panel">
      <div className="demo-panel__toolbar">
        <p className="mono demo-panel__badge">Measured · {M7_SOURCE.m7bArtifact}</p>
      </div>
      <p className="muted">{reliabilityCopy.summary}</p>

      <div className="strategy-grid">
        <article className="strategy-card">
          <p className="mono strategy-card__key">M7A point rule</p>
          <p className="strategy-card__label mono">{formatGain(data.pointGain)} chips/hand</p>
          <p className="muted">{formatPct(data.pointHarmRate)} losing runs</p>
        </article>
        <article className="strategy-card strategy-card--accent">
          <p className="mono strategy-card__key">M7B frozen rule</p>
          <p className="strategy-card__label mono">{formatGain(data.frozenGain)} chips/hand</p>
          <p className="muted">{formatPct(data.frozenHarmRate)} losing runs</p>
        </article>
        <article className="strategy-card">
          <p className="mono strategy-card__key">M6 mixing</p>
          <p className="strategy-card__label mono">{formatGain(data.mixGain)} chips/hand</p>
          <p className="muted">Same δ = 0.10 · 100 hands · held-out</p>
        </article>
      </div>

      <div className="reliability-compare" aria-label="Gain versus harm comparison">
        <div>
          <h3 className="robust-subhead">Average gain</h3>
          {(
            [
              { label: "Point", value: data.pointGain },
              { label: "Frozen", value: data.frozenGain, accent: true },
              { label: "Mixing", value: data.mixGain },
            ] as const
          ).map((item) => (
            <div key={item.label} className="frontier-chart__row">
              <span className="frontier-chart__label">{item.label}</span>
              <div className="frontier-chart__track">
                <div
                  className={`frontier-chart__bar${"accent" in item && item.accent ? " frontier-chart__bar--accent" : ""}`}
                  style={{ width: `${(item.value / maxGain) * 100}%` }}
                />
              </div>
              <span className="mono frontier-chart__value">{formatGain(item.value)}</span>
            </div>
          ))}
        </div>
        <div>
          <h3 className="robust-subhead">Losing-run rate</h3>
          {(
            [
              { label: "Point", value: data.pointHarmRate },
              { label: "Frozen", value: data.frozenHarmRate, accent: true },
            ] as const
          ).map((item) => (
            <div key={item.label} className="frontier-chart__row">
              <span className="frontier-chart__label">{item.label}</span>
              <div className="frontier-chart__track">
                <div
                  className={`frontier-chart__bar${"accent" in item && item.accent ? " frontier-chart__bar--accent" : ""}`}
                  style={{ width: `${(item.value / maxHarm) * 100}%` }}
                />
              </div>
              <span className="mono frontier-chart__value">{formatPct(item.value)}</span>
            </div>
          ))}
        </div>
      </div>

      <p>
        Paired frozen − mix at 100 hands: {formatGain(data.frozenMinusMix.mean)} [
        {formatGain(data.frozenMinusMix.ciLow)}, {formatGain(data.frozenMinusMix.ciHigh)}].
        Frozen − point: {formatGain(data.frozenMinusPoint.mean)} [
        {formatGain(data.frozenMinusPoint.ciLow)}, {formatGain(data.frozenMinusPoint.ciHigh)}].
      </p>
      <p className="muted">{reliabilityCopy.tradeoff}</p>
      <p className="research-disclaimer" role="note">
        {data.note}
      </p>
    </div>
  );
}
