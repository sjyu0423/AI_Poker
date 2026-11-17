import {
  M7A_PERSONAL_INCREMENT_HELD_OUT_D01,
  M7A_PERSONAL_INCREMENT_OOP_D01,
  M7A_RNR_HELD_OUT_BY_HANDS_D01,
  M7_SOURCE,
} from "@/content/m7RobustResponse";
import { personalCopy } from "@/content/robustResponsePage";

function formatGain(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

export function PersonalizationPanel() {
  const maxRnr = Math.max(...M7A_RNR_HELD_OUT_BY_HANDS_D01.map((row) => row.rnr));
  const focusHands = new Set([0, 100, 500]);
  const focusRows = M7A_RNR_HELD_OUT_BY_HANDS_D01.filter((row) =>
    focusHands.has(row.hands),
  );

  return (
    <div className="demo-panel">
      <div className="demo-panel__toolbar">
        <p className="mono demo-panel__badge">Measured · δ = 0.10 · {M7_SOURCE.m7aArtifact}</p>
      </div>
      <p className="muted">{personalCopy.summary}</p>

      <div className="strategy-grid">
        {focusRows.map((row) => (
          <article key={row.hands} className="strategy-card">
            <p className="mono strategy-card__key">{row.hands} hands</p>
            <p className="strategy-card__label mono">RNR {formatGain(row.rnr)}</p>
            <p className="muted">Mixing {formatGain(row.mix)} · held-out</p>
          </article>
        ))}
      </div>

      <div
        className="m5-chart"
        role="img"
        aria-label="RNR held-out gain rising slowly with more observed hands"
      >
        {M7A_RNR_HELD_OUT_BY_HANDS_D01.map((row) => (
          <div key={row.hands} className="m5-chart__row">
            <span className="mono m5-chart__hands">{row.hands}</span>
            <div className="m5-chart__bar-track">
              <div
                className="m5-chart__bar"
                style={{ width: `${(row.rnr / maxRnr) * 100}%` }}
              />
            </div>
            <span className="mono m5-chart__value">{formatGain(row.rnr)}</span>
          </div>
        ))}
        <p className="muted m5-chart__axis">
          Hands observed → RNR gain over CFR (chips/hand, held-out, δ = 0.10)
        </p>
      </div>

      <h3 className="robust-subhead">Increment beyond the population prior</h3>
      <p className="muted">
        Personal increment = gain after n hands minus gain with 0 hands (population model only).
      </p>
      <div className="strategy-grid">
        {M7A_PERSONAL_INCREMENT_HELD_OUT_D01.map((row) => (
          <article key={`inc-${row.hands}`} className="strategy-card">
            <p className="mono strategy-card__key">Held-out · {row.hands} hands</p>
            <p className="strategy-card__label mono">
              {formatGain(row.mean)} [{formatGain(row.ciLow)}, {formatGain(row.ciHigh)}]
            </p>
          </article>
        ))}
        {M7A_PERSONAL_INCREMENT_OOP_D01.map((row) => (
          <article key={`oop-${row.hands}`} className="strategy-card">
            <p className="mono strategy-card__key">Out-of-pop. · {row.hands} hands</p>
            <p className="strategy-card__label mono">
              {formatGain(row.mean)} [{formatGain(row.ciLow)}, {formatGain(row.ciHigh)}]
            </p>
          </article>
        ))}
      </div>

      <p className="research-disclaimer" role="note">
        {personalCopy.caveat}
      </p>
    </div>
  );
}
