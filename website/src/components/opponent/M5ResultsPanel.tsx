import {
  M5_CALIBRATION,
  M5_EXCESS_LOG_LOSS,
  M5_PUBLIC_CEILING,
  M5_SOURCE,
} from "@/content/m5OpponentModeling";
import { resultsCopy } from "@/content/opponentModelingPage";

export function M5ResultsPanel() {
  const maxLoss = M5_EXCESS_LOG_LOSS[0].excessLogLoss;

  return (
    <div className="demo-panel">
      <div className="demo-panel__toolbar">
        <p className="mono demo-panel__badge">Measured · {M5_SOURCE.artifact}</p>
      </div>
      <p className="muted">{resultsCopy.metric}</p>

      <div className="m5-chart" role="img" aria-label="Excess log loss falling with more hands">
        {M5_EXCESS_LOG_LOSS.map((row) => (
          <div key={row.hands} className="m5-chart__row">
            <span className="mono m5-chart__hands">{row.hands}</span>
            <div className="m5-chart__bar-track">
              <div
                className="m5-chart__bar"
                style={{ width: `${(row.excessLogLoss / maxLoss) * 100}%` }}
              />
            </div>
            <span className="mono m5-chart__value">{row.excessLogLoss.toFixed(3)}</span>
          </div>
        ))}
        <p className="muted m5-chart__axis">Hands observed → excess log loss (nats/decision)</p>
      </div>

      <div className="strategy-grid">
        {M5_EXCESS_LOG_LOSS.map((row) => (
          <div key={`card-${row.hands}`} className="strategy-card">
            <p className="mono strategy-card__key">{row.hands} hands</p>
            <p className="strategy-card__label">
              Excess log loss {row.excessLogLoss.toFixed(3)} [{row.ciLow.toFixed(3)},{" "}
              {row.ciHigh.toFixed(3)}]
            </p>
            <p className="muted">
              Top-1 accuracy {row.accuracy.toFixed(3)}
              {row.learnableGapClosed !== null
                ? ` · ~${Math.round(row.learnableGapClosed * 100)}% of learnable public gap closed`
                : null}
            </p>
          </div>
        ))}
      </div>

      <aside className="research-banner" aria-label="Public information ceiling">
        <div>
          <p className="mono research-banner__label">Public-information ceiling</p>
          <p className="research-banner__value">
            Card oracle is {Math.abs(M5_PUBLIC_CEILING.cardOracleRelativeToPublic).toFixed(3)} nats
            better than any public model
          </p>
          <p className="muted">{M5_PUBLIC_CEILING.note}</p>
          <p className="muted">
            Best public top-1 accuracy {M5_PUBLIC_CEILING.publicOracleAccuracy.toFixed(3)}. Rare
            round-2 re-raise spots stay sparse even after hundreds of hands.
          </p>
        </div>
      </aside>

      <p className="research-disclaimer" role="note">
        {M5_CALIBRATION.note}
      </p>
      <p className="muted timeline-source">{resultsCopy.sourceNote}</p>
    </div>
  );
}
