"use client";

import { NAIVE_EXPLOITATION_ANCHORS } from "@/content/m7RobustResponse";
import { dangerCopy } from "@/content/robustResponsePage";

const maxExplVisual = 4;

export function DangerPanel() {
  const anchors = NAIVE_EXPLOITATION_ANCHORS;
  const cfrPct = (anchors.cfr.exploitability / maxExplVisual) * 100;
  const budgetPct = (anchors.rnrAtBudget.exploitabilityCap / maxExplVisual) * 100;

  return (
    <div className="demo-panel">
      <div className="demo-panel__toolbar">
        <p className="mono demo-panel__badge">Conceptual · with measured anchors</p>
      </div>
      <p className="muted">{dangerCopy.summary}</p>

      <div
        className="danger-spectrum"
        role="img"
        aria-label="Exploitability spectrum from CFR through certified budget to unrestricted best response"
      >
        <div className="danger-spectrum__track">
          <div
            className="danger-spectrum__zone danger-spectrum__zone--safe"
            style={{ width: `${budgetPct}%` }}
          />
          <span
            className="danger-spectrum__mark danger-spectrum__mark--cfr"
            style={{ left: `${cfrPct}%` }}
            title="CFR"
          />
          <span
            className="danger-spectrum__mark danger-spectrum__mark--budget"
            style={{ left: `${budgetPct}%` }}
            title="Budget δ = 0.10"
          />
          <span
            className="danger-spectrum__mark danger-spectrum__mark--br"
            style={{ left: "85%" }}
            title="Unrestricted BR"
          />
        </div>
        <div className="danger-spectrum__labels">
          <span>Safer</span>
          <span>More exploitable →</span>
        </div>
      </div>

      <div className="strategy-grid danger-cards">
        <article className="strategy-card">
          <p className="mono strategy-card__key">Exact</p>
          <h3>{anchors.cfr.label}</h3>
          <p className="strategy-card__label mono">
            expl = {anchors.cfr.exploitability.toFixed(4)}
          </p>
          <p className="muted">{anchors.cfr.gainNote}</p>
          <p className="muted">{dangerCopy.cfrNote}</p>
        </article>

        <article className="strategy-card">
          <p className="mono strategy-card__key">Measured · M6</p>
          <h3>{anchors.unrestrictedBr.label}</h3>
          <p className="strategy-card__label mono">
            expl ≈ {anchors.unrestrictedBr.exploitabilityRange}
          </p>
          <p className="muted">{anchors.unrestrictedBr.context}</p>
          <p className="muted">{dangerCopy.brNote}</p>
        </article>

        <article className="strategy-card">
          <p className="mono strategy-card__key">Measured · M7A</p>
          <h3>{anchors.mixAtBudget.label}</h3>
          <p className="strategy-card__label mono">
            gain +{anchors.mixAtBudget.gain.toFixed(3)} · cap ≤{" "}
            {anchors.mixAtBudget.exploitabilityCap.toFixed(4)}
          </p>
          <p className="muted">{anchors.mixAtBudget.context}</p>
          <p className="muted">{dangerCopy.mixNote}</p>
        </article>

        <article className="strategy-card strategy-card--accent">
          <p className="mono strategy-card__key">Measured · M7A</p>
          <h3>{anchors.rnrAtBudget.label}</h3>
          <p className="strategy-card__label mono">
            gain +{anchors.rnrAtBudget.gain.toFixed(3)} · cap ≤{" "}
            {anchors.rnrAtBudget.exploitabilityCap.toFixed(4)}
          </p>
          <p className="muted">{anchors.rnrAtBudget.context}</p>
          <p className="muted">{dangerCopy.rnrNote}</p>
        </article>
      </div>
    </div>
  );
}
