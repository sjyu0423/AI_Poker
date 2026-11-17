"use client";

import { useId, useState } from "react";
import { rnrCopy } from "@/content/robustResponsePage";

export function RnrConstructionDial() {
  const [p, setP] = useState(0.3);
  const dialId = useId();
  const restrictedPct = Math.round(p * 100);
  const freePct = 100 - restrictedPct;

  return (
    <div className="demo-panel">
      <div className="demo-panel__toolbar">
        <p className="mono demo-panel__badge">Conceptual · p dial</p>
      </div>
      <p>{rnrCopy.summary}</p>

      <div className="rnr-coin" aria-hidden="true">
        <div className="rnr-coin__branch" style={{ flexGrow: Math.max(p, 0.08) }}>
          <span className="mono">p = {p.toFixed(2)}</span>
          <strong>Restricted</strong>
          <span>Must play σ̂</span>
        </div>
        <div className="rnr-coin__branch rnr-coin__branch--free" style={{ flexGrow: Math.max(1 - p, 0.08) }}>
          <span className="mono">1 − p = {(1 - p).toFixed(2)}</span>
          <strong>Free</strong>
          <span>Any strategy</span>
        </div>
      </div>

      <div className="rnr-bars" role="img" aria-label={`Opponent restricted ${restrictedPct} percent of the time`}>
        <div className="rnr-bars__row">
          <span>Restricted</span>
          <div className="rnr-bars__track">
            <div className="rnr-bars__fill rnr-bars__fill--restricted" style={{ width: `${restrictedPct}%` }} />
          </div>
          <span className="mono">{restrictedPct}%</span>
        </div>
        <div className="rnr-bars__row">
          <span>Free</span>
          <div className="rnr-bars__track">
            <div className="rnr-bars__fill rnr-bars__fill--free" style={{ width: `${freePct}%` }} />
          </div>
          <span className="mono">{freePct}%</span>
        </div>
      </div>

      <label className="bayes-field" htmlFor={dialId}>
        Restriction probability p
        <input
          id={dialId}
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={p}
          onChange={(event) => setP(Number(event.target.value))}
          aria-valuemin={0}
          aria-valuemax={1}
          aria-valuenow={p}
          aria-valuetext={`p equals ${p.toFixed(2)}`}
        />
        <span className="mono">p = {p.toFixed(2)}</span>
      </label>

      <ul className="rnr-notes">
        <li>{rnrCopy.pDefinition}</li>
        <li>
          <strong>Important:</strong> {rnrCopy.pNotBudget}
        </li>
        <li>{rnrCopy.extremes}</li>
      </ul>
      <p className="research-disclaimer" role="note">
        {rnrCopy.dialDisclaimer}
      </p>
    </div>
  );
}
