import {
  guarantees,
  researchDisclaimer,
  researchHighlights,
  links,
} from "@/content/site";
import { withBasePath } from "@/lib/paths";
import { Button } from "@/components/Button";

const kindLabel = {
  exact: "Exact guarantee",
  measured: "Measured (simulated opponents)",
  engineering: "Engineering measurement",
} as const;

export function ResearchHighlights() {
  return (
    <section className="section research" id="research" aria-labelledby="research-title">
      <div className="shell">
        <header className="section-header">
          <p className="section-kicker">Evidence</p>
          <h2 id="research-title">Research highlights</h2>
          <p>
            Grounded results from the milestone reports. Exact guarantees are
            computer-checked; measured gains use simulated Leduc opponents with
            confidence intervals.
          </p>
        </header>

        <aside className="research-banner" aria-label="Safety budget">
          <div>
            <p className="mono research-banner__label">Exact guarantee</p>
            <p className="research-banner__value">
              Every played strategy ≤ {guarantees.budgetLimit}
            </p>
            <p className="muted">
              Baseline CFR exploitability {guarantees.baselineExploitability}; budget
              δ = {guarantees.budgetDelta}. Independent evaluators agree to{" "}
              {guarantees.evaluatorsAgree}.
            </p>
          </div>
          <a href={links.researchFreeze} target="_blank" rel="noopener noreferrer">
            Research freeze →
          </a>
        </aside>

        <ul className="research-grid">
          {researchHighlights.map((item) => (
            <li key={item.id} className="research-card">
              <div className="research-card__meta">
                <span className={`pill pill--${item.kind}`}>{kindLabel[item.kind]}</span>
              </div>
              <h3>{item.title}</h3>
              <p className="research-card__value mono">{item.value}</p>
              <p>{item.detail}</p>
              <p className="research-card__metric muted">
                <strong>Metric:</strong> {item.metricDefinition}
              </p>
              <a href={item.sourceHref} target="_blank" rel="noopener noreferrer">
                {item.sourceLabel}
              </a>
            </li>
          ))}
        </ul>

        <p className="research-disclaimer" role="note">
          {researchDisclaimer}
        </p>

        <div className="hero__actions" style={{ marginTop: "1.5rem" }}>
          <Button href={withBasePath(links.cfrPage)}>Open the interactive CFR explainer</Button>
          <Button href={withBasePath(links.opponentModelingPage)} variant="secondary">
            Opponent modeling exhibit
          </Button>
          <Button href={links.cfr} variant="ghost" external>
            Technical CFR note
          </Button>
        </div>
      </div>
    </section>
  );
}
