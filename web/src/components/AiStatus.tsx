import type { SessionMetrics, StrategySummary } from "@/lib/types";

import { exploitability } from "./format";

const SOURCES: Record<StrategySummary["source"], string> = {
  baseline_cfr: "equilibrium baseline (CFR)",
  rnr: "adapted to you (RNR)",
  cfr_fallback: "equilibrium baseline (not confident enough to adapt)",
};

interface AiStatusProps {
  strategy: StrategySummary;
  metrics: SessionMetrics | null;
}

/** Read-only AI status. Verified numbers only; detailed model estimates are for M8C. */
export function AiStatus({ strategy, metrics }: AiStatusProps) {
  const updates = metrics?.strategy_updates;
  const learning = strategy.recomputation_running || updates?.running === true;
  return (
    <section className="panel" aria-labelledby="ai-title">
      <div className="panel-head">
        <h2 id="ai-title" className="panel-title">
          AI status
        </h2>
        <span className={`learning${learning ? " on" : ""}`} role="status" aria-live="polite">
          {learning ? "Learning…" : ""}
        </span>
      </div>
      <dl className="facts">
        <dt>Strategy version</dt>
        <dd data-testid="strategy-version">
          v{strategy.active_version}
          <span className="muted small"> {SOURCES[strategy.source]}</span>
        </dd>
        <dt>Hands observed</dt>
        <dd>{metrics ? metrics.verified.observation_counts.hands_observed : "…"}</dd>
        <dt>Certified exploitability</dt>
        <dd>{exploitability(strategy.certified_exploitability)}</dd>
        <dt>Exploitability limit</dt>
        <dd>{exploitability(strategy.exploitability_limit)}</dd>
      </dl>
      <p className="muted small">
        PokerMind keeps playing its current strategy and studies your play in the background.
        {updates
          ? ` It reconsiders its strategy every ${updates.recompute_every_hands} hands (next after hand ${updates.next_scheduled_after_hands}), or sooner if your style seems to change.`
          : ""}{" "}
        A new strategy is used only from the next hand, and only after its exploitability has
        been verified exactly against the limit.
      </p>
      {updates?.last_failure && (
        <p className="muted small">
          The last update did not complete ({updates.last_failure.kind}); the verified strategy
          stays in use.
        </p>
      )}
    </section>
  );
}
