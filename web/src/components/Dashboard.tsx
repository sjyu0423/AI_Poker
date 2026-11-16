"use client";

import {
  explainStrategy,
  explainTendency,
  isSparse,
  SPARSE_DECISIONS,
  type Statement,
  type StatementKind,
} from "@/lib/explain";
import type {
  SessionMetrics,
  StrategySource,
  StrategyStatus,
  StrategyVersion,
  Tendency,
  TendencyAction,
} from "@/lib/types";

import { percent } from "./format";

const KIND_LABEL: Record<StatementKind, string> = {
  observation: "Observed",
  estimate: "Estimate",
  certified: "Certified",
  status: "Status",
};

const SOURCE_LABEL: Record<StrategySource, string> = {
  baseline_cfr: "Base CFR strategy",
  rnr: "Adapted to you (RNR)",
  cfr_fallback: "Base CFR (reviewed, kept)",
};

const SITUATION_TITLES: Record<string, string> = {
  "R1:cr": "Round 1 · no bet to you",
  "R1:fcr": "Round 1 · facing a bet",
  "R1:fc": "Round 1 · facing a re-raise",
  "R2:cr": "Round 2 · no bet to you",
  "R2:fcr": "Round 2 · facing a bet",
  "R2:fc": "Round 2 · facing a re-raise",
};

const chips = (x: number) => `${x.toFixed(4)}`;

function Statements({ items }: { items: Statement[] }) {
  return (
    <ul className="statements">
      {items.map((s, i) => (
        <li key={i}>
          <span className={`kind kind-${s.kind}`}>{KIND_LABEL[s.kind]}</span> {s.text}
        </li>
      ))}
    </ul>
  );
}

function ActionRow({ label, a, n }: { label: string; a: TendencyAction; n: number }) {
  const [lo, hi] = a.interval_90;
  const seen = Math.round(a.observed);
  const summary =
    `${label}: estimated ${percent(a.posterior_mean)}, 90% interval ${percent(lo)} to ${percent(hi)}, ` +
    `typical player ${percent(a.population_prior_mean)}, seen ${seen} of ${Math.round(n)}`;
  return (
    <div className="trow" role="listitem" aria-label={summary}>
      <span className="trow-label">{label}</span>
      <div className="track" aria-hidden="true">
        <span
          className="track-interval"
          style={{ left: `${lo * 100}%`, width: `${Math.max(0.5, (hi - lo) * 100)}%` }}
        />
        <span className="track-mean" style={{ left: `${a.posterior_mean * 100}%` }} />
        <span className="track-prior" style={{ left: `${a.population_prior_mean * 100}%` }} />
      </div>
      <span className="trow-value" aria-hidden="true">
        {percent(a.posterior_mean)}
        <span className="muted small">
          {" "}
          {percent(lo)}–{percent(hi)}
        </span>
      </span>
      <span className="trow-seen muted small" aria-hidden="true">
        seen {seen}/{Math.round(n)}
      </span>
    </div>
  );
}

function TendencyCard({ t }: { t: Tendency }) {
  const n = t.observations;
  const sparse = isSparse(t);
  return (
    <article className="tcard" aria-labelledby={`t-${t.situation}`}>
      <header className="tcard-head">
        <h4 id={`t-${t.situation}`}>{SITUATION_TITLES[t.situation] ?? t.description}</h4>
        <span className={`badge${sparse ? " badge-warn" : ""}`}>
          {Math.round(n)} decision{Math.round(n) === 1 ? "" : "s"}
          {sparse ? " · sparse" : ""}
        </span>
      </header>
      <div role="list" aria-label={`Estimated action frequencies, ${t.description}`}>
        {Object.entries(t.actions).map(([label, a]) => (
          <ActionRow key={label} label={label} a={a} n={n} />
        ))}
      </div>
      <Statements items={explainTendency(t)} />
    </article>
  );
}

/** First hand a version played, or a note that a newer version replaced it before any
 * hand was dealt (both installed after the same hand). Derived from served values only. */
export function inPlayFrom(v: StrategyVersion, all: StrategyVersion[]): string {
  const replaced = all.some(
    (w) => w.version > v.version && w.installed_after_hand === v.installed_after_hand,
  );
  if (replaced) return "replaced before use";
  return String(v.version === 1 ? 1 : v.installed_after_hand + 1);
}

function ExploitabilityMeter({ v, baseline }: { v: StrategyVersion; baseline: number }) {
  const limit = v.exploitability_limit;
  const share = Math.min(1, v.certified_exploitability / limit);
  return (
    <div className="meter-wrap">
      <div
        className="meter"
        role="meter"
        aria-label="Certified exploitability against the limit"
        aria-valuemin={0}
        aria-valuemax={limit}
        aria-valuenow={v.certified_exploitability}
        aria-valuetext={`${chips(v.certified_exploitability)} of at most ${chips(limit)} chips per hand`}
      >
        <span className="meter-fill" style={{ width: `${share * 100}%` }} />
        <span className="meter-base" style={{ left: `${(baseline / limit) * 100}%` }} />
      </div>
      <div className="meter-scale muted small">
        <span>0</span>
        <span>limit {chips(limit)}</span>
      </div>
    </div>
  );
}

function StrategyEvolution({
  metrics,
  strategy,
}: {
  metrics: SessionMetrics;
  strategy: StrategyStatus;
}) {
  const active = strategy.active;
  const updates = strategy.updates;
  const baseline = metrics.verified.baseline_cfr_exploitability;
  const last = updates.last_success;
  const versions = [...strategy.versions].reverse();
  return (
    <section className="panel dash-section" aria-labelledby="evo-title">
      <h3 id="evo-title" className="panel-title">
        Strategy evolution <span className="kind kind-certified">Certified</span>
      </h3>
      <div className="evo-grid">
        <dl className="facts">
          <dt>Current version</dt>
          <dd data-testid="dash-version">v{active.version}</dd>
          <dt>Status</dt>
          <dd>{SOURCE_LABEL[active.source]}</dd>
          <dt>Certified exploitability</dt>
          <dd data-testid="dash-exploitability">{chips(active.certified_exploitability)} chips/hand</dd>
          <dt>Exploitability limit</dt>
          <dd>{chips(active.exploitability_limit)} chips/hand</dd>
          <dt>Base CFR exploitability</dt>
          <dd>{chips(baseline)} chips/hand</dd>
          <dt>Background update</dt>
          <dd data-testid="dash-update-status">
            {updates.running ? "Preparing…" : updates.pending ? "Queued" : "Idle"}
          </dd>
          <dt>Last successful update</dt>
          <dd>
            {last
              ? `v${last.version} from ${last.based_on_hands} hands (${last.seconds.toFixed(1)} s)`
              : "none yet"}
          </dd>
        </dl>
        <div>
          <ExploitabilityMeter v={active} baseline={baseline} />
          <p className="muted small">
            Exploitability is how much a perfect counter-strategy could win per hand. Every
            strategy PokerMind plays is checked exactly against the limit before use; the tick
            marks the base CFR strategy.
          </p>
        </div>
      </div>
      <Statements items={explainStrategy(active, updates, metrics.verified.observation_counts.change_resets_after_hands)} />
      <h4 className="sub-title">Versions this session</h4>
      <div className="table-scroll">
        <table className="history versions">
          <thead>
            <tr>
              <th scope="col">Version</th>
              <th scope="col">Strategy</th>
              <th scope="col">Exploitability</th>
              <th scope="col">From hands</th>
              <th scope="col">In play from hand</th>
            </tr>
          </thead>
          <tbody>
            {versions.map((v) => (
              <tr key={v.version} aria-current={v.version === active.version ? "true" : undefined}>
                <td>v{v.version}</td>
                <td>
                  {SOURCE_LABEL[v.source]}
                  {v.rnr_p !== null && <span className="muted small"> p={v.rnr_p}</span>}
                </td>
                <td className="mono">{chips(v.certified_exploitability)}</td>
                <td>{v.based_on_hands}</td>
                <td>{inPlayFrom(v, strategy.versions)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small">
        Only certified facts about each version are shown. PokerMind&apos;s own action
        probabilities stay private: showing them would reveal how to play against it.
      </p>
    </section>
  );
}

interface DashboardProps {
  metrics: SessionMetrics | null;
  strategy: StrategyStatus | null;
  pending: boolean;
  onRefresh: () => void;
}

export function Dashboard({ metrics, strategy, pending, onRefresh }: DashboardProps) {
  if (!metrics || !strategy) {
    return (
      <div className="empty" role="status">
        Loading what PokerMind has learned…
      </div>
    );
  }
  const counts = metrics.verified.observation_counts;
  const tendencies = metrics.estimates.opponent_tendencies;
  const seenAny = tendencies.some((t) => t.observations > 0);
  return (
    <div className="dashboard">
      <div className="dash-top">
        <p className="muted">
          How PokerMind sees your play, and how its verified strategy has changed this session.
        </p>
        <button type="button" className="btn btn-ghost btn-small" onClick={onRefresh} disabled={pending}>
          Refresh
        </button>
      </div>

      <section className="panel dash-section" aria-labelledby="obs-title">
        <h3 id="obs-title" className="panel-title">
          What PokerMind has observed <span className="kind kind-observation">Observed</span>
        </h3>
        <dl className="counts">
          <div>
            <dt>Hands observed</dt>
            <dd data-testid="dash-hands">{counts.hands_observed}</dd>
          </div>
          <div>
            <dt>Hands in the current model</dt>
            <dd>{counts.hands_in_model}</dd>
          </div>
          <div>
            <dt>Your decisions in the model</dt>
            <dd>{counts.human_decisions_in_model}</dd>
          </div>
          <div>
            <dt>Your cards seen at showdown</dt>
            <dd>{counts.showdowns_in_model}</dd>
          </div>
        </dl>
        <p className="muted small">
          PokerMind sees only what a player at the table sees: your actions, and your card when
          it is shown at a showdown, never a folded card.
        </p>
      </section>

      <section className="panel dash-section" aria-labelledby="tend-title">
        <h3 id="tend-title" className="panel-title">
          Your estimated tendencies <span className="kind kind-estimate">Estimate</span>
        </h3>
        <p className="disclaimer" role="note">
          These are probabilistic estimates, not facts about you. The 90% intervals were checked
          only against simulated players and have not been validated for real human opponents.
        </p>
        <div className="legend muted small" aria-hidden="true">
          <span>
            <span className="lg lg-mean" /> estimate
          </span>
          <span>
            <span className="lg lg-interval" /> 90% interval
          </span>
          <span>
            <span className="lg lg-prior" /> typical player (prior)
          </span>
        </div>
        <p className="muted small">
          Each estimate starts at a typical player&apos;s frequency and moves toward yours as
          PokerMind sees more of your decisions. The prior counts as{" "}
          {Math.round(tendencies[0]?.prior_strength ?? 0)} decisions, so intervals look narrow
          even with little data: situations with fewer than {SPARSE_DECISIONS} of your decisions
          are marked sparse.
        </p>
        {!seenAny && (
          <p className="empty-inline">
            No decisions observed yet. Play a few hands and your tendencies will appear here.
          </p>
        )}
        <div className="tgrid">
          {tendencies.map((t) => (
            <TendencyCard key={t.situation} t={t} />
          ))}
        </div>
      </section>

      <StrategyEvolution metrics={metrics} strategy={strategy} />
    </div>
  );
}
