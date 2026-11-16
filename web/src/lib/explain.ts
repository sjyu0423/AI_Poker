/**
 * Plain-language explanations generated only from values the server returned.
 *
 * Every statement says what kind of claim it is:
 * - observation: counted from the player's own decisions;
 * - estimate: the Bayesian model's inference (probabilistic, not validated for humans);
 * - certified: an exact, verified property of the strategy being played;
 * - status: what the background process is doing.
 *
 * Adaptation is claimed only when the server reports an installed, certified RNR strategy;
 * a changed frequency estimate alone never implies the AI changed its play.
 */

import type { SessionMetrics, StrategyVersion, Tendency, UpdateStatus } from "./types";

export type StatementKind = "observation" | "estimate" | "certified" | "status";

export interface Statement {
  kind: StatementKind;
  text: string;
}

/** Fewer decisions than this in a situation: estimates there are mostly the prior. */
export const SPARSE_DECISIONS = 5;
/** Observed frequency this far from the population prior is worth pointing out. */
export const NOTABLE_DIFFERENCE = 0.15;

const PAST: Record<string, string> = {
  check: "checked",
  bet: "bet",
  fold: "folded",
  call: "called",
  raise: "raised",
};

const pct = (x: number) => `${Math.round(x * 100)}%`;
const times = (n: number) => (n === 1 ? "once" : n === 2 ? "twice" : `${n} times`);
const chips = (x: number) => `${x.toFixed(4)} chips/hand`;

export function isSparse(t: Tendency): boolean {
  return t.observations < SPARSE_DECISIONS;
}

/** Statements about one betting situation. */
export function explainTendency(t: Tendency): Statement[] {
  const n = Math.round(t.observations);
  if (n === 0) {
    return [
      { kind: "observation", text: "PokerMind hasn't seen you in this situation yet." },
      {
        kind: "estimate",
        text: "Until it does, its estimate is the typical (population) player's frequencies.",
      },
    ];
  }
  const out: Statement[] = [];
  if (n < SPARSE_DECISIONS) {
    out.push({
      kind: "observation",
      text: `PokerMind has only seen this situation ${times(n)}, so its estimate stays close to a typical player's.`,
    });
  }
  // Notable actions: those played more often than typical; if there are none, the one
  // played most notably less often. (One statement per situation keeps the panel short.)
  const notable = Object.entries(t.actions)
    .filter(([, a]) => a.observed_frequency !== null && n >= SPARSE_DECISIONS)
    .map(([label, a]) => ({ label, a, diff: (a.observed_frequency ?? 0) - a.population_prior_mean }))
    .filter((x) => Math.abs(x.diff) >= NOTABLE_DIFFERENCE);
  const more = notable.filter((x) => x.diff > 0);
  const shown = more.length > 0 ? more : notable.sort((x, y) => x.diff - y.diff).slice(0, 1);
  for (const { label, a, diff } of shown) {
    if (a.observed_frequency === null) continue;
    out.push({
      kind: "observation",
      text: `You've ${PAST[label] ?? label} ${Math.round(a.observed)} of ${n} times here (${pct(a.observed_frequency)}), ${diff > 0 ? "more" : "less"} often than a typical player (${pct(a.population_prior_mean)}).`,
    });
    out.push({
      kind: "estimate",
      text: `Estimated ${label} frequency: ${pct(a.posterior_mean)} (90% interval ${pct(a.interval_90[0])}–${pct(a.interval_90[1])}).`,
    });
  }
  if (out.length === 0) {
    out.push({
      kind: "observation",
      text: `In ${n} decisions here your play has been close to a typical player's.`,
    });
  }
  return out;
}

/** Statements about the strategy and the background process. */
export function explainStrategy(
  active: StrategyVersion,
  updates: UpdateStatus,
  resets: number[],
): Statement[] {
  const out: Statement[] = [];
  if (updates.running) {
    const based = updates.running_job?.based_on_hands;
    out.push({
      kind: "status",
      text: `PokerMind is preparing an updated strategy${based !== undefined ? ` from ${based} hands` : ""}. It keeps playing version ${active.version} until the new one is verified.`,
    });
  } else if (updates.pending) {
    out.push({ kind: "status", text: "An updated strategy is queued and will start shortly." });
  }
  if (active.source === "rnr") {
    out.push({
      kind: "certified",
      text: `Strategy v${active.version} is adapted to your play (from ${active.based_on_hands} hands) and passed the exploitability check: ${chips(active.certified_exploitability)}, within the limit of ${chips(active.exploitability_limit)}.`,
    });
  } else if (active.source === "cfr_fallback") {
    out.push({
      kind: "certified",
      text: `After reviewing ${active.based_on_hands} hands, PokerMind was not confident that adapting would pay off, so v${active.version} is its verified baseline strategy (${chips(active.certified_exploitability)}).`,
    });
  } else {
    out.push({
      kind: "certified",
      text: `PokerMind is playing its baseline equilibrium strategy (${chips(active.certified_exploitability)}). It has not adapted to you yet.`,
    });
    if (!updates.running && !updates.pending) {
      out.push({
        kind: "status",
        text: `Its first review of your play is scheduled after hand ${updates.next_scheduled_after_hands}.`,
      });
    }
  }
  if (updates.last_failure) {
    out.push({
      kind: "status",
      text: `The last update did not complete (${updates.last_failure.kind}), so the verified strategy stays in use.`,
    });
  }
  if (resets.length > 0) {
    out.push({
      kind: "estimate",
      text: `PokerMind's change detector flagged a shift in your play after hand ${resets.join(", ")} and restarted its estimates from your most recent hands.`,
    });
  }
  return out;
}

export function sparseCount(m: SessionMetrics): number {
  return m.estimates.opponent_tendencies.filter(isSparse).length;
}
