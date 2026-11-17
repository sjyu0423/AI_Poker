/**
 * Curated M7A / M7B robust-response results for the W4 exhibit.
 *
 * Sources of truth (do not invent or interpolate values here):
 * - docs/design/robust_response.md
 * - docs/experiments/m7a_robust_response.md
 * - docs/experiments/m7b_adaptive_reliability.md
 * - docs/research/research_freeze.md
 * - runs/m7a-robust-response/metrics.json
 * - runs/m7b-adaptive-reliability/evaluation.json
 *
 * Gain units: chips per hand above the fixed CFR hero against the true
 * opponent (seats alternate). Budgets are exact exploitability caps
 * expl(CFR) + δ with expl(CFR) = 0.0030.
 */

export const M7_SOURCE = {
  design: "docs/design/robust_response.md",
  m7aDocs: "docs/experiments/m7a_robust_response.md",
  m7bDocs: "docs/experiments/m7b_adaptive_reliability.md",
  freeze: "docs/research/research_freeze.md",
  m7aArtifact: "runs/m7a-robust-response/metrics.json",
  m7bArtifact: "runs/m7b-adaptive-reliability/evaluation.json",
  m7aCommand:
    "pokermind robust-study --hero runs/m4-leduc-teacher/strategies/strategy_100000.json",
  unit: "chips per hand",
  cfrExploitability: 0.003,
} as const;

/** Documented exploitability budgets used in the interactive frontier. */
export type ExploitabilityBudgetDelta = 0.02 | 0.1 | 0.25;

export interface FrontierMethodGains {
  readonly mix: number;
  readonly rnr: number;
  readonly dbr: number;
}

export interface FrontierBudgetRow {
  readonly delta: ExploitabilityBudgetDelta;
  /** Absolute exploitability cap = 0.0030 + δ. */
  readonly budgetCap: number;
  readonly label: string;
  /** Held-out opponents, after 100 observed hands. */
  readonly heldOut100: FrontierMethodGains;
  /** Out-of-population opponents, after 100 observed hands. */
  readonly outOfPopulation100: FrontierMethodGains;
}

/**
 * Rounded means matching the M7A results tables (verified against metrics.json).
 * Interactive controls only select among these recorded rows—no live RNR solve.
 */
export const M7A_FRONTIER_AT_100: readonly FrontierBudgetRow[] = [
  {
    delta: 0.02,
    budgetCap: 0.023,
    label: "CFR + 0.02",
    heldOut100: { mix: 0.004, rnr: 0.087, dbr: 0.028 },
    outOfPopulation100: { mix: 0.02, rnr: 0.281, dbr: 0.101 },
  },
  {
    delta: 0.1,
    budgetCap: 0.103,
    label: "CFR + 0.10",
    heldOut100: { mix: 0.016, rnr: 0.138, dbr: 0.063 },
    outOfPopulation100: { mix: 0.082, rnr: 0.619, dbr: 0.374 },
  },
  {
    delta: 0.25,
    budgetCap: 0.253,
    label: "CFR + 0.25",
    heldOut100: { mix: 0.037, rnr: 0.186, dbr: 0.094 },
    outOfPopulation100: { mix: 0.19, rnr: 0.775, dbr: 0.579 },
  },
] as const;

export function frontierRowForDelta(
  delta: ExploitabilityBudgetDelta,
): FrontierBudgetRow {
  const row = M7A_FRONTIER_AT_100.find((item) => item.delta === delta);
  if (!row) {
    throw new Error(`No recorded frontier row for δ = ${delta}`);
  }
  return row;
}

/** RNR (point rule) held-out gain over CFR at δ = 0.10 by observation count. */
export const M7A_RNR_HELD_OUT_BY_HANDS_D01: readonly {
  readonly hands: number;
  readonly rnr: number;
  readonly mix: number;
}[] = [
  { hands: 0, rnr: 0.124, mix: 0.017 },
  { hands: 20, rnr: 0.121, mix: 0.015 },
  { hands: 50, rnr: 0.129, mix: 0.016 },
  { hands: 100, rnr: 0.138, mix: 0.016 },
  { hands: 200, rnr: 0.143, mix: 0.016 },
  { hands: 500, rnr: 0.164, mix: 0.019 },
] as const;

/** Personal increment = gain(n) − gain(0), RNR held-out, δ = 0.10. */
export const M7A_PERSONAL_INCREMENT_HELD_OUT_D01: readonly {
  readonly hands: number;
  readonly mean: number;
  readonly ciLow: number;
  readonly ciHigh: number;
}[] = [
  { hands: 20, mean: -0.003, ciLow: -0.01, ciHigh: 0.004 },
  { hands: 100, mean: 0.015, ciLow: 0.003, ciHigh: 0.027 },
  { hands: 500, mean: 0.041, ciLow: 0.026, ciHigh: 0.055 },
] as const;

/** Out-of-population personal increment for RNR at δ = 0.10. */
export const M7A_PERSONAL_INCREMENT_OOP_D01: readonly {
  readonly hands: number;
  readonly mean: number;
  readonly ciLow: number;
  readonly ciHigh: number;
}[] = [
  { hands: 100, mean: 0.195, ciLow: 0.062, ciHigh: 0.329 },
  { hands: 500, mean: 0.263, ciLow: 0.116, ciHigh: 0.41 },
] as const;

/**
 * M7B frozen selection rule vs M7A point rule vs mixing, held-out, δ = 0.10,
 * after 100 hands. Harm rates pool opponent × seed × hands runs with gain < 0.
 */
export const M7B_RELIABILITY_HELD_OUT_D01 = {
  hands: 100,
  delta: 0.1 as const,
  pointGain: 0.138,
  frozenGain: 0.131,
  mixGain: 0.016,
  pointHarmRate: 0.067,
  frozenHarmRate: 0.023,
  frozenMinusMix: { mean: 0.115, ciLow: 0.075, ciHigh: 0.155 },
  frozenMinusPoint: { mean: -0.007, ciLow: -0.012, ciHigh: -0.003 },
  zeroHand: {
    point: 0.124,
    frozen: 0.099,
    mix: 0.017,
  },
  at500: {
    point: 0.164,
    frozen: 0.162,
    mix: 0.019,
  },
  note:
    "Measured experimental outcomes on simulated Leduc opponents—not universal guarantees or human-player results.",
} as const;

export const M7B_OOP_AT_100_D01 = {
  point: 0.619,
  frozen: 0.619,
  mix: 0.082,
} as const;

/** Conceptual anchors for the “naive exploitation” section (labeled). */
export const NAIVE_EXPLOITATION_ANCHORS = {
  cfr: {
    label: "CFR equilibrium",
    exploitability: 0.003,
    gainNote: "Baseline (gain defined as 0)",
    kind: "exact" as const,
  },
  mixAtBudget: {
    label: "M6 mixing at δ = 0.10",
    exploitabilityCap: 0.103,
    gain: 0.016,
    kind: "measured" as const,
    context: "Held-out, 100 hands",
  },
  rnrAtBudget: {
    label: "RNR at δ = 0.10",
    exploitabilityCap: 0.103,
    gain: 0.138,
    kind: "measured" as const,
    context: "Held-out, 100 hands (M7A point rule)",
  },
  unrestrictedBr: {
    label: "Unrestricted best response to σ̂",
    exploitabilityRange: "1–4 chips/hand",
    kind: "measured" as const,
    context:
      "M6 reported BR exploitability of roughly 1–4 chips/hand—far above any certified budget.",
  },
} as const;

export const RNR_P_CONCEPT = {
  definition:
    "p is the probability that the opponent is forced to follow the estimated model σ̂. With probability 1 − p the opponent may play any strategy. Hero never observes which mode occurred.",
  notBudget:
    "p is not the exploitability budget. PokerMind sweeps a grid of p values, measures each candidate’s exact exploitability, and keeps only those within expl(CFR) + δ.",
  extremes:
    "At p = 0 the construction recovers a Nash equilibrium; at p = 1 it recovers a best response to σ̂ (for an exact solve).",
} as const;
