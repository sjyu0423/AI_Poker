/**
 * Held-out M5 opponent-modeling results used by the W3 exhibit.
 *
 * Source: docs/experiments/m5_opponent_modeling.md
 * Artifact: runs/m5-opponent-modeling/metrics.json
 * Reproduce: pokermind opponent-study --hero runs/m4-leduc-teacher/strategies/strategy_100000.json
 *
 * Excess log loss is nats per opponent decision above the best public-
 * information predictor—not a classification accuracy percentage.
 */

export const M5_SOURCE = {
  docs: "docs/experiments/m5_opponent_modeling.md",
  design: "docs/design/opponent_modeling.md",
  m6Design: "docs/design/adaptive_strategy.md",
  artifact: "runs/m5-opponent-modeling/metrics.json",
  command:
    "pokermind opponent-study --hero runs/m4-leduc-teacher/strategies/strategy_100000.json",
  unit: "nats per opponent decision",
} as const;

export interface ExcessLogLossRow {
  readonly hands: number;
  readonly excessLogLoss: number;
  readonly ciLow: number;
  readonly ciHigh: number;
  /** Top-1 action accuracy on future hands (reported separately from excess log loss). */
  readonly accuracy: number;
  /** Share of the learnable public gap closed vs the prior (when defined). */
  readonly learnableGapClosed: number | null;
}

/** Held-out stationary opponents, selected hierarchical model. */
export const M5_EXCESS_LOG_LOSS: readonly ExcessLogLossRow[] = [
  {
    hands: 0,
    excessLogLoss: 0.123,
    ciLow: 0.106,
    ciHigh: 0.14,
    accuracy: 0.6,
    learnableGapClosed: 0,
  },
  {
    hands: 20,
    excessLogLoss: 0.06,
    ciLow: 0.056,
    ciHigh: 0.064,
    accuracy: 0.657,
    learnableGapClosed: 0.51,
  },
  {
    hands: 50,
    excessLogLoss: 0.039,
    ciLow: 0.035,
    ciHigh: 0.044,
    accuracy: 0.669,
    learnableGapClosed: 0.68,
  },
  {
    hands: 100,
    excessLogLoss: 0.03,
    ciLow: 0.026,
    ciHigh: 0.033,
    accuracy: 0.676,
    learnableGapClosed: 0.76,
  },
  {
    hands: 500,
    excessLogLoss: 0.014,
    ciLow: 0.012,
    ciHigh: 0.016,
    accuracy: 0.683,
    learnableGapClosed: 0.88,
  },
] as const;

export const M5_PUBLIC_CEILING = {
  /** Excess log loss of the best public-information predictor (by definition 0). */
  publicOracleExcess: 0,
  /** Best possible top-1 accuracy from public actions alone. */
  publicOracleAccuracy: 0.691,
  /**
   * How much worse the public oracle is than an oracle that also sees the
   * opponent's private card (nats/decision). Negative excess = better than public oracle.
   */
  cardOracleRelativeToPublic: -0.156,
  note:
    "Even a perfect public model is 0.156 nats per decision worse than one that sees the opponent's card—larger than the entire 0→500 learning curve.",
} as const;

export const M5_CALIBRATION = {
  expectedCalibrationErrorAt20: 0.016,
  nominal90CoverageAt20: 0.74,
  nominal90CoverageAt0: 0.57,
  nominal90CoverageAt500: 0.87,
  note:
    "Action probabilities were well calibrated on simulated opponents, but nominal 90% credible intervals were too narrow with little data (74% coverage after 20 hands). Intervals were not validated for real humans.",
} as const;

export interface SwitchAdaptationRow {
  readonly handsSinceSwitch: number;
  readonly cumulative: number;
  readonly forgetting: number;
  readonly cusumReset: number;
}

/** Excess log loss after a style switch (held-out switching opponents). */
export const M5_SWITCH_ADAPTATION: readonly SwitchAdaptationRow[] = [
  { handsSinceSwitch: 0, cumulative: 0.315, forgetting: 0.258, cusumReset: 0.311 },
  { handsSinceSwitch: 10, cumulative: 0.283, forgetting: 0.187, cusumReset: 0.183 },
  { handsSinceSwitch: 20, cumulative: 0.255, forgetting: 0.14, cusumReset: 0.093 },
  { handsSinceSwitch: 50, cumulative: 0.197, forgetting: 0.068, cusumReset: 0.036 },
  { handsSinceSwitch: 100, cumulative: 0.142, forgetting: 0.035, cusumReset: 0.026 },
  { handsSinceSwitch: 200, cumulative: 0.088, forgetting: 0.023, cusumReset: 0.021 },
] as const;

export const M5_DETECTOR = {
  forgettingGamma: 0.98,
  cusumShortMemory: 0.8,
  cusumThreshold: 5,
  switchesDetected: "235 / 240",
  detectionDelayMedianHands: 11,
  detectionDelayMeanHands: 15.2,
  detectionDelayCi: [13.7, 16.7] as const,
  falseAlarmsBeforeSwitchPerThousandHands: 0.4,
  stationaryFalseAlarmShareOfRuns: 0.4,
  stationaryExcessAt500: {
    cumulative: 0.014,
    cusum: 0.02,
    forgetting: 0.031,
  },
} as const;

/** Illustrative public-frequency collision used in the hidden-card section. */
export const HIDDEN_CARD_EXAMPLE = {
  publicBetRate: 0.5,
  alwaysMix: {
    name: "Card-blind bettor",
    detail: "Bets about half the time with every private card.",
    jBet: 0.5,
    qBet: 0.5,
    kBet: 0.5,
  },
  polarized: {
    name: "Strength-seeking bettor",
    detail: "Almost never bets weak cards; almost always bets strong ones.",
    jBet: 0.05,
    qBet: 0.35,
    kBet: 0.95,
  },
  note:
    "Both can produce a ~50% public betting frequency while needing opposite replies. M6 therefore reconstructs card-conditional policies from the full hand likelihood, treating folded cards as latent.",
} as const;
