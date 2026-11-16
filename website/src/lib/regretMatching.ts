/**
 * Regret matching as in PokerMind's vanilla CFR
 * (`src/pokermind/learning/cfr.py` → `regret_matching`).
 *
 * σ(a) = R⁺(a) / Σ_b R⁺(b) when the positive sum is > 0,
 * otherwise uniform over the action set.
 */

export type RegretVector = readonly number[];

export interface RegretMatchingResult {
  /** Non-negative parts of the input regrets. */
  positive: number[];
  /** Sum of positive regrets. */
  positiveSum: number;
  /** Resulting behavioural strategy (sums to 1). */
  probabilities: number[];
  /** True when the uniform fallback was used. */
  usedUniformFallback: boolean;
}

export function positivePart(value: number): number {
  return Math.max(value, 0);
}

export function regretMatching(regrets: RegretVector): RegretMatchingResult {
  if (regrets.length === 0) {
    throw new Error("regretMatching requires a non-empty regret vector");
  }
  if (!regrets.every((value) => Number.isFinite(value))) {
    throw new Error("regretMatching requires finite regrets");
  }

  const positive = regrets.map(positivePart);
  const positiveSum = positive.reduce((sum, value) => sum + value, 0);

  if (positiveSum > 0) {
    return {
      positive,
      positiveSum,
      probabilities: positive.map((value) => value / positiveSum),
      usedUniformFallback: false,
    };
  }

  const uniform = 1 / regrets.length;
  return {
    positive,
    positiveSum: 0,
    probabilities: regrets.map(() => uniform),
    usedUniformFallback: true,
  };
}

/** Format a probability as a percentage string for UI display. */
export function formatProbability(probability: number, digits = 1): string {
  return `${(probability * 100).toFixed(digits)}%`;
}
