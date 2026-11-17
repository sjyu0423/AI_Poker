import { describe, expect, it } from "vitest";
import {
  betaBinomialPosterior,
  betaCdf,
  betaQuantile,
  centralCredibleInterval,
  posteriorParams,
  priorParams,
} from "@/lib/betaBinomial";

describe("Beta–Binomial educational learner", () => {
  it("builds prior pseudo-counts from mean and strength", () => {
    expect(priorParams(0.25, 8)).toEqual({ alpha: 2, beta: 6 });
  });

  it("returns the prior when there are zero observations", () => {
    const posterior = betaBinomialPosterior({
      priorMean: 0.4,
      priorStrength: 10,
      successes: 0,
      trials: 0,
    });
    expect(posterior.observedFrequency).toBeNull();
    expect(posterior.mean).toBeCloseTo(0.4, 12);
    expect(posterior.alpha).toBeCloseTo(4, 12);
    expect(posterior.beta).toBeCloseTo(6, 12);
  });

  it("updates with successes and failures", () => {
    const params = posteriorParams({
      priorMean: 0.5,
      priorStrength: 2,
      successes: 3,
      trials: 4,
    });
    // prior α=1, β=1; after 3/4 → α=4, β=2
    expect(params).toEqual({ alpha: 4, beta: 2 });
    const posterior = betaBinomialPosterior({
      priorMean: 0.5,
      priorStrength: 2,
      successes: 3,
      trials: 4,
    });
    expect(posterior.mean).toBeCloseTo(4 / 6, 12);
    expect(posterior.observedFrequency).toBeCloseTo(0.75, 12);
  });

  it("lets a stronger prior pull the mean toward μ", () => {
    const weak = betaBinomialPosterior({
      priorMean: 0.4,
      priorStrength: 2,
      successes: 3,
      trials: 3,
    });
    const strong = betaBinomialPosterior({
      priorMean: 0.4,
      priorStrength: 20,
      successes: 3,
      trials: 3,
    });
    expect(weak.mean).toBeGreaterThan(strong.mean);
    expect(strong.mean).toBeGreaterThan(0.4);
    expect(weak.observedFrequency).toBe(1);
  });

  it("handles all-failure extreme observations", () => {
    const posterior = betaBinomialPosterior({
      priorMean: 0.5,
      priorStrength: 2,
      successes: 0,
      trials: 20,
    });
    expect(posterior.observedFrequency).toBe(0);
    expect(posterior.mean).toBeCloseTo(1 / 22, 12);
  });

  it("computes a central 90% credible interval from Beta quantiles", () => {
    const interval = centralCredibleInterval(8, 2, 0.9);
    expect(interval.lower).toBeLessThan(interval.upper);
    expect(betaCdf(interval.lower, 8, 2)).toBeCloseTo(0.05, 3);
    expect(betaCdf(interval.upper, 8, 2)).toBeCloseTo(0.95, 3);
    expect(betaQuantile(0.5, 5, 5)).toBeCloseTo(0.5, 5);
  });

  it("narrows the interval as observations accumulate at the same frequency", () => {
    const sparse = betaBinomialPosterior({
      priorMean: 0.5,
      priorStrength: 2,
      successes: 3,
      trials: 6,
    });
    const plenty = betaBinomialPosterior({
      priorMean: 0.5,
      priorStrength: 2,
      successes: 50,
      trials: 100,
    });
    const sparseWidth = sparse.interval90.upper - sparse.interval90.lower;
    const plentyWidth = plenty.interval90.upper - plenty.interval90.lower;
    expect(plentyWidth).toBeLessThan(sparseWidth);
  });
});
