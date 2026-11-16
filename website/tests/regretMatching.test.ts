import { describe, expect, it } from "vitest";
import { positivePart, regretMatching } from "@/lib/regretMatching";

describe("regretMatching", () => {
  it("matches positive regrets proportionally", () => {
    const result = regretMatching([2, 1]);
    expect(result.usedUniformFallback).toBe(false);
    expect(result.positiveSum).toBe(3);
    expect(result.probabilities[0]).toBeCloseTo(2 / 3, 12);
    expect(result.probabilities[1]).toBeCloseTo(1 / 3, 12);
  });

  it("gives equal probabilities for equal positive regrets", () => {
    const result = regretMatching([4, 4]);
    expect(result.probabilities).toEqual([0.5, 0.5]);
  });

  it("clips negative regrets before normalizing", () => {
    const result = regretMatching([3, -2]);
    expect(result.positive).toEqual([3, 0]);
    expect(result.probabilities).toEqual([1, 0]);
  });

  it("falls back to uniform when all regrets are zero", () => {
    const result = regretMatching([0, 0]);
    expect(result.usedUniformFallback).toBe(true);
    expect(result.probabilities).toEqual([0.5, 0.5]);
  });

  it("falls back to uniform when all regrets are negative", () => {
    const result = regretMatching([-1.5, -0.25]);
    expect(result.usedUniformFallback).toBe(true);
    expect(result.positiveSum).toBe(0);
    expect(result.probabilities).toEqual([0.5, 0.5]);
  });

  it("supports more than two actions", () => {
    const result = regretMatching([1, 0, 3]);
    expect(result.probabilities[0]).toBeCloseTo(0.25, 12);
    expect(result.probabilities[1]).toBeCloseTo(0, 12);
    expect(result.probabilities[2]).toBeCloseTo(0.75, 12);
  });

  it("rejects empty or non-finite input", () => {
    expect(() => regretMatching([])).toThrow(/non-empty/);
    expect(() => regretMatching([1, Number.NaN])).toThrow(/finite/);
  });

  it("exposes a positive-part helper consistent with matching", () => {
    expect(positivePart(-3)).toBe(0);
    expect(positivePart(2.5)).toBe(2.5);
  });
});
