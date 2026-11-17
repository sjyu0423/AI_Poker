import { describe, expect, it } from "vitest";
import {
  M7A_FRONTIER_AT_100,
  M7A_PERSONAL_INCREMENT_HELD_OUT_D01,
  M7A_RNR_HELD_OUT_BY_HANDS_D01,
  M7B_RELIABILITY_HELD_OUT_D01,
  M7_SOURCE,
  NAIVE_EXPLOITATION_ANCHORS,
  RNR_P_CONCEPT,
  frontierRowForDelta,
} from "@/content/m7RobustResponse";

describe("M7 robust-response dataset integrity", () => {
  it("anchors provenance and units", () => {
    expect(M7_SOURCE.unit).toBe("chips per hand");
    expect(M7_SOURCE.cfrExploitability).toBe(0.003);
    expect(M7_SOURCE.m7aDocs).toContain("m7a_robust_response");
    expect(M7_SOURCE.m7bDocs).toContain("m7b_adaptive_reliability");
  });

  it("matches documented M7A frontier gains at 100 hands", () => {
    const byDelta = Object.fromEntries(
      M7A_FRONTIER_AT_100.map((row) => [row.delta, row]),
    );

    expect(byDelta[0.02].heldOut100).toEqual({
      mix: 0.004,
      rnr: 0.087,
      dbr: 0.028,
    });
    expect(byDelta[0.1].heldOut100).toEqual({
      mix: 0.016,
      rnr: 0.138,
      dbr: 0.063,
    });
    expect(byDelta[0.25].heldOut100).toEqual({
      mix: 0.037,
      rnr: 0.186,
      dbr: 0.094,
    });

    expect(byDelta[0.02].budgetCap).toBeCloseTo(0.023, 5);
    expect(byDelta[0.1].budgetCap).toBeCloseTo(0.103, 5);
    expect(byDelta[0.25].budgetCap).toBeCloseTo(0.253, 5);
  });

  it("looks up only recorded budgets", () => {
    expect(frontierRowForDelta(0.1).label).toBe("CFR + 0.10");
    expect(M7A_FRONTIER_AT_100.map((row) => row.delta)).toEqual([0.02, 0.1, 0.25]);
  });

  it("keeps personalization rows consistent with the RNR time series", () => {
    const at0 = M7A_RNR_HELD_OUT_BY_HANDS_D01.find((row) => row.hands === 0)!;
    const at100 = M7A_RNR_HELD_OUT_BY_HANDS_D01.find((row) => row.hands === 100)!;
    const increment100 = M7A_PERSONAL_INCREMENT_HELD_OUT_D01.find(
      (row) => row.hands === 100,
    )!;

    expect(at100.rnr).toBe(0.138);
    expect(at0.rnr).toBe(0.124);
    // Documented paired increment (0.015), not the difference of rounded means.
    expect(increment100.mean).toBe(0.015);
    expect(Math.abs(increment100.mean - (at100.rnr - at0.rnr))).toBeLessThan(0.002);
  });

  it("records M7B reliability headline numbers", () => {
    expect(M7B_RELIABILITY_HELD_OUT_D01.pointGain).toBe(0.138);
    expect(M7B_RELIABILITY_HELD_OUT_D01.frozenGain).toBe(0.131);
    expect(M7B_RELIABILITY_HELD_OUT_D01.mixGain).toBe(0.016);
    expect(M7B_RELIABILITY_HELD_OUT_D01.pointHarmRate).toBeCloseTo(0.067, 5);
    expect(M7B_RELIABILITY_HELD_OUT_D01.frozenHarmRate).toBeCloseTo(0.023, 5);
  });

  it("distinguishes p from the exploitability budget", () => {
    expect(RNR_P_CONCEPT.notBudget.toLowerCase()).toMatch(/not the exploitability budget/);
    expect(NAIVE_EXPLOITATION_ANCHORS.unrestrictedBr.exploitabilityRange).toMatch(/1–4/);
    expect(NAIVE_EXPLOITATION_ANCHORS.rnrAtBudget.gain).toBe(0.138);
  });
});
