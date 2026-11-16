import { describe, expect, it } from "vitest";
import {
  getM1Checkpoint,
  M1_CHECKPOINTS,
  M1_FOCUS_INFOSETS,
  M1_SOURCE,
  M1_TIMELINE_ITERATIONS,
} from "@/content/m1CfrCheckpoints";

describe("M1 CFR checkpoint dataset", () => {
  it("comes from the documented Kuhn vanilla CFR artifact", () => {
    expect(M1_SOURCE.artifact).toBe("runs/m1-cfr/metrics.json");
    expect(M1_SOURCE.exploitabilityConvention).toMatch(/NashConv \/ 2/);
  });

  it("includes the visitor-facing timeline iterations with exact values", () => {
    expect([...M1_TIMELINE_ITERATIONS]).toEqual([1, 10, 100, 1000, 10000, 50000]);

    expect(getM1Checkpoint(1).exploitability).toBeCloseTo(0.45833333333333326, 12);
    expect(getM1Checkpoint(10).exploitability).toBeCloseTo(0.09620850020140503, 12);
    expect(getM1Checkpoint(100).exploitability).toBeCloseTo(0.02567473584694787, 12);
    expect(getM1Checkpoint(1000).exploitability).toBeCloseTo(0.007269106408563809, 12);
    expect(getM1Checkpoint(10000).exploitability).toBeCloseTo(0.002317786314974535, 12);
    expect(getM1Checkpoint(50000).exploitability).toBeCloseTo(0.0009988288043084083, 12);
  });

  it("keeps exploitability decreasing across the full recorded curve", () => {
    for (let index = 1; index < M1_CHECKPOINTS.length; index += 1) {
      expect(M1_CHECKPOINTS[index].exploitability).toBeLessThan(
        M1_CHECKPOINTS[index - 1].exploitability,
      );
    }
  });

  it("stores selected average-strategy probabilities that sum to one", () => {
    for (const checkpoint of M1_CHECKPOINTS) {
      for (const info of M1_FOCUS_INFOSETS) {
        const probs = checkpoint.selectedAverageStrategy[info.key];
        const sum = info.actions.reduce((total, action) => total + probs[action.code], 0);
        expect(sum).toBeCloseTo(1, 10);
      }
    }
  });

  it("shows K facing a bet converging toward always call", () => {
    const late = getM1Checkpoint(50000).selectedAverageStrategy["K|b"];
    expect(late.c).toBeGreaterThan(0.999);
    expect(late.f).toBeLessThan(0.001);
  });
});
