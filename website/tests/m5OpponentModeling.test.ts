import { describe, expect, it } from "vitest";
import {
  M5_EXCESS_LOG_LOSS,
  M5_PUBLIC_CEILING,
  M5_SOURCE,
  M5_SWITCH_ADAPTATION,
} from "@/content/m5OpponentModeling";

describe("M5 opponent-modeling dataset", () => {
  it("cites the held-out experiment artifact", () => {
    expect(M5_SOURCE.artifact).toBe("runs/m5-opponent-modeling/metrics.json");
    expect(M5_SOURCE.unit).toMatch(/nats/);
  });

  it("includes the required hand budgets with decreasing excess log loss", () => {
    expect(M5_EXCESS_LOG_LOSS.map((row) => row.hands)).toEqual([0, 20, 50, 100, 500]);
    expect(M5_EXCESS_LOG_LOSS[0].excessLogLoss).toBeCloseTo(0.123, 12);
    expect(M5_EXCESS_LOG_LOSS[1].excessLogLoss).toBeCloseTo(0.06, 12);
    expect(M5_EXCESS_LOG_LOSS[4].excessLogLoss).toBeCloseTo(0.014, 12);
    for (let i = 1; i < M5_EXCESS_LOG_LOSS.length; i += 1) {
      expect(M5_EXCESS_LOG_LOSS[i].excessLogLoss).toBeLessThan(
        M5_EXCESS_LOG_LOSS[i - 1].excessLogLoss,
      );
    }
  });

  it("records the public-information ceiling against a card-seeing oracle", () => {
    expect(M5_PUBLIC_CEILING.cardOracleRelativeToPublic).toBeCloseTo(-0.156, 12);
    expect(M5_PUBLIC_CEILING.publicOracleAccuracy).toBeCloseTo(0.691, 12);
  });

  it("keeps switch-adaptation rows ordered and favors CUSUM after 20 hands", () => {
    expect(M5_SWITCH_ADAPTATION[0].handsSinceSwitch).toBe(0);
    const at20 = M5_SWITCH_ADAPTATION.find((row) => row.handsSinceSwitch === 20);
    expect(at20?.cusumReset).toBeCloseTo(0.093, 12);
    expect(at20!.cusumReset).toBeLessThan(at20!.forgetting);
    expect(at20!.forgetting).toBeLessThan(at20!.cumulative);
  });
});
