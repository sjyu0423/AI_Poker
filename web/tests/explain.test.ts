import { describe, expect, it } from "vitest";

import { explainStrategy, explainTendency, isSparse } from "@/lib/explain";

import { tendencies, updates, version } from "./fake";

const byKind = (items: { kind: string; text: string }[], kind: string) =>
  items.filter((s) => s.kind === kind).map((s) => s.text);

describe("explainTendency", () => {
  it("says when a situation has not been seen and falls back to the prior", () => {
    const t = tendencies()[0]!;
    expect(isSparse(t)).toBe(true);
    const s = explainTendency(t);
    expect(byKind(s, "observation")).toEqual(["PokerMind hasn't seen you in this situation yet."]);
    expect(byKind(s, "estimate")[0]).toMatch(/typical \(population\) player/);
  });

  it("warns about sparse data and makes no tendency claims", () => {
    const t = tendencies({ "R1:fcr": { raise: 2 } })[1]!;
    const s = explainTendency(t);
    expect(s).toHaveLength(1);
    expect(s[0]!.text).toBe(
      "PokerMind has only seen this situation twice, so its estimate stays close to a typical player's.",
    );
  });

  it("reports notable differences from the prior as observations, with the estimate", () => {
    // raised 6 of 8 times; typical player raises 18%
    const t = tendencies({ "R1:fcr": { raise: 6, call: 2 } })[1]!;
    const s = explainTendency(t);
    const observed = byKind(s, "observation");
    expect(observed).toContain(
      "You've raised 6 of 8 times here (75%), more often than a typical player (18%).",
    );
    // only the notable "more often" action is described, not its mirror image
    expect(observed).toHaveLength(1);
    expect(byKind(s, "estimate").some((x) => /^Estimated raise frequency: \d+% \(90% interval/.test(x))).toBe(true);
    // never claims PokerMind changed its play
    expect(s.every((x) => !/adapt|strategy/i.test(x.text))).toBe(true);
  });

  it("falls back to the strongest 'less often' action when no increase is notable", () => {
    // typical fold/call/raise 38/44/18%; observed 20/55/25%: only fold differs by >= 15 points
    const t = tendencies({ "R1:fcr": { fold: 4, call: 11, raise: 5 } })[1]!;
    expect(byKind(explainTendency(t), "observation")).toEqual([
      "You've folded 4 of 20 times here (20%), less often than a typical player (38%).",
    ]);
  });

  it("says when play is close to typical", () => {
    const t = tendencies({ "R1:cr": { check: 6, bet: 4 } })[0]!;
    expect(explainTendency(t)).toEqual([
      { kind: "observation", text: "In 10 decisions here your play has been close to a typical player's." },
    ]);
  });
});

describe("explainStrategy", () => {
  it("does not claim adaptation while the baseline is in play", () => {
    const s = explainStrategy(version(), updates(), []);
    expect(byKind(s, "certified")[0]).toMatch(/baseline equilibrium strategy .* has not adapted/);
    expect(byKind(s, "status")).toEqual(["Its first review of your play is scheduled after hand 50."]);
  });

  it("describes a running update without claiming a new strategy", () => {
    const s = explainStrategy(
      version(),
      updates({ running: true, running_job: { job_id: 1, reason: "scheduled", based_on_hands: 50, elapsed_seconds: 1 } }),
      [],
    );
    expect(byKind(s, "status")[0]).toBe(
      "PokerMind is preparing an updated strategy from 50 hands. It keeps playing version 1 until the new one is verified.",
    );
    expect(s.some((x) => /passed the exploitability check/.test(x.text))).toBe(false);
  });

  it("claims adaptation only for an installed, certified RNR strategy", () => {
    const rnr = version({ version: 2, source: "rnr", rnr_p: 0.5, certified_exploitability: 0.0662, based_on_hands: 50, installed_after_hand: 50 });
    expect(byKind(explainStrategy(rnr, updates(), []), "certified")[0]).toBe(
      "Strategy v2 is adapted to your play (from 50 hands) and passed the exploitability check: 0.0662 chips/hand, within the limit of 0.1030 chips/hand.",
    );
    const kept = version({ version: 2, source: "cfr_fallback", based_on_hands: 50 });
    const text = byKind(explainStrategy(kept, updates(), []), "certified")[0]!;
    expect(text).toMatch(/not confident that adapting would pay off/);
    expect(text).not.toMatch(/adapted to your play/);
  });

  it("reports failures and change resets", () => {
    const s = explainStrategy(
      version(),
      updates({ last_failure: { job_id: 2, kind: "timeout", message: "x", based_on_hands: 9, at: 0 } }),
      [12],
    );
    expect(byKind(s, "status")).toContain("The last update did not complete (timeout), so the verified strategy stays in use.");
    expect(byKind(s, "estimate")[0]).toMatch(/flagged a shift in your play after hand 12/);
  });
});
