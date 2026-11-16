import { describe, expect, it } from "vitest";
import {
  differentiators,
  finalCta,
  guarantees,
  hero,
  limitations,
  links,
  navItems,
  pipelineSteps,
  researchDisclaimer,
  researchHighlights,
  screenshots,
  siteMeta,
  technology,
} from "@/content/site";

describe("portfolio content source of truth", () => {
  it("keeps the required homepage messaging", () => {
    expect(hero.brand).toBe("PokerMind");
    expect(hero.headline).toBe("An AI That Learns How You Play.");
    expect(hero.summary).toMatch(/Bayesian inference and game theory/i);
    expect(hero.primaryCta.label).toBe("Explore the Research");
    expect(hero.secondaryCta.label).toBe("View on GitHub");
    expect(finalCta.headline).toMatch(/Challenge PokerMind locally/i);
  });

  it("anchors research metrics to documented values", () => {
    expect(guarantees.baselineExploitability).toBe("0.0030 chips/hand");
    expect(guarantees.budgetLimit).toBe("0.1030 chips/hand");

    const byId = Object.fromEntries(researchHighlights.map((item) => [item.id, item]));
    expect(byId.cfr.value).toContain("0.0030");
    expect(byId.neural.detail).toMatch(/not adopted/i);
    expect(byId.bayesian.value).toMatch(/½|1\/2|half/i);
    expect(byId.rnr.value).toContain("+0.131");
    expect(byId.rnr.detail).toContain("+0.016");
    expect(byId.m7b.value).toContain("2.3%");
    expect(byId.m7b.value).toContain("6.7%");
  });

  it("states simulated-opponent scope and key non-claims", () => {
    expect(researchDisclaimer).toMatch(/simulated Leduc opponents/i);
    expect(limitations.some((item) => /human/i.test(item))).toBe(true);
    expect(limitations.some((item) => /Hold'?em/i.test(item))).toBe(true);
    expect(limitations.some((item) => /Neural/i.test(item))).toBe(true);
    expect(hero.localNote).toMatch(/not in the browser here/i);
  });

  it("exposes navigation, pipeline, and technology structure", () => {
    expect(navItems.map((item) => item.href)).toEqual([
      "#difference",
      "#how-it-works",
      "#research",
      "#preview",
      "#technology",
    ]);
    expect(pipelineSteps).toHaveLength(5);
    expect(differentiators).toHaveLength(3);
    expect(technology.some((item) => /PyTorch/i.test(item.name))).toBe(true);
    expect(technology.find((item) => /PyTorch/i.test(item.name))?.role).toMatch(
      /Not used as the deployed/i,
    );
  });

  it("references real screenshots and canonical external links", () => {
    expect(screenshots).toHaveLength(8);
    expect(screenshots.every((shot) => shot.src.startsWith("screenshots/"))).toBe(true);
    expect(screenshots.every((shot) => shot.alt.length > 20)).toBe(true);
    expect(links.github).toBe("https://github.com/sjyu0423/AI_Poker");
    expect(links.release).toContain("/releases/tag/v1.0.0");
    expect(siteMeta.version).toBe("1.0.0");
  });
});
