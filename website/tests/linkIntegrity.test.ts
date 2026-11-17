import { describe, expect, it } from "vitest";
import { continueCopy as cfrContinue } from "@/content/cfrPage";
import { continueCopy as opponentContinue } from "@/content/opponentModelingPage";
import { closingCopy } from "@/content/robustResponsePage";
import { links, navItems, publishedRoutes } from "@/content/site";

const knownInternal = new Set<string>(publishedRoutes);

describe("cross-page link integrity", () => {
  it("keeps homepage nav routes inside the published set", () => {
    for (const item of navItems) {
      if (item.href.startsWith("/") && !item.href.startsWith("//")) {
        expect(knownInternal.has(item.href)).toBe(true);
      }
    }
  });

  it("chains the research exhibits in order", () => {
    expect(cfrContinue.primary.href).toBe(links.opponentModelingPage);
    expect(opponentContinue.primary.href).toBe(links.robustResponsePage);
    expect(closingCopy.secondary.href).toBe(links.opponentModelingPage);
    expect(closingCopy.tertiary.href).toBe(links.cfrPage);
  });

  it("points documentation links at the public GitHub repository", () => {
    const docs = [
      links.overview,
      links.researchFreeze,
      links.cfr,
      links.m5,
      links.m7a,
      links.m7b,
      links.quickStart,
    ];
    for (const href of docs) {
      expect(href.startsWith("https://github.com/sjyu0423/AI_Poker")).toBe(true);
    }
  });
});
