import { describe, expect, it } from "vitest";
import {
  closingCopy,
  robustNav,
  robustPageMeta,
  rnrCopy,
} from "@/content/robustResponsePage";
import { links, navItems } from "@/content/site";
import { withBasePath } from "@/lib/paths";

describe("robust-response navigation", () => {
  it("exposes a static-export friendly deep link", () => {
    expect(robustPageMeta.path).toBe("/research/robust-response/");
    expect(links.robustResponsePage).toBe("/research/robust-response/");
    expect(withBasePath(links.robustResponsePage)).toBe(
      "/AI_Poker/research/robust-response/",
    );
  });

  it("keeps in-page anchors and closing navigation", () => {
    expect(robustNav.every((item) => item.href.startsWith("#"))).toBe(true);
    expect(closingCopy.primary.href).toBe("/");
    expect(closingCopy.secondary.href).toBe("/research/opponent-modeling/");
    expect(closingCopy.tertiary.href).toBe("/research/cfr/");
  });

  it("is linked from the homepage nav", () => {
    expect(navItems.some((item) => item.href === "/research/robust-response/")).toBe(
      true,
    );
  });

  it("states that the p dial is conceptual", () => {
    expect(rnrCopy.dialDisclaimer.toLowerCase()).toMatch(/does not invent/);
    expect(rnrCopy.pNotBudget.toLowerCase()).toMatch(/not the exploitability budget/);
  });
});
