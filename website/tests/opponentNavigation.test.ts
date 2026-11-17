import { describe, expect, it } from "vitest";
import { continueCopy, opponentNav, opponentPageMeta } from "@/content/opponentModelingPage";
import { links, navItems } from "@/content/site";
import { withBasePath } from "@/lib/paths";

describe("opponent-modeling navigation", () => {
  it("exposes a static-export friendly deep link", () => {
    expect(opponentPageMeta.path).toBe("/research/opponent-modeling/");
    expect(links.opponentModelingPage).toBe("/research/opponent-modeling/");
    expect(withBasePath(links.opponentModelingPage)).toBe(
      "/AI_Poker/research/opponent-modeling/",
    );
  });

  it("keeps in-page anchors and continues into the RNR exhibit", () => {
    expect(opponentNav.every((item) => item.href.startsWith("#"))).toBe(true);
    expect(continueCopy.primary.href).toBe("/research/robust-response/");
    expect(continueCopy.secondary.href).toBe("/research/cfr/");
  });

  it("is linked from the homepage nav", () => {
    expect(navItems.some((item) => item.href === "/research/opponent-modeling/")).toBe(true);
  });
});
