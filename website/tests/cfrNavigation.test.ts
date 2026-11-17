import { describe, expect, it } from "vitest";
import { cfrNav, cfrPageMeta, continueCopy } from "@/content/cfrPage";
import { links, navItems } from "@/content/site";
import { withBasePath } from "@/lib/paths";

describe("CFR page navigation helpers", () => {
  it("exposes a static-export friendly research path", () => {
    expect(cfrPageMeta.path).toBe("/research/cfr/");
    expect(links.cfrPage).toBe("/research/cfr/");
    expect(withBasePath(links.cfrPage)).toBe("/AI_Poker/research/cfr/");
  });

  it("keeps in-page anchors and links forward to opponent modeling", () => {
    expect(cfrNav.every((item) => item.href.startsWith("#"))).toBe(true);
    expect(continueCopy.primary.href).toBe("/research/opponent-modeling/");
    expect(withBasePath(continueCopy.primary.href)).toBe(
      "/AI_Poker/research/opponent-modeling/",
    );
  });

  it("links the homepage nav into the CFR exhibit", () => {
    expect(navItems.some((item) => item.href === "/research/cfr/")).toBe(true);
  });
});
