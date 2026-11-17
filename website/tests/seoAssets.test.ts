import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { publishedRoutes, siteMeta } from "@/content/site";
import { withBasePath } from "@/lib/paths";

const publicDir = join(process.cwd(), "public");

describe("SEO and sharing assets", () => {
  it("publishes a dedicated social-sharing image", () => {
    expect(siteMeta.ogImage).toBe("og-share.jpg");
    expect(existsSync(join(publicDir, siteMeta.ogImage))).toBe(true);
  });

  it("ships robots.txt and sitemap.xml for the Pages base path", () => {
    const robots = readFileSync(join(publicDir, "robots.txt"), "utf8");
    const sitemap = readFileSync(join(publicDir, "sitemap.xml"), "utf8");

    expect(robots).toContain("Sitemap: https://sjyu0423.github.io/AI_Poker/sitemap.xml");
    expect(sitemap).toContain("https://sjyu0423.github.io/AI_Poker/");
    expect(sitemap).toContain("https://sjyu0423.github.io/AI_Poker/research/cfr/");
    expect(sitemap).toContain(
      "https://sjyu0423.github.io/AI_Poker/research/opponent-modeling/",
    );
    expect(sitemap).toContain(
      "https://sjyu0423.github.io/AI_Poker/research/robust-response/",
    );
  });

  it("lists every published route under the GitHub Pages prefix", () => {
    expect(publishedRoutes).toEqual([
      "/",
      "/research/cfr/",
      "/research/opponent-modeling/",
      "/research/robust-response/",
    ]);
    expect(withBasePath("/")).toBe("/AI_Poker/");
    expect(withBasePath("/research/robust-response/")).toBe(
      "/AI_Poker/research/robust-response/",
    );
  });
});
