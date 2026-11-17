import { describe, expect, it } from "vitest";
import { DEFAULT_BASE_PATH, getBasePath, isInternalHref, withBasePath } from "@/lib/paths";

describe("path helpers", () => {
  it("defaults to the GitHub Pages repository subpath", () => {
    expect(DEFAULT_BASE_PATH).toBe("/AI_Poker");
    expect(getBasePath()).toBe("/AI_Poker");
  });

  it("prefixes asset and page paths for Pages hosting", () => {
    expect(withBasePath("/screenshots/01-new-session.png")).toBe(
      "/AI_Poker/screenshots/01-new-session.png",
    );
    expect(withBasePath("screenshots/07-dashboard-strategy.png")).toBe(
      "/AI_Poker/screenshots/07-dashboard-strategy.png",
    );
    expect(withBasePath("/")).toBe("/AI_Poker/");
  });

  it("recognizes internal navigation targets", () => {
    expect(isInternalHref("#research")).toBe(true);
    expect(isInternalHref("/")).toBe(true);
    expect(isInternalHref("https://github.com/sjyu0423/AI_Poker")).toBe(false);
  });
});
