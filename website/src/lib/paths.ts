/**
 * Helpers for GitHub Pages repository subpath support (/AI_Poker/).
 *
 * next.config sets basePath/assetPrefix. Use these helpers for:
 * - absolute public asset paths when constructing URLs in content/tests
 * - documenting the expected Pages prefix
 */

export const DEFAULT_BASE_PATH = "/AI_Poker";

/** Base path used at build time (empty string means site root). */
export function getBasePath(): string {
  const fromEnv = process.env.NEXT_PUBLIC_BASE_PATH;
  if (fromEnv !== undefined) {
    return fromEnv.replace(/\/$/, "");
  }
  return DEFAULT_BASE_PATH;
}

/**
 * Prefix a site-root path with the configured base path.
 * Pass paths like "/screenshots/01.png" or "screenshots/01.png".
 */
export function withBasePath(path: string): string {
  const base = getBasePath();
  const normalized = path.startsWith("/") ? path : `/${path}`;
  if (!base) {
    return normalized;
  }
  if (normalized === "/") {
    return `${base}/`;
  }
  return `${base}${normalized}`;
}

/** True when href is an in-page hash or a same-site path (not external). */
export function isInternalHref(href: string): boolean {
  return href.startsWith("#") || href.startsWith("/");
}
