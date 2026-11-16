import type { NextConfig } from "next";

/**
 * GitHub Pages serves this project site under /AI_Poker/.
 * Override with NEXT_PUBLIC_BASE_PATH="" for a root-domain preview if needed.
 */
const basePath =
  process.env.NEXT_PUBLIC_BASE_PATH !== undefined
    ? process.env.NEXT_PUBLIC_BASE_PATH
    : "/AI_Poker";

const nextConfig: NextConfig = {
  output: "export",
  basePath: basePath || undefined,
  assetPrefix: basePath || undefined,
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  reactStrictMode: true,
  poweredByHeader: false,
};

export default nextConfig;
