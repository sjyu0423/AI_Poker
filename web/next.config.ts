import type { NextConfig } from "next";

/** The FastAPI backend (`pokermind serve`). The browser talks only to this Next.js
 * server, which forwards `/api/*` to it, so the backend needs no CORS settings. */
const API_URL = process.env.POKERMIND_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/:path*` }];
  },
};

export default nextConfig;
