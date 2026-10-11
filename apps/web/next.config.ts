import type { NextConfig } from "next";

// On Vercel there is no Python API or training worker, so default to the read-only
// snapshot in public/snapshot (scripts/export_snapshot.py). Set
// NEXT_PUBLIC_AETHER_SNAPSHOT=false there if the dashboard is pointed at a live API.
const onVercel = process.env.VERCEL === "1";
const vercelRepo =
  process.env.VERCEL_GIT_PROVIDER === "github" && process.env.VERCEL_GIT_REPO_OWNER && process.env.VERCEL_GIT_REPO_SLUG
    ? `https://github.com/${process.env.VERCEL_GIT_REPO_OWNER}/${process.env.VERCEL_GIT_REPO_SLUG}`
    : "";

const snapshot = (process.env.NEXT_PUBLIC_AETHER_SNAPSHOT ?? (onVercel ? "true" : "false")) === "true";
const apiOrigin = new URL(process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").origin;

// Production only: `next dev` needs eval for fast refresh. Next's bootstrap and
// Recharts use inline scripts/styles, hence 'unsafe-inline' (no third-party origins).
const csp = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  `connect-src 'self'${snapshot ? "" : ` ${apiOrigin}`}`,
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
].join("; ");

const nextConfig: NextConfig = {
  output: "standalone",
  env: {
    NEXT_PUBLIC_AETHER_SNAPSHOT: process.env.NEXT_PUBLIC_AETHER_SNAPSHOT ?? (onVercel ? "true" : "false"),
    NEXT_PUBLIC_REPO_URL: process.env.NEXT_PUBLIC_REPO_URL ?? vercelRepo,
  },
  reactStrictMode: true,
  poweredByHeader: false,
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "X-Frame-Options", value: "DENY" },
          ...(process.env.NODE_ENV === "production" ? [{ key: "Content-Security-Policy", value: csp }] : []),
        ],
      },
    ];
  },
};

export default nextConfig;
