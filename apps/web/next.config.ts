import type { NextConfig } from "next";

// On Vercel there is no Python API or training worker, so default to the read-only
// snapshot in public/snapshot (scripts/export_snapshot.py). Set
// NEXT_PUBLIC_AETHER_SNAPSHOT=false there if the dashboard is pointed at a live API.
const onVercel = process.env.VERCEL === "1";
const vercelRepo =
  process.env.VERCEL_GIT_PROVIDER === "github" && process.env.VERCEL_GIT_REPO_OWNER && process.env.VERCEL_GIT_REPO_SLUG
    ? `https://github.com/${process.env.VERCEL_GIT_REPO_OWNER}/${process.env.VERCEL_GIT_REPO_SLUG}`
    : "";

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
        ],
      },
    ];
  },
};

export default nextConfig;
