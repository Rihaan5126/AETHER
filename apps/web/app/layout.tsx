import { GeistMono } from "geist/font/mono";
import { GeistSans } from "geist/font/sans";
import type { Metadata } from "next";

import { Providers } from "@/components/layout/providers";
import { Shell } from "@/components/layout/shell";

import "./globals.css";

export const metadata: Metadata = {
  title: { default: "AETHER", template: "%s · AETHER" },
  description: "Autonomous Evolutionary Trading Arena: a research platform for evaluating evolving ML trading agents in simulation.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`}>
      <body className="font-sans">
        <Providers>
          <Shell>{children}</Shell>
        </Providers>
      </body>
    </html>
  );
}
