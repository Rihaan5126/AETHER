"use client";

import {
  Activity,
  Beaker,
  Bot,
  Database,
  Dna,
  FlaskConical,
  GitCompareArrows,
  LayoutDashboard,
  Menu,
  Trophy,
  X,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import * as React from "react";

import { cn } from "@/lib/utils";

export const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/agents", label: "Agent Arena", icon: Bot },
  { href: "/tournaments", label: "Tournaments", icon: Trophy },
  { href: "/compare", label: "Agent Comparison", icon: GitCompareArrows },
  { href: "/evolution", label: "Evolution Lab", icon: Dna },
  { href: "/analytics", label: "Research Analytics", icon: FlaskConical },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/experiments", label: "Experiments", icon: Beaker },
  { href: "/system", label: "System", icon: Activity },
] as const;

function isActive(pathname: string, href: string) {
  return href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(`${href}/`);
}

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <nav aria-label="Main" className="flex flex-col gap-0.5">
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = isActive(pathname, href);
        return (
          <Link
            key={href}
            href={href}
            onClick={onNavigate}
            aria-current={active ? "page" : undefined}
            className={cn(
              "group flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm transition-colors",
              active ? "bg-raised text-fg" : "text-fg-muted hover:bg-raised/60 hover:text-fg",
            )}
          >
            <Icon className={cn("size-4", active ? "text-accent" : "text-fg-subtle group-hover:text-fg-muted")} aria-hidden />
            {label}
          </Link>
        );
      })}
    </nav>
  );
}

function Brand() {
  return (
    <Link href="/" className="flex items-center gap-2.5 px-2.5">
      <svg viewBox="0 0 24 24" className="size-5 text-accent" aria-hidden>
        <path d="M12 2 3 20h4.5L12 10.5 16.5 20H21L12 2Z" fill="currentColor" opacity="0.9" />
        <path d="M9.2 15.5h5.6" stroke="var(--color-bg)" strokeWidth="1.6" />
      </svg>
      <span className="text-sm font-semibold tracking-[0.2em] text-fg">AETHER</span>
    </Link>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="min-h-dvh">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-60 flex-col border-r border-line bg-panel px-3 py-4 lg:flex">
        <Brand />
        <p className="mt-1 px-2.5 text-[11px] text-fg-subtle">Evolutionary trading research</p>
        <div className="mt-6 flex-1">
          <NavLinks />
        </div>
        <p className="px-2.5 text-[11px] leading-relaxed text-fg-subtle">
          Simulation only. No brokerage connection. Not investment advice.
        </p>
      </aside>

      <header className="sticky top-0 z-30 flex h-12 items-center justify-between border-b border-line bg-panel/95 px-4 backdrop-blur lg:hidden">
        <Brand />
        <button
          type="button"
          aria-label={open ? "Close navigation" : "Open navigation"}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
          className="rounded-md p-1.5 text-fg-muted hover:bg-raised"
        >
          {open ? <X className="size-5" /> : <Menu className="size-5" />}
        </button>
      </header>
      {open ? (
        <div className="fixed inset-x-0 top-12 z-20 border-b border-line bg-panel p-3 lg:hidden">
          <NavLinks onNavigate={() => setOpen(false)} />
        </div>
      ) : null}

      <main className="lg:pl-60">
        <div className="mx-auto w-full max-w-[1400px] px-4 py-6 sm:px-6 lg:px-8 lg:py-8">{children}</div>
      </main>
    </div>
  );
}
