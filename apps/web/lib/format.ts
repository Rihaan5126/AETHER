/** Display helpers. Undefined metrics render as an em dash, never as 0 or Infinity. */

export const DASH = "—";

export function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

export function pct(v: number | null | undefined, digits = 1, signed = false): string {
  if (!isNum(v)) return DASH;
  const s = (v * 100).toFixed(digits);
  return `${signed && v > 0 ? "+" : ""}${s}%`;
}

export function num(v: number | null | undefined, digits = 2, signed = false): string {
  if (!isNum(v)) return DASH;
  return `${signed && v > 0 ? "+" : ""}${v.toFixed(digits)}`;
}

export function int(v: number | null | undefined): string {
  if (!isNum(v)) return DASH;
  return Math.round(v).toLocaleString("en-US");
}

export function money(v: number | null | undefined): string {
  if (!isNum(v)) return DASH;
  return v.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}

export function duration(seconds: number | null | undefined): string {
  if (!isNum(seconds)) return DASH;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return `${m}m ${s}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return DASH;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return DASH;
  return d.toLocaleString("en-GB", {
    year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

export function relative(iso: string | null | undefined, now: number = Date.now()): string {
  if (!iso) return DASH;
  const diff = (now - new Date(iso).getTime()) / 1000;
  if (!Number.isFinite(diff)) return DASH;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export function signClass(v: number | null | undefined): string {
  if (!isNum(v) || v === 0) return "text-fg";
  return v > 0 ? "text-pos" : "text-neg";
}

export const ALGORITHM_LABEL: Record<string, string> = {
  random: "Random",
  buy_and_hold: "Buy & Hold",
  momentum: "Momentum",
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  mlp: "MLP",
};

export const ALGORITHM_COLOR: Record<string, string> = {
  random: "var(--color-algo-random)",
  buy_and_hold: "var(--color-algo-buy_and_hold)",
  momentum: "var(--color-algo-momentum)",
  random_forest: "var(--color-algo-random_forest)",
  xgboost: "var(--color-algo-xgboost)",
  mlp: "var(--color-algo-mlp)",
};

export function algoLabel(a: string): string {
  return ALGORITHM_LABEL[a] ?? a;
}

export function algoColor(a: string): string {
  return ALGORITHM_COLOR[a] ?? "var(--color-fg-muted)";
}

/** Short display id: "exp_6c6e37da5d0f-g2-07-mlp" -> "G2·07". */
export function shortAgentId(id: string): string {
  const m = id.match(/-g(\d+)-(\d+)-/);
  return m ? `G${m[1]}·${m[2]}` : id.slice(-10);
}
