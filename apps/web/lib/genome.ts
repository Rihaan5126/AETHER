/** Human-readable summaries of typed genomes. */

type Genome = Record<string, unknown>;

const fmt = (v: unknown): string => {
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toPrecision(3);
  if (Array.isArray(v)) return `[${v.join(", ")}]`;
  if (v === null || v === undefined) return "—";
  return String(v);
};

const KEY_GENES: Record<string, string[]> = {
  random: ["p_long", "switch_prob", "max_allocation"],
  buy_and_hold: ["allocation"],
  momentum: ["rule", "fast_window", "slow_window", "trend_lookback", "max_allocation"],
  random_forest: ["n_estimators", "max_depth", "min_samples_leaf", "max_features"],
  xgboost: ["n_estimators", "max_depth", "learning_rate", "subsample", "reg_lambda"],
  mlp: ["hidden_layers", "dropout", "learning_rate", "weight_decay", "max_epochs"],
};

export function genomeSummary(genome: Genome): [string, string][] {
  const family = String(genome.family ?? "");
  const rows: [string, string][] = (KEY_GENES[family] ?? []).map((k) => [k, fmt(genome[k])]);
  const policy = genome.policy as Genome | undefined;
  if (policy) {
    rows.push(["threshold", fmt(policy.threshold)], ["sizing", fmt(policy.sizing)], ["max allocation", fmt(policy.max_allocation)]);
  }
  if (Array.isArray(genome.features)) rows.push(["features", `${genome.features.length} selected`]);
  if ("train_window" in genome) rows.push(["train window", genome.train_window ? fmt(genome.train_window) : "all"]);
  return rows;
}

/** Flatten a genome into dotted keys for diffing. */
export function flattenGenome(genome: Genome | null | undefined, prefix = ""): Record<string, string> {
  const out: Record<string, string> = {};
  if (!genome) return out;
  for (const [k, v] of Object.entries(genome)) {
    if (v && typeof v === "object" && !Array.isArray(v)) Object.assign(out, flattenGenome(v as Genome, `${prefix}${k}.`));
    else out[`${prefix}${k}`] = fmt(v);
  }
  return out;
}

export function genomeDiff(before: Genome | null | undefined, after: Genome): { key: string; before: string | null; after: string }[] {
  const a = flattenGenome(before);
  const b = flattenGenome(after);
  return Object.keys(b)
    .filter((k) => a[k] !== b[k])
    .map((k) => ({ key: k, before: a[k] ?? null, after: b[k]! }));
}
