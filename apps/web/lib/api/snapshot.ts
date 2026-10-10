/**
 * Read-only snapshot mode: answer GET requests from static files exported by
 * `scripts/export_snapshot.py` instead of calling the API. Lets the dashboard be
 * hosted without the Python backend (e.g. on Vercel).
 *
 * The exported files are verbatim API responses. List endpoints are exported in
 * full and filtered/paginated here with the same rules as the API routes, so a
 * page sees the response it would get from the live API.
 */

export type Query = Record<string, string | number | boolean | null | undefined | string[]>;

type Page = { items: Record<string, unknown>[]; total: number; limit: number; offset: number };

/** Mirrors routes/deps.py Pagination defaults. */
const DEFAULT_LIMIT = 50;

/** Snapshot file (relative to /snapshot, without .json) plus an optional in-browser transform. */
export type Resolved = { file: string; transform?: (data: unknown) => unknown };

function str(q: Query, key: string): string | undefined {
  const v = q[key];
  if (v === undefined || v === null || v === "" || Array.isArray(v)) return undefined;
  return String(v);
}

function list(q: Query, key: string): string[] | undefined {
  const v = q[key];
  if (v === undefined || v === null || v === "") return undefined;
  return Array.isArray(v) ? v : [String(v)];
}

function paginate(q: Query, filter: (item: Record<string, unknown>) => boolean) {
  return (data: unknown): Page => {
    const items = (data as Page).items.filter(filter);
    const limit = Number(str(q, "limit") ?? DEFAULT_LIMIT);
    const offset = Number(str(q, "offset") ?? 0);
    return { items: items.slice(offset, offset + limit), total: items.length, limit, offset };
  };
}

/** Field equals query value when the query value is set (string comparison, like the API's typed match). */
const eq = (q: Query, key: string, field = key) => (item: Record<string, unknown>) => {
  const want = str(q, key);
  return want === undefined || String(item[field]) === want;
};

const all =
  (...preds: ((item: Record<string, unknown>) => boolean)[]) =>
  (item: Record<string, unknown>) =>
    preds.every((p) => p(item));

export function resolveSnapshot(path: string, query: Query = {}): Resolved {
  const clean = path.replace(/^\/+|\/+$/g, "");
  const mp = str(query, "max_points");
  let m: RegExpMatchArray | null;

  switch (clean) {
    case "experiments":
      return { file: "experiments", transform: paginate(query, all(eq(query, "status"), eq(query, "kind"))) };
    case "agents": {
      const noBench = str(query, "include_benchmarks") === "false";
      return {
        file: "agents",
        transform: paginate(
          query,
          all(eq(query, "experiment_id"), eq(query, "generation"), eq(query, "algorithm"), (a) => !(noBench && a.is_benchmark)),
        ),
      };
    }
    case "tournaments":
      return { file: "tournaments", transform: paginate(query, all(eq(query, "experiment_id"), eq(query, "generation"))) };
    case "datasets":
      return { file: "datasets", transform: paginate(query, eq(query, "source")) };
  }

  if ((m = clean.match(/^tournaments\/([^/]+)\/equity-curves$/))) {
    const ids = list(query, "agent_ids");
    return {
      file: `tournaments/${m[1]}/equity-curves.mp${mp ?? 600}`,
      transform: ids
        ? (data) => {
            const d = data as { series: { agent_id: string }[] };
            return { ...d, series: d.series.filter((s) => ids.includes(s.agent_id)) };
          }
        : undefined,
    };
  }
  if ((m = clean.match(/^tournaments\/([^/]+)\/trades$/))) {
    return { file: `tournaments/${m[1]}/trades`, transform: paginate(query, eq(query, "agent_id")) };
  }
  if ((m = clean.match(/^datasets\/([^/]+)\/prices$/))) {
    return { file: `datasets/${m[1]}/prices.${str(query, "symbol") ?? ""}.mp${mp ?? 500}` };
  }
  if ((m = clean.match(/^experiments\/([^/]+)\/final-test\/(\d+)\/equity-curves$/))) {
    return { file: `experiments/${m[1]}/final-test/${m[2]}/equity-curves.mp${mp ?? 600}` };
  }
  return { file: clean };
}

export function snapshotUrl(file: string): string {
  return `/snapshot/${file.split("/").map(encodeURIComponent).join("/")}.json`;
}
