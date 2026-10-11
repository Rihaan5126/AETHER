/**
 * Zod schemas mirroring the FastAPI response models (services/api/aether_api/schemas).
 * Every response is validated at the boundary so a contract drift fails loudly
 * instead of rendering wrong numbers.
 */
import { z } from "zod";

const num = z.number().nullable();
const json = z.record(z.string(), z.unknown());

export const page = <T extends z.ZodTypeAny>(item: T) =>
  z.object({ items: z.array(item), total: z.number(), limit: z.number(), offset: z.number() });

export const JobRef = z.object({ job_id: z.string(), status: z.string() });

export const Job = z.object({
  id: z.string(),
  job_type: z.string(),
  status: z.enum(["queued", "running", "completed", "failed", "cancelled"]),
  progress: z.number(),
  message: z.string(),
  experiment_id: z.string().nullable(),
  attempts: z.number(),
  cancel_requested: z.boolean(),
  result: json.nullable(),
  error_message: z.string().nullable(),
  created_at: z.string(),
  started_at: z.string().nullable(),
  completed_at: z.string().nullable(),
  heartbeat_at: z.string().nullable(),
});
export type Job = z.infer<typeof Job>;

export const Dataset = z.object({
  id: z.string(),
  name: z.string(),
  source: z.string(),
  version: z.string(),
  hash: z.string(),
  symbols: z.array(z.string()),
  start_date: z.string(),
  end_date: z.string(),
  rows: z.number(),
  created_at: z.string(),
  retrieved_at: z.string(),
});
export type Dataset = z.infer<typeof Dataset>;

export const ValidationIssue = z.object({
  severity: z.string(),
  code: z.string(),
  message: z.string(),
  symbol: z.string().nullable(),
  count: z.number(),
});

export const DatasetDetail = Dataset.extend({
  params: json,
  validation: z.object({
    rows_in: z.number(),
    rows_out: z.number(),
    issues: z.array(ValidationIssue),
  }),
  used_by_experiments: z.number(),
});
export type DatasetDetail = z.infer<typeof DatasetDetail>;

export const DatasetPrices = z.object({
  dataset_id: z.string(),
  symbol: z.string(),
  downsampled: z.boolean(),
  points: z.array(
    z.object({
      timestamp: z.string(), open: z.number(), high: z.number(), low: z.number(),
      close: z.number(), volume: z.number(),
    }),
  ),
});

export const Warning = z.object({ code: z.string(), severity: z.string(), message: z.string() });
export type Warning = z.infer<typeof Warning>;

export const GenerationSummary = z.object({
  generation: z.number(),
  tournament_id: z.string(),
  status: z.string(),
  n_agents: z.number(),
  best_score: num,
  median_score: num,
  best_agent_id: z.string().nullable(),
  random_baseline_score: num,
  buy_and_hold_return: num,
});

export const Experiment = z.object({
  id: z.string(),
  name: z.string(),
  kind: z.string(),
  status: z.string(),
  dataset_id: z.string(),
  symbol: z.string(),
  seed: z.number(),
  current_generation: z.number().nullable(),
  planned_generations: z.number(),
  population_size: z.number(),
  progress: num,
  test_contaminated: z.boolean(),
  created_at: z.string(),
  started_at: z.string().nullable(),
  completed_at: z.string().nullable(),
});
export type Experiment = z.infer<typeof Experiment>;

export const FinalTestRun = z.object({
  id: z.string(),
  run_number: z.number(),
  status: z.string(),
  agent_ids: z.array(z.string()),
  results: z.array(json),
  benchmark_metrics: json.nullable(),
  test_period: json.nullable(),
  acknowledged_contamination: z.boolean(),
  created_at: z.string(),
});
export type FinalTestRun = z.infer<typeof FinalTestRun>;

export const ExperimentDetail = Experiment.extend({
  config: json,
  manifest: json.nullable(),
  code_version: z.string().nullable(),
  development_period: json.nullable(),
  test_period: json.nullable(),
  error_message: z.string().nullable(),
  wall_seconds: num,
  resource_usage: json.nullable(),
  workload: json,
  generations: z.array(GenerationSummary),
  jobs: z.array(Job),
  final_test_runs: z.array(FinalTestRun),
  optimization_studies: z.array(json),
});
export type ExperimentDetail = z.infer<typeof ExperimentDetail>;

export const AgentSummary = z.object({
  id: z.string(),
  experiment_id: z.string(),
  name: z.string(),
  algorithm: z.string(),
  generation: z.number(),
  parent_id: z.string().nullable(),
  origin: z.string(),
  seed: z.number(),
  is_benchmark: z.boolean(),
  genome: json,
  has_artifact: z.boolean(),
  created_at: z.string(),
  status: z.string().nullable(),
  rank: z.number().nullable(),
  score: num,
  total_return: num,
  sharpe_ratio: num,
  max_drawdown: num,
  warnings: z.array(Warning),
});
export type AgentSummary = z.infer<typeof AgentSummary>;

export const LineageEntry = z.object({
  parent_id: z.string().nullable(),
  operation: z.string(),
  mutations: z.array(z.string()),
  genome_before: json.nullable(),
  genome_after: json,
  training_seed: z.number(),
  validation_score: num,
});

export const AgentDetail = AgentSummary.extend({
  artifact_path: z.string().nullable(),
  artifact_sha256: z.string().nullable(),
  training_runs: z.array(
    z.object({
      id: z.string(),
      status: z.string(),
      tournament_id: z.string().nullable(),
      metrics: json,
      error: z.string().nullable(),
    }),
  ),
  lineage: z.array(LineageEntry),
  children: z.array(z.string()),
});
export type AgentDetail = z.infer<typeof AgentDetail>;

export const FoldMetric = z
  .object({
    fold: z.number(),
    val_start: z.string(),
    val_end: z.string(),
    train_start: z.string().optional(),
    train_end: z.string().optional(),
    total_return: z.number(),
    sharpe_ratio: num,
    max_drawdown: z.number(),
    n_trades: z.number(),
    benchmark_return: num,
  })
  .loose();

export const AgentMetrics = z.object({
  agent_id: z.string(),
  results: z.array(
    z.object({
      tournament_id: z.string(),
      generation: z.number(),
      period: z.string(),
      rank: z.number().nullable(),
      score: num,
      status: z.string(),
      status_reason: z.string(),
      metrics: json,
      fold_metrics: z.array(FoldMetric),
      score_components: z.record(z.string(), z.number()),
      warnings: z.array(Warning),
      risk_events: z.array(json),
    }),
  ),
  final_test: z.array(json),
});
export type AgentMetrics = z.infer<typeof AgentMetrics>;

export const LineageGraph = z.object({
  nodes: z.array(
    z.object({
      agent_id: z.string(),
      algorithm: z.string(),
      generation: z.number(),
      origin: z.string(),
      is_benchmark: z.boolean(),
      score: num,
      rank: z.number().nullable(),
      status: z.string().nullable(),
    }),
  ),
  edges: z.array(
    z.object({
      parent_id: z.string(),
      child_id: z.string(),
      operation: z.string(),
      mutations: z.array(z.string()),
    }),
  ),
});
export type LineageGraph = z.infer<typeof LineageGraph>;

export const Tournament = z.object({
  id: z.string(),
  experiment_id: z.string(),
  experiment_name: z.string(),
  generation: z.number(),
  period: z.string(),
  status: z.string(),
  config_fingerprint: z.string(),
  n_agents: z.number(),
  created_at: z.string(),
  completed_at: z.string().nullable(),
});
export type Tournament = z.infer<typeof Tournament>;

export const TournamentDetail = Tournament.extend({
  evaluation_config: json,
  folds: z.array(json),
  benchmark_metrics: json.nullable(),
});

export const LeaderboardRow = z.object({
  rank: z.number().nullable(),
  agent_id: z.string(),
  agent_name: z.string(),
  algorithm: z.string(),
  generation: z.number(),
  is_benchmark: z.boolean(),
  status: z.string(),
  status_reason: z.string(),
  score: num,
  score_components: z.record(z.string(), z.number()),
  total_return: num,
  annualized_return: num,
  sharpe_ratio: num,
  sortino_ratio: num,
  max_drawdown: num,
  turnover: num,
  n_trades: num,
  n_round_trips: num,
  win_rate: num,
  exposure: num,
  excess_return: num,
  stability: num,
  warnings: z.array(Warning),
});
export type LeaderboardRow = z.infer<typeof LeaderboardRow>;

export const Leaderboard = z.object({
  tournament_id: z.string(),
  experiment_id: z.string(),
  generation: z.number(),
  period: z.literal("validation"),
  benchmark: json.nullable(),
  ranking_policy: json,
  rows: z.array(LeaderboardRow),
});
export type Leaderboard = z.infer<typeof Leaderboard>;

export const CurvePoint = z.object({
  t: z.string(),
  equity: z.number(),
  drawdown: z.number(),
  weight: z.number(),
  fold: z.number(),
});
export type CurvePoint = z.infer<typeof CurvePoint>;

export const EquitySeries = z.object({ agent_id: z.string(), points: z.array(CurvePoint) });
export type EquitySeries = z.infer<typeof EquitySeries>;

export const EquityCurves = z.object({
  period: z.string(),
  downsampled: z.boolean(),
  series: z.array(EquitySeries),
  benchmark: EquitySeries.nullable(),
});
export type EquityCurves = z.infer<typeof EquityCurves>;

export const TradeRow = z.object({
  id: z.number(),
  agent_id: z.string(),
  symbol: z.string(),
  timestamp: z.string(),
  signal_time: z.string(),
  side: z.string(),
  quantity: z.number(),
  execution_price: z.number(),
  reference_price: z.number(),
  fees: z.number(),
  slippage: z.number(),
  notional: z.number(),
  target_weight: z.number(),
});
export type TradeRow = z.infer<typeof TradeRow>;

/** Stationary block-bootstrap interval (see trading_engine.research.robustness). */
export const BootInterval = z.object({ estimate: num, low: num, high: num, prob_positive: num });

export const Analytics = z.object({
  experiment_id: z.string(),
  available: z.boolean(),
  generations: z
    .array(
      z.object({
        generation: z.number(),
        tournament_id: z.string(),
        agents: z.array(
          z.object({
            agent_id: z.string(), algorithm: z.string(), score: num, status: z.string(),
            sharpe_ratio: num,
          }),
        ),
        best_score: num,
        median_score: num,
        random_score: num,
        buy_and_hold_score: num,
      }),
    )
    .optional(),
  fold_performance: z
    .array(
      z.object({
        agent_id: z.string(),
        algorithm: z.string(),
        is_benchmark: z.boolean(),
        folds: z.array(
          z.object({
            fold: z.number().nullable(), sharpe_ratio: num, total_return: num,
            max_drawdown: num, benchmark_return: num, val_start: z.string().nullable(),
            val_end: z.string().nullable(),
          }),
        ),
      }),
    )
    .optional(),
  family_variance: z
    .array(
      z.object({
        algorithm: z.string(), n: z.number(), mean: z.number(), std: num, min: z.number(),
        max: z.number(),
      }),
    )
    .optional(),
  parameter_sensitivity: z
    .array(z.object({ algorithm: z.string(), gene: z.string(), n: z.number(), spearman: num }))
    .optional(),
  generalisation: z
    .array(
      z.object({
        run_number: z.number(), agent_id: z.string(), algorithm: z.string(),
        train_sharpe: num.optional(),
        validation_sharpe: num, test_sharpe: num, validation_return: num, test_return: num,
        validation_excess: num, test_excess: num, contaminated: z.boolean(),
      }),
    )
    .optional(),
  regimes: z.array(json).optional(),
  uncertainty: z
    .array(
      z.object({
        agent_id: z.string(), algorithm: z.string(), status: z.string(), sessions: z.number(),
        train_sharpe: num.optional(), validation_sharpe: num,
        sharpe: BootInterval, excess_return: BootInterval,
      }),
    )
    .optional(),
  multiple_comparisons: z
    .object({
      agents_evaluated: z.number(),
      unique_training_runs: z.number(),
      validation_years: z.number(),
      expected_max_sharpe_under_null: num,
      best_observed_sharpe: num,
      random_baseline_sharpe: num,
      explanation: z.string(),
    })
    .optional(),
  optimization: z
    .array(
      z.object({
        id: z.string(), family: z.string(), status: z.string(), best_value: num,
        best_params: json.nullable(),
        trials: z.array(
          z.object({
            number: z.number(), state: z.string(), value: num,
            fold_scores: z.array(z.number()), duration_seconds: z.number(),
          }),
        ),
      }),
    )
    .optional(),
});
export type Analytics = z.infer<typeof Analytics>;

export const SystemStatus = z.object({
  status: z.string(),
  version: z.string(),
  environment: z.string(),
  database: json,
  queue: json,
  jobs: z.record(z.string(), z.number()),
  resources: json,
  limits: z.record(z.string(), z.number()),
  storage: json,
  dependencies: z.record(z.string(), z.string()),
  code_version: z.string(),
  write_protected: z.boolean(),
  disclaimer: z.string(),
});
export type SystemStatus = z.infer<typeof SystemStatus>;

export const Workload = z.object({
  agent_evaluations: z.number(),
  model_fits_upper_bound: z.number(),
  simulations_upper_bound: z.number(),
  relative_cost_index: z.number(),
  note: z.string(),
});
export type Workload = z.infer<typeof Workload>;
