# Evaluation

## Metrics

Computed from the simulated equity curve (cash + marked position, after all
fees and slippage) and the trade ledger. Risk-free rate is 0; annualisation
uses 252 sessions. Undefined quantities are `null` (shown as "—"), never 0 or
infinity: Sharpe with zero volatility, win rate without completed round trips,
profit factor without losing trades, or any ratio on fewer than 20 sessions.

| Metric | Definition |
| --- | --- |
| Total return | final equity / initial capital − 1 |
| Annualised return | geometric, from total return and sessions |
| Volatility | std of daily returns × √252 |
| Sharpe | mean daily return / std × √252 |
| Sortino | mean daily return / downside deviation × √252 |
| Max drawdown | largest peak-to-trough fall, as a positive fraction |
| Calmar | annualised return / max drawdown |
| Exposure | share of sessions with a position |
| Turnover | traded notional / average equity, annualised |
| Trades, round trips | fills; completed entry-to-exit cycles |
| Win rate, profit factor, average holding period | over round trips |
| Excess return | total return − buy-and-hold total return on the same bars |
| Costs | total fees and slippage paid |

Validation metrics for an agent are computed on the chained validation folds;
per-fold metrics are stored as well.

## Ranking policy

Implemented in `tournaments/ranking.py`; parameters are part of the tournament
config fingerprint.

1. **Hard exclusion** (`excluded`): validation max drawdown above 50%, or a
   failed training run.
2. **Evidence requirement** (`insufficient_evidence`): active strategies need
   at least 5 completed round trips across all folds.
3. **Score** for everyone else:

   ```
   score = mean(fold Sharpe)
         − 0.5  × std(fold Sharpe)
         − 1.0  × max drawdown
         − 0.01 × annual turnover
   ```

   A fold with undefined Sharpe (never invested) counts as 0.
4. **Benchmarks** (`benchmark`): Random and Buy & Hold are scored and ranked in
   the same table for reference, exempt from exclusion (they are the
   reference), and never selected as parents or final-test candidates.

Order: by status (eligible and benchmark first, then insufficient evidence,
then excluded), then score, then id for a deterministic tie-break.

**Total return is deliberately not in the score.** A strategy that made more
money by taking far more risk, or by one lucky fold, should not win. The agent
detail page shows each score component, so a rank is always explainable.

## Statistical warnings

Attached to every result (`evaluation/warnings.py`):

| Code | Trigger |
| --- | --- |
| `small_sample` | fewer than 10 round trips |
| `high_drawdown` | max drawdown above 30% |
| `excessive_turnover` | annual turnover above 25× |
| `negative_after_costs` | positive before costs, negative after |
| `cost_dominated` | costs consumed more than half of the gross gain |
| `below_benchmark` | trailed buy-and-hold |
| `unstable_folds` | std of fold Sharpe above 1.0 |
| `single_period_dependence` | one fold produced more than 80% of the total gain |
| `suspiciously_strong` | validation Sharpe at least 2 with fewer than 30 round trips or under 2 years of validation data (high severity: usually luck, leakage or a bug) |
| `overfit_gap` | in-sample (training-window) Sharpe exceeds validation Sharpe by more than 1.0 |

Thresholds live in `WarningThresholds` and are part of the tournament's
`EvaluationConfig`, so they are recorded in every manifest. Warnings never
change a score; the ranking policy's hard exclusions are separate.

## Training, validation and test performance

Every number is labelled with the period it comes from:

| Period | How it is produced | Used for selection? |
| --- | --- | --- |
| **Training (in-sample)** | Each fold's fitted agent is also simulated on the rows it was trained on (`train_sharpe_ratio`, `train_sharpe_mean`) | No; reporting only |
| **Validation** | Walk-forward folds inside the development period | **Yes**: ranking, selection, evolution, Optuna |
| **Final test** | Locked tail of the data; run once on agents fitted on all development data | Never |

A large training -> validation drop is the classic overfitting signature
(`overfit_gap`). A validation -> test drop is selection bias plus regime
change. The Analytics page shows all three side by side.

## Uncertainty: block-bootstrap intervals

Point estimates of Sharpe ratio from a few years of daily data are noisy.
For each final-generation agent the analytics endpoint reports 95%
**stationary block bootstrap** intervals (Politis & Romano; geometric blocks,
mean 20 sessions, 1,000 resamples) for:

* the annualised Sharpe ratio, with P(Sharpe > 0);
* the annualised **excess return over buy-and-hold**. Agent and benchmark
  days are resampled with the same indices, so their correlation is kept.

Assumptions: returns are stationary and only weakly dependent within the
window. Blocks preserve short-range dependence such as volatility
clustering; ordinary i.i.d. resampling would understate uncertainty. The
implementation is tested for nominal coverage (90% intervals cover a known
true Sharpe in 80-97% of simulated trials) and for widening under
autocorrelation (AR(1), phi = 0.5: intervals 1.3-2.1x wider than i.i.d.,
theory 1.73x). An interval that spans zero is shown in amber: the data cannot
distinguish the result from no effect.

## Random-strategy null and cost sensitivity (research harness)

`trading_engine.research` also provides:

* **Random null**: the Sharpe distribution of N seeded random long/flat
  strategies through the same simulator, window and costs. An agent's
  percentile in it measures how unusual its result is compared with luck,
  without distributional assumptions.
* **Cost sensitivity**: the same target weights re-simulated with fees and
  slippage scaled 0x, 1x, 2x, 5x and 10x. This is exact, because decisions do
  not depend on costs.

Both are used in `scripts/research/phase2_study.py`; results are in
`docs/EVOLUTION_RESEARCH.md` and `docs/research/PHASE2_TABLES.md`.

## Multiple comparisons

With *N* skill-less strategies evaluated on the same data, the best annualised
Sharpe ratio is expected to reach roughly `√(2 ln N) / √years` by chance alone.
Research Analytics shows this threshold next to the best observed validation
Sharpe and the Random baseline, using the number of unique training runs as
*N*. If the best agent does not clear it, the dashboard says so. This is a
rough guide (it assumes independent strategies, which evolved relatives are
not), not a formal test; the Deflated Sharpe Ratio would be the next step.

## Other analytics

- **Fold heatmap**: Sharpe per fold for the final generation; large spread
  means results depend on the period.
- **Family variance**: score distribution within each algorithm family.
- **Parameter sensitivity**: Spearman correlation between each numeric gene
  and score, per family (descriptive; small samples).
- **Regimes**: annualised mean return when trailing 60-day buy-and-hold trend
  was up or down, and volatility high or low.
- **Generalisation gap**: training versus validation versus final-test Sharpe
  and excess return per agent, with contaminated runs labelled.
- **Uncertainty**: bootstrap intervals (above).
- **Optuna trials**: objective per trial, pruned and failed trials.

## Example: the bundled demo

`aether demo --generations 3 --population 8 --seed 42` on the bundled
synthetic market is a pipeline check, not a result. In the runs made
while building it, the best evolved agent scored *below* Buy & Hold and its
validation Sharpe sat under the multiple-comparisons threshold. That is the expected outcome on data with no exploitable structure beyond a
small autocorrelation, and exactly what the platform is designed to reveal.
