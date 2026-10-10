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
- **Generalisation gap**: validation versus final-test Sharpe and excess
  return per agent, with contaminated runs labelled.
- **Optuna trials**: objective per trial, pruned and failed trials.

## Example: the bundled demo

`aether demo --generations 3 --population 8 --seed 42` on the bundled
synthetic market is a pipeline check, not a result. In the runs made
while building it, the best evolved agent scored *below* Buy & Hold and its
validation Sharpe sat under the multiple-comparisons threshold. That is the expected outcome on data with no exploitable structure beyond a
small autocorrelation, and exactly what the platform is designed to reveal.
