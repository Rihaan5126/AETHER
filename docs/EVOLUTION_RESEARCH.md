# Does evolution find strategies that generalise?

**Short answer: no.** On five liquid US instruments over 21 years, evolution
reliably improves the score it is selected on, and it improves the
*average* member of the population on unseen data. But it does not produce a
better *best* agent than its own first generation, nor than an equal-budget
random search. On the locked final test, none of its 15 champions beat
buy-and-hold.

Every number below comes from `docs/research/phase2_results.json`, tabulated
by `scripts/research/report.py` into `docs/research/PHASE2_TABLES.md`
(the full per-run tables are there).

## Design (pre-registered)

The design, dataset hash, code version and selection rules were written to
`docs/research/phase2_preregistration.json` before any model trained.

* **Data**: Yahoo Finance daily bars, split and dividend adjusted, for SPY,
  QQQ, AAPL, MSFT and NVDA, 2005-01-03 to 2026-10-09: 27,385 rows, one
  immutable dataset (`ds_aa0bb65d9164`, SHA-256 identified).
* **Periods** (SPY; other symbols are within a day):

  | Period | Dates | Used for |
  | --- | --- | --- |
  | Selection | 2005-01-03 to 2018-12-04 | Walk-forward folds: ranking, evolution, random search |
  | Inner holdout | 2018-12-06 to 2022-05-27 | Checking whether selection generalises (never used to select) |
  | Locked final test | 2022-05-31 to 2026-10-09 | Run once, on pre-registered champions only |

* **Costs**: 1 bp fee + 5 bp slippage per trade, fills at the next open,
  long-only, no leverage.
* **Main grid**: 8 agents × 3 generations; 5 symbols × seeds 1, 2, 3 = 15 runs.
* **Scaling**: SPY with 16 × 5 and 32 × 5, seeds 1 and 2.
* **Controls**:
  * *Initial champion*: rank-1 eligible agent of generation 0 (no evolution).
  * *Random-search champion*: generation 0 plus random genomes, with the same
    number of unique evaluations as the evolution run (19-20 for the main
    grid). It answers "is evolution better than trying random configurations?"
  * *Buy-and-hold* and a *null* of 300 random long/flat strategies per window.
* **Statistics**: per-run paired differences with an exact two-sided sign
  test (ties dropped); stationary block-bootstrap 95% intervals (mean block 20
  sessions, 2,000 resamples); random-null percentiles. Runs on the same
  symbol share data, so p-values are descriptive.

## Results on the main grid (15 runs)

| Question | Wins / losses / ties | Mean diff | Sign test p |
| --- | --- | ---: | ---: |
| Does the best **selection** score rise from gen 0 to the last gen? | 13 / 0 / 2 | +0.120 | 0.0002 |
| Is the evolved champion better than the gen-0 champion on the **holdout** (Sharpe)? | 6 / 7 / 2 | -0.069 | 1.00 |
| Is the evolved champion better than the **random-search** champion on the holdout? | 4 / 9 / 2 | -0.052 | 0.27 |
| Is the evolved **population** (mean holdout Sharpe) better than generation 0? | 14 / 1 / 0 | +0.134 | 0.001 |
| Does the evolved champion beat **buy-and-hold** on the holdout? | 1 / 14 / 0 | -0.285 | 0.001 |

* The evolved champion's holdout excess-return interval was entirely above
  zero in **0 of 15** runs.
* It reached the 95th percentile of the 300-strategy random null in **0 of 15**
  runs (median percentile 0.55, i.e. as good as a typical coin-flip strategy).
* In 2 of 15 runs evolution never displaced the generation-0 champion.

### Reading these together

1. **Selection improvement is guaranteed, not evidence.** Elitism means the
   best selection-data score cannot fall. The 13/0 result is the mechanism
   working, not proof that better strategies were found.
2. **The population gets better, the champion does not.** Evolution removes
   bad random configurations, so the *average* agent improves out of sample
   (14/1). But the agent selection picks as best is no better on new data
   than the best of the starting population (6/7), and slightly worse than
   random search (4/9). That is the signature of **selection overfitting**:
   the top of a ranking built on reused validation folds is partly noise.
3. **Even the improved population converges towards, not past, the market.**
   Population mean holdout Sharpe rose towards buy-and-hold (e.g. SPY 32x5
   seed 1: 0.28 -> 0.62 vs buy-and-hold 0.74) but stayed below it in every
   main-grid run.

## Scaling: more agents and generations (SPY)

| Run | Unique evaluations | Best selection score gen0 -> last | Evolved champion holdout Sharpe | Buy-and-hold | Population mean holdout Sharpe gen0 -> last |
| --- | ---: | --- | ---: | ---: | --- |
| 8x3 seed 1 | 20 | 0.602 -> 0.602 | 0.31 | 0.74 | 0.39 -> 0.53 |
| 8x3 seed 2 | 19 | 0.628 -> 0.642 | 0.57 | 0.74 | 0.50 -> 0.57 |
| 16x5 seed 1 | 71 | 0.670 -> 0.745 | 0.76 | 0.74 | 0.33 -> 0.66 |
| 16x5 seed 2 | 72 | 0.625 -> 0.776 | 0.78 | 0.74 | 0.34 -> 0.61 |
| 32x5 seed 1 | 149 | 0.586 -> 0.767 | 0.81 | 0.74 | 0.28 -> 0.62 |
| 32x5 seed 2 | 150 | 0.664 -> 0.738 | 0.60 | 0.74 | 0.39 -> 0.58 |

Larger budgets raise selection scores further, and three of four larger runs
produced a champion with a holdout Sharpe slightly above buy-and-hold's 0.74.
But the excess-return intervals all still span zero (e.g. 32x5 seed 1:
-10.3% annual excess, 95% CI [-21.2%, +2.8%]). With 4 runs this is not
evidence of an effect, and more evaluations also mean more chances of a lucky
winner (multiple comparisons).

## Locked final test (2022-05-31 to 2026-10-09)

Run once on 2026-10-11 after all selection was complete. The champions were
refitted on the full development period. Nothing was changed afterwards.

| | Count |
| --- | --- |
| Evolved champions tested | 15 |
| Higher test Sharpe than buy-and-hold | **0** |
| Positive excess return vs buy-and-hold | **0** |
| Excess-return CI entirely above zero | **0** |
| Sharpe CI entirely above zero (i.e. made money with some confidence) | 8 |
| At or above the 95th percentile of the random null | **0** |

The best final-test results came from non-evolved controls: the generation-0
MLP on QQQ seed 3 (test Sharpe 1.20 vs buy-and-hold 1.07; +2.2% annual excess,
CI [-2.9%, +8.3%]) and a random-search momentum agent on MSFT seed 1 (0.99
vs 0.72; excess CI [-23.7%, +14.0%]). Neither interval excludes zero, and
with 40 agent-tests some such results are expected by chance.

A separate set of five showcase experiments (seed 11, run through the API
and visible in the dashboard) gives the same picture on 2022-2026: every
evolved champion trailed buy-and-hold on return (SPY +17.6% vs +99.1%, NVDA
+90.7% vs +1,128.8%). In all five, the random baseline earned more than the
champion, because a strong bull market rewards staying invested.

## Conclusion

On this data and cost model, **evolutionary search over these model
families does not find strategies that generalise better than its starting
point or than random search, and none beats passive exposure out of
sample.** The platform's integrity features (inner holdout, equal-budget
control, locked test, bootstrap intervals, random null) are what make that
conclusion possible. Without them, the rising selection curve in the
Evolution Lab would look like success.

## Threats to validity

* 15 runs over 5 correlated US large-cap instruments; one market regime
  family (2018-2026 was largely a bull market, which favours buy-and-hold).
* Daily bars and a single prediction horizon (1 day); long-only; no
  short-term or cross-sectional strategies.
* Small evolution budgets (8x3) in the main grid; scaling was run on SPY only.
* The selection score reuses the same validation folds every generation by
  design. This study measures that effect rather than removing it.
* The first study attempt was discarded because of a simulator bug (see
  `docs/research/archive/attempt1_negative_cash_bug/README.md`). This rerun
  uses the same design on the fixed code and never touched the final test
  before the end.
