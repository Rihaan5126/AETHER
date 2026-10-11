# Attempt 1 (discarded): simulator negative-cash bug

These files are the first, incomplete run of the Phase 2 study (14 of 15
main-grid runs). They are kept as a record but **must not be used for
conclusions**.

During this run, the per-fold training-window backtest (new in Phase 2) hit a
latent simulator bug: the negative-cash guard used a fixed tolerance of 1e-9.
When an overfit model compounded in-sample equity past ~1e7, float rounding
after an all-in buy exceeded that tolerance. The simulator then raised
"negative cash" on a correct trade, and the agent was marked failed and
excluded from ranking. The tolerance is now relative to the cash balance
(regression test: `test_full_allocation_at_large_equity_never_trips_cash_check`).

The study was stopped and re-run from scratch on the fixed code. The final
test was never run for this attempt, so it did not touch the locked test
period. The rerun's design and selection rules are unchanged.
