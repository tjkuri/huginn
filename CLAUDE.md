# Huginn

Sports prediction backtest and simulation engine. Reads data from sibling repo `../yggdrasil/cache/`.

## Project Structure

```
nba/config.py                    Cache path, vig constants, model config reference
nba/data/loader.py               Reads paired prediction/result JSON, merges by game_id
nba/data/model_inputs_loader.py  Reads model-inputs + results + predictions for fitness
nba/backtest/grader.py           Grades games: V2 + V1 + beat-the-book fields
nba/backtest/metrics.py          Aggregates stats: overall, breakdowns, book_comparison
nba/backtest/report.py           Rich terminal output
nba/models/v2_current.py         Port of Yggdrasil's nbaMath.js (weighted stats, model math)
nba/optimizer/fitness.py         Evaluates candidate config; supports temporal CV
nba/scripts/run_backtest.py      CLI entry point (--days, --team, --json, --cache)
nba/scripts/run_optimizer.py     Optuna parameter search CLI (--trials, --target, --export)
tests/nba/                       pytest test suite (132 tests)
```

## Commands

```bash
# Activate venv
source venv/bin/activate

# Run backtest
python nba/scripts/run_backtest.py

# Run tests
python -m pytest tests/nba/ -v

# Run optimizer
python nba/scripts/run_optimizer.py --trials 500 --target beat_rate
python nba/scripts/run_optimizer.py --target avg_miss --export
```

## Key Conventions

- **Parity with Yggdrasil math is critical.** The Python model math must match `yggdrasil/utils/nbaMath.js` exactly. Verified via integration tests against cached predictions.
- **game_id is always a string.** Both prediction and result files may store it as a number — coerce to string for matching.
- **avg_miss uses absolute values.** `mean(|v2_miss|)` for all games including NO_BET.
- **Calibration recomputes normalCDF from z_score.** Does NOT use the pre-computed `opening_win_prob` from predictions. Uses Abramowitz & Stegun approximation matching Yggdrasil's implementation.
- **V1 skips grading when gap is zero.** If `v1_line == opening_dk_line`, v1_direction and v1_result are None.
- **win_rate rounds to 4 decimals, roi to 2.** Matches JS `parseFloat((...).toFixed(N))`.

## Yggdrasil Reference

Key files in the sibling repo for understanding the data:
- `yggdrasil/config/nba.js` — model config values
- `yggdrasil/utils/nbaMath.js` — model math that `nba/models/v2_current.py` ports
- `yggdrasil/cache/` — `YYYY-MM-DD-nba-{predictions,results,model-inputs}.json` files

## Key Conventions (Phase 2)

- **Beat-the-book uses strict less-than.** `v2_beat_book = (v2_abs_miss < dk_abs)` — ties are not a beat.
- **book_comparison includes all games.** Beat rate and avg advantage computed over all games with projections, not just bets.
- **Fitness function config uses per-dollar vig.** `vig_win=0.9091, vig_risk=1.0` (not the dollar amounts in nba/config.py).
- **Model port matches JS exactly.** Parity verified via integration tests against cached predictions.

## Key Conventions (Phase 3)

- **evaluate_config returns flat dict.** Keys: beat_rate, avg_miss, avg_advantage, roi, win_rate, total_bets, total_games. Penalty values for degenerate cases.
- **Temporal CV splits by date.** Games on the same date stay in the same fold. Falls back to full evaluation if fewer game-days than k.
- **sample_size truncates game lists.** The cached model-inputs have 10 games per team; sample_size < 10 uses fewer.
- **Search space constraints validated.** z_medium < z_high and min_z_threshold <= z_medium, otherwise penalty.

## Future Work

- Multi-objective optimization (beat_rate + roi simultaneously)
- Model architecture search (different formulas, not just parameters)
- Jupyter notebooks for EDA
