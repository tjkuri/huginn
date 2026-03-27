# Huginn

NBA prediction backtest and analysis repo. Reads data from sibling repo `../yggdrasil/cache/`.

## Project Structure

```
config.py              Cache path, vig constants, model config reference
data/loader.py         Reads paired prediction/result JSON, merges by game_id
backtest/grader.py     Grades games: V2 (O/U/PUSH/NO_BET) + V1 baseline
backtest/metrics.py    Aggregates stats: overall, by-confidence/direction/gap, calibration
backtest/report.py     Rich terminal output
scripts/run_backtest.py  CLI entry point (--days, --team, --json, --cache)
tests/                 pytest test suite (49 tests)
```

## Commands

```bash
# Activate venv
source venv/bin/activate

# Run backtest
python scripts/run_backtest.py

# Run tests
python -m pytest tests/ -v
```

## Key Conventions

- **Parity with Yggdrasil is critical.** The Python backtest must produce identical metrics to `node scripts/backtest.js --json` in Yggdrasil. When changing grading or metrics logic, always verify parity.
- **game_id is always a string.** Both prediction and result files may store it as a number — coerce to string for matching.
- **avg_miss uses absolute values.** `mean(|v2_miss|)` for all games including NO_BET.
- **Calibration recomputes normalCDF from z_score.** Does NOT use the pre-computed `opening_win_prob` from predictions. Uses Abramowitz & Stegun approximation matching Yggdrasil's implementation.
- **V1 skips grading when gap is zero.** If `v1_line == opening_dk_line`, v1_direction and v1_result are None.
- **win_rate rounds to 4 decimals, roi to 2.** Matches JS `parseFloat((...).toFixed(N))`.

## Yggdrasil Reference

Key files in the sibling repo for understanding the data:
- `yggdrasil/services/nbaBacktest.js` — grading + metrics logic this repo mirrors
- `yggdrasil/config/nba.js` — model config values
- `yggdrasil/cache/` — `YYYY-MM-DD-nba-{predictions,results}.json` files

## Future Work

- `optimizer/` — Optuna parameter search
- `models/` — model experimentation
- Jupyter notebooks for EDA
