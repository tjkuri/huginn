# NBA Backtest & Optimizer

Backtest and parameter optimization engine for the NBA prediction model in [Yggdrasil](../../yggdrasil). Reads cached predictions and game results, grades each game against the V2 model output and the opening DraftKings line, and searches for config parameters that maximize beat-the-book rate.

## Usage

```bash
source venv/bin/activate

# Backtest — all available data
python nba/scripts/run_backtest.py

# Filtered views
python nba/scripts/run_backtest.py --days 7
python nba/scripts/run_backtest.py --team Lakers
python nba/scripts/run_backtest.py --json            # raw JSON output

# Override cache location
python nba/scripts/run_backtest.py --cache /path/to/cache

# Parameter optimizer (Optuna)
python nba/scripts/run_optimizer.py --trials 500 --target beat_rate
python nba/scripts/run_optimizer.py --target avg_miss --export

# Run tests
python -m pytest tests/nba/ -v
```

## Methodology

The backtest grades each game along three dimensions:

- **V2 model** — Huginn's weighted-stats projection (port of `yggdrasil/utils/nbaMath.js`). Produces a z-score and a bet decision.
- **V1 baseline** — the opening DraftKings line shift as a naive signal.
- **Beat-the-book** — whether the V2 miss was strictly smaller than the book's miss. Uses strict less-than (ties don't count as a beat).

`avg_miss` is the mean absolute error across all games (including NO_BET).

### Model Parity

The Python model math produces identical outputs to Yggdrasil's `utils/nbaMath.js`. Parity is validated by integration tests comparing projections and z-scores against cached predictions.

### Optimizer

`run_optimizer.py` uses [Optuna](https://optuna.org/) to search the config parameter space. The fitness function evaluates each candidate config via temporal cross-validation (folds split by game date so future data never leaks into a fold's training window). Available targets: `beat_rate`, `avg_miss`, `avg_advantage`, `roi`.

## Module Structure

```
nba/
  config.py                Cache path, vig constants, model config reference
  data/
    loader.py              Reads paired prediction/result JSON, merges by game_id
    model_inputs_loader.py Reads model-inputs + results + predictions for fitness
  backtest/
    grader.py              Grades games: V2 + V1 + beat-the-book fields
    metrics.py             Aggregates stats: overall, breakdowns, book comparison
    report.py              Rich terminal output
  models/
    v2_current.py          Port of Yggdrasil's nbaMath.js (weighted stats, z-scores)
  optimizer/
    fitness.py             Evaluates candidate config; supports temporal CV
  scripts/
    run_backtest.py        CLI entry point (--days, --team, --json, --cache)
    run_optimizer.py       Optuna parameter search CLI (--trials, --target, --export)
tests/nba/                 pytest suite (132 tests)
```

## Yggdrasil Interface

Huginn reads from Yggdrasil's `cache/` directory and (optionally) writes an optimized config back:

- **Input:** `YYYY-MM-DD-nba-{predictions,results,model-inputs}.json`
- **Output:** `output/nba_config.json` — optimized parameters to copy into Yggdrasil
