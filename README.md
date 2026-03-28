```
    ///,        ////
     \  /,      /  >.
      \  /,   _/  /.
       \_  /_/   /.
        \__/_   <
        /<<< \_\_
       /,)^>>_._ \
       (/   \\ /\\\
            // ````
     ======((`=======
```

# Huginn

**NBA prediction backtest and analysis toolkit.**

Named after Odin's raven who flies out across the world to gather knowledge and report back.

Huginn reads prediction and result data from [Yggdrasil](../yggdrasil) (the NBA prediction backend), runs backtests, and will eventually house parameter optimization and model experimentation.

## Interface

Huginn talks to Yggdrasil through a single, narrow interface:

- **Input:** JSON files from Yggdrasil's `cache/` directory
- **Output:** Optimized config JSON (`output/nba_config.json`) that gets copied back into Yggdrasil

## Usage

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run backtest
python scripts/run_backtest.py                       # all data
python scripts/run_backtest.py --days 7              # last 7 days
python scripts/run_backtest.py --team Lakers          # filter by team
python scripts/run_backtest.py --json                 # raw JSON output
python scripts/run_backtest.py --cache /path/to/cache # override cache dir

# Run optimizer
python scripts/run_optimizer.py --trials 500 --target beat_rate
python scripts/run_optimizer.py --target avg_miss --export

# Run tests
python -m pytest tests/ -v
```

## Architecture

```
data/loader.py              Read + merge prediction/result JSON from cache
data/model_inputs_loader.py Read raw model-inputs for optimizer
        |
models/v2_current.py        V2 model math (weighted stats, projections, z-scores)
        |
backtest/grader.py          Grade each game: V2 model + V1 baseline + beat-the-book
        |
backtest/metrics.py         Aggregate into stats (overall, breakdowns, book comparison)
        |
backtest/report.py          Rich terminal output
optimizer/fitness.py        Evaluate candidate configs with temporal CV
```

## Model Parity

The Python model math produces identical outputs to Yggdrasil's `utils/nbaMath.js`. This is validated by integration tests comparing projections and z-scores against cached predictions.
