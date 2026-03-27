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
- **Output (future):** Optimized config JSON that gets copied back into Yggdrasil

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

# Run tests
python -m pytest tests/ -v
```

## Architecture

```
data/loader.py        Read + merge prediction/result JSON from cache
        |
backtest/grader.py    Grade each game: V2 model + V1 baseline
        |
backtest/metrics.py   Aggregate into stats (overall, by-confidence,
        |              by-direction, by-gap-size, calibration)
        |
backtest/report.py    Rich terminal output
```

## Metrics Parity

The Python backtest produces identical metrics to Yggdrasil's Node.js backtest (`node scripts/backtest.js`). This is validated by comparing JSON output field-by-field across all breakdowns.
