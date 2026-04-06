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

**Sports prediction backtest and simulation engine.**

Named after Odin's raven who flies out across the world to gather knowledge and report back.

Huginn currently has two engines:

- **[NBA](nba/)** — prediction backtest and parameter optimizer. Reads cached predictions from [Yggdrasil](../yggdrasil) and measures how well the model beat the closing line over time.
- **[MLB](mlb/)** — Monte Carlo game simulation engine. Fetches live rosters, stats, and lineups, simulates thousands of games via the odds-ratio method, and produces betting-line projections and player prop estimates.

## Getting Started

**Requirements:** Python 3.11+. The NBA engine reads from [`../yggdrasil/cache/`](../yggdrasil) — clone that repo as a sibling if you need NBA backtest data.

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd huginn

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Verify everything works
python -m pytest tests/ -v        # all tests
python -m mlb.scripts.test_smoke  # MLB end-to-end smoke test (no network needed)
```

## Usage

```bash
source venv/bin/activate

# NBA backtest
python nba/scripts/run_backtest.py
python nba/scripts/run_backtest.py --days 7 --team Lakers

# NBA parameter optimizer
python nba/scripts/run_optimizer.py --trials 500 --target beat_rate

# MLB simulation — today's games (rich terminal output)
python -m mlb.scripts.simulate_game

# MLB simulation — specific date or team
python -m mlb.scripts.simulate_game --date 2026-04-04 --team Yankees

# MLB simulation — JSON output (for Yggdrasil consumption)
python -m mlb.scripts.simulate_game --json

# Run all tests
python -m pytest tests/ -v
```

See [`nba/README.md`](nba/README.md) and [`mlb/README.md`](mlb/README.md) for full usage and methodology.
