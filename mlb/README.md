# MLB Simulation Engine

Monte Carlo game simulation engine for MLB prediction. Fetches live player stats and lineups, simulates each game thousands of times using the odds-ratio method, and outputs win probabilities, totals projections, and player prop estimates.

## Usage

```bash
source venv/bin/activate

# Simulate today's games (rich terminal output)
python -m mlb.scripts.simulate_game

# Specific date or team filter
python -m mlb.scripts.simulate_game --date 2026-04-04
python -m mlb.scripts.simulate_game --team Yankees
python -m mlb.scripts.simulate_game --game-id 748123

# Control simulation count and reproducibility
python -m mlb.scripts.simulate_game --sims 10000 --seed 42

# JSON output (for Yggdrasil; stdout is clean, logs go to stderr)
python -m mlb.scripts.simulate_game --json

# Show progress during long runs
python -m mlb.scripts.simulate_game --verbose

# End-to-end smoke test (no network required)
python -m mlb.scripts.test_smoke

# Calibration diagnostic (league-average neutral game)
python -m mlb.scripts.diagnose_calibration

# Run tests
python -m pytest tests/mlb/ -v
```

## Reading the Output

The terminal output is organized into four panels:

### Game Summary
Win probabilities and projected run totals for the matchup.

```
NYY 54.2%  vs  BOS 45.8%
Total: 9.1 runs  (σ 3.2)
```

### Betting Lines
Model-derived lines with vig-free implied probabilities. Use these to compare against posted odds — a line discrepancy is where the model sees edge.

| Line | Away | Home |
|------|------|------|
| Moneyline (American) | -115 | +105 |
| Run line (±1.5) | +145 | -165 |
| Total (over/under) | O 8.5 -110 | U 8.5 -110 |
| Team total (away) | O 4.5 | U 4.5 |

### Line Score
A sample game from the simulation set showing the inning-by-inning breakdown. This is a single representative game, not an average.

### Player Projections
Per-player stats averaged across all simulations, matchup-adjusted for the specific opposing pitcher and park.

**Batters** — per-game averages from simulations:
- `AVG` / `OBP` — simulated batting average and on-base percentage
- `TB` — total bases per game (drives power prop decisions)
- `K%` — strikeout rate
- `2+H%` — probability of recording 2 or more hits
- `HR%` — home run rate

**Pitchers** — starter projections:
- `IP` — innings pitched (simulated, accounts for pitch count limits and run limits)
- `K` — strikeouts per game
- `5+K%` — probability of recording 5 or more strikeouts
- `ER` — earned runs allowed per game
- `QS%` — quality start probability (6+ IP, ≤3 ER)

### Data Quality Notes
Warns when players are using fallback data (prior-year stats or league averages instead of current-year data).

## Methodology

### Plate Appearance Probability: Odds Ratio

For each plate appearance we combine batter and pitcher split rates against league-average baselines using the **odds-ratio method** (Tom Tango, "The Book"):

```
P(outcome | matchup) = odds(P_batter) × odds(P_pitcher) / odds(P_league)
```

This produces a raw probability table across eight outcomes (K, BB, HBP, 1B, 2B, 3B, HR, OUT), which is then:
1. Adjusted for **park factors** (split by batter handedness)
2. Adjusted for **weather** (temperature, wind speed/direction)
3. Normalized to sum to 1.0

### Game Simulation

Each simulation runs a full game:

1. Loop through innings 1–15 (mercy rule cap)
2. Each half-inning: cycle the batting order until 3 outs
3. Each PA: build probability table → sample outcome → advance baserunners
4. Walk-offs: end the bottom half of the 9th or later the moment the home team takes the lead
5. Extra innings: start each half-inning with a runner on 2nd (MLB ghost runner rule, 2020+)
6. Pitcher substitution: starter exits when pitch count exceeds limit (~100), ERA gets too high, or 9 innings pitched

### Aggregation

After N simulations (default 10,000):
- **Run distributions** — mean, std, percentiles for each team and the total
- **Win probability** — fraction of sims each team won
- **Betting lines** — American moneyline, run line (±1.5), total, team totals with vig
- **Player stats** — per-game averages and distributions across all sim innings where that player appeared

### Data Sources

| Source | Used for |
|--------|----------|
| `pybaseball` | Season batting/pitching stats (split by handedness in a future version) |
| `MLB-StatsAPI` | Today's schedule, confirmed lineups, roster |
| `mlb/data/park_factors.py` | Hardcoded 2025 park factors (refresh annually) |
| League averages in `mlb/config.py` | Fallback for unknown players; calibration baseline |

**Fallback chain:** current-year stats → prior-year stats (early season) → league average. The data quality panel reports how many players are on each tier.

### Key Design Decisions

- **No global random state.** `np.random.Generator` is threaded through every function — enables reproducibility (`--seed`) and parallel execution.
- **Frozen base state.** Each PA produces a new `BaseState` object; runners are never mutated in place.
- **Bullpen in order.** Relievers are used sequentially from the roster — handedness matching is a future enhancement.
- **No GIDP in v1.** Ground-ball double plays not modeled; runners hold on generic outs (except sac fly).
- **Sac fly approximation.** Runner on 3rd scores 50% of the time on a generic out with fewer than 2 outs.
- **Overall stats stand in for splits.** True L/R split scraping is a future enhancement.

## Module Structure

```
mlb/
  config.py              Enums (Hand, Outcome, WindDirection), league averages,
                         season settings, simulation defaults, cache config
  data/
    models.py            Typed dataclasses: BatterStats, PitcherStats, ParkFactors,
                         Weather, Lineup, GameContext, BaseState, GameState,
                         PAResult, SimulatedGame, SimulationResult, PlayerSimStats
    cache.py             File-based JSON caching with TTL expiry
    stats.py             pybaseball fetchers → BatterStats/PitcherStats builders
    lineups.py           MLB Stats API schedule, lineup, and roster fetchers
    park_factors.py      Hardcoded park-factor table + team→venue lookup
    weather.py           Neutral weather stub + indoor-park detection
    builder.py           Assembles GameContext from fetched data + fallbacks
  engine/
    probabilities.py     Odds ratio, park/weather adjustments, PA probability tables
    simulate.py          Game simulation: outcome sampling, baserunner advancement,
                         pitcher substitution, half-inning loop, full 9-inning game
    aggregate.py         run_simulations, compute_player_stats (incl. PitcherSimStats),
                         compute_betting_lines, aggregate_simulations
  scripts/
    simulate_game.py     CLI entry point: fetch → filter → simulate → serialize/report
    format_output.py     Rich terminal formatter (box score, player tables, data notes)
    test_smoke.py        Synthetic end-to-end smoke test (no network required)
    diagnose_calibration.py  League-average neutral game calibration check
tests/mlb/               pytest suite (185+ tests)
```
