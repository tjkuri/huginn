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
# Output shape: {"games": [...], "global_source_statuses": [...]}
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
Model-derived betting views from the simulation set. Moneyline is shown as no-vig American odds; totals and team totals are shown as model over/under probabilities at common lines.

| Market | Example output |
|--------|----------------|
| Moneyline (no-vig American) | NYY -118 / BOS +118 |
| Total 8.5 | Over 52.1% / Under 47.9% |
| Team total (away) 4.5 | Over 48.7% / Under 51.3% |
| Run line (±1.5) | Favorite cover 43.8% / Underdog cover 56.2% |

### Line Score
A single randomly selected simulation from the run set showing the inning-by-inning breakdown. This is not an average or median game.

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
The CLI reports data quality in two layers:

- **Run-wide data quality** — shared batting/pitching season availability and handedness split availability across the full Marcel window, including whether each source was fetched fresh, loaded from cache, or degraded out of the blend.
- **Per-game data quality** — lineup confirmation vs roster fallback, starter provenance, bullpen fallback, park-factor provenance, placeholder weather, and player-level fallback notes for only the matchup being shown.

Player source tags still reflect Marcel blend depth (`marcel_3yr`, `marcel_2yr`, `marcel_1yr`, `league_avg`).

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
6. Pitcher substitution: starter exits at 100 pitches, 6 runs allowed, or 8.0 innings pitched

### Aggregation

After N simulations (default 10,000):
- **Run distributions** — mean, std, median, min/max, and discrete frequency distributions for each team and the total
- **Win probability** — fraction of sims each team won
- **Betting lines** — no-vig American moneyline plus run-line / total / team-total probabilities
- **Player stats** — per-game averages and distributions across all sim innings where that player appeared

### Data Sources

| Source | Used for |
|--------|----------|
| MLB Stats API | Season batting/pitching leaderboards (overall stats and handedness `statSplits`), today's schedule, confirmed lineups, roster |
| MLB People API | Player handedness enrichment for leaderboard rows |
| Baseball Savant | Handedness-split park factors fetched at runtime and cached |
| `mlb/data/park_factors.py` | Hardcoded 2025 fallback park factors when Savant data is unavailable |
| `mlb/config.py` `LEAGUE_AVERAGES` | Last-resort fallback only; overwritten at runtime by fetched matchup rates |

**League average fallback chain:** computed from current-season MLB Stats API overall rows plus MLB Stats API handedness `statSplits` → `computed_league_averages-{season}.json` cache → hardcoded constants in `config.py` with a warning.

**Player projection model:** batter and pitcher rates are Marcel-style 3-season regressions. Batters use recency weights (`5/4/3`), while pitchers use lower DIPS-inspired weights (`3/2/1`) because pitcher contact-quality outcomes are noisier than strikeout and walk rates. These weights are scaled by sample size and regressed toward a league baseline. Missing seasons naturally contribute zero weight. Completely unknown players still fall back to league-average rates.

### Current Data Flow

- MLB Stats API provides overall season batting and pitching rows for each player.
  - These overall rows are hand-agnostic.

- MLB Stats API `statSplits` provides handedness-specific rows.
  - Batters: `vs LHP`, `vs RHP`
  - Pitchers: `vs LHB`, `vs RHB`

- Those rows are converted into player profiles.
  - Batter profile:
    - overall
    - vs LHP
    - vs RHP
  - Pitcher profile:
    - overall
    - vs LHB
    - vs RHB

- We retain both volume and outcome information.
  - Batters use `PA`
  - Pitchers use `BF` / `TBF` when available
  - If a pitching row does not include `BF` / `TBF`, we estimate batters faced as:
    - `IP * 3 + H + BB + HBP`

- Using those player-level split rows, we derive league-average matchup baselines for:
  - `LHB vs RHP`
  - `RHB vs LHP`
  - `LHB vs LHP`
  - `RHB vs RHP`

- Matchup league averages are PA-weighted.
  - For each player split row, add that split's opportunities to the matchup bucket total.
  - Add `opportunities * outcome_rate` to each outcome total.
  - Divide each outcome total by total matchup opportunities.

- Marcel is applied to both overall and handedness split rows.
  - Overall projections blend up to three seasons of overall data.
  - Split projections blend up to three seasons of `vs LHP` / `vs RHP` or `vs LHB` / `vs RHB` data.
  - If split rows are unavailable, the simulation path falls back to the player's overall projected rates for that handedness selection.

- Merged player profiles carry Marcel-based source tags.
  - `marcel_3yr` = all three seasons contributed
  - `marcel_2yr` = two seasons contributed
  - `marcel_1yr` = one season contributed
  - `league_avg` = no real player data was available

### Key Design Decisions

- **No global random state.** `np.random.Generator` is threaded through every function — enables reproducibility (`--seed`) and parallel execution.
- **Frozen base state.** Each PA produces a new `BaseState` object; runners are never mutated in place.
- **Single aggregate bullpen arm.** Each team uses one bullpen pitcher built from team-level relief stats. Individual reliever usage is intentionally abstracted away.
- **No GIDP in v1.** Ground-ball double plays not modeled; runners hold on generic outs (except sac fly).
- **Sac fly approximation.** Runner on 3rd scores 50% of the time on a generic out with fewer than 2 outs.
- **Per-PA handedness splits.** Batters facing the starter use vs-LHP or vs-RHP rates; batters facing the bullpen use overall rates. Pitchers always use their vs-LHB or vs-RHB split based on the batter's effective side.
- **Runtime league averages preloaded once per run.** `preload_run_context()` resolves matchup-specific rates (LHB vs RHP, etc.) before any game context is assembled, and injects them into the global `LEAGUE_AVERAGES` mapping. Falls back through: valid computed cache → fresh recomputation → stale computed cache → hardcoded constants.

## Simulation Heuristics

The simulation uses a few explicit heuristics to keep runtime reasonable while producing plausible game flow.

### Pitch Count Estimates

Pitch count is tracked per plate appearance using fixed outcome-based estimates:

- `K` = 5 pitches
- `BB` = 5 pitches
- `HBP` = 2 pitches
- `OUT` = 4 pitches
- `1B` / `2B` / `3B` / `HR` = 3 pitches

These are rough MLB-average approximations, not pitch-by-pitch models.

### Starter Pull Rules

Starters are removed when any of these conditions is met:

- 100 pitches
- 6 runs allowed
- 8.0 innings pitched

Innings pitched is tracked from actual outs recorded, not inferred from pitch count.

### Bullpen Model

Each team gets a single bullpen arm named `{Team} Bullpen`.

- It is built from team-level relief pitching stats
- Relief rows are filtered using `GS == 0` or `GS / G < 0.2`
- The bullpen arm pitches the rest of the game once the starter is pulled
- If team bullpen data is unavailable, the model falls back to league-average rates

### What Is Not Modeled

- Individual reliever selection
- Bullpen handedness matchups
- Closer / setup / leverage roles
- Pinch hitting

## Module Structure

```
mlb/
  config.py              Enums (Hand, Outcome, WindDirection), league averages,
                         season settings, simulation defaults, cache config
  data/
    models.py            Typed dataclasses: BatterStats, PitcherStats, ParkFactors,
                         Weather, Lineup, GameContext, BaseState, GameState,
                         PAResult, SimulatedGame, SimulationResult, PlayerSimStats
    mlb_stats_api.py     Repo-local MLB Stats API client for overall + split leaderboards
    stats.py             MLB Stats API fetchers → BatterStats/PitcherStats builders
    team_codes.py        Shared team-name → stat-source code mapping
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
tests/mlb/               pytest suite
```
