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
mlb/                             MLB Monte Carlo simulation engine
  config.py                      Enums, league averages, season config, simulation defaults
  data/models.py                 Typed dataclasses for all MLB data structures
  data/cache.py                  File-based JSON caching with TTL
  data/stats.py                  pybaseball fetchers + BatterStats/PitcherStats builders
  data/lineups.py                MLB Stats API schedule, lineup, and roster fetchers
  data/park_factors.py           Hardcoded park-factor table + team→venue lookup
  data/weather.py                Neutral weather stub + indoor-park detection
  data/builder.py                Assembles GameContext from fetched data + fallbacks
  engine/probabilities.py        Odds ratio, park/weather adjustments, PA probability tables
  engine/simulate.py             Game simulation: outcome sampling, baserunners, 9-inning loop
  scripts/simulate_game.py       CLI entry point for schedule→data→simulation→JSON/report flow
  scripts/format_output.py       Rich terminal formatter for box score, summary, data quality, players
  scripts/test_smoke.py          Synthetic end-to-end smoke test (no APIs)
  scripts/diagnose_calibration.py League-average diagnostic for PA/run calibration
  scripts/diagnose_pitchers.py    Diagnostic for probable pitcher assignment vs roster fallback
tests/mlb/                       pytest test suite (183 tests)
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

# Run MLB tests
python -m pytest tests/mlb/ -v

# Install MLB data dependencies
pip install pybaseball MLB-StatsAPI

# Run MLB CLI smoke test
python -m mlb.scripts.test_smoke

# Run league-average calibration diagnostic
python -m mlb.scripts.diagnose_calibration

# Diagnose probable pitcher assignment vs roster fallback
python -m mlb.scripts.diagnose_pitchers

# Run MLB simulation tests
python -m pytest tests/mlb/test_simulate.py -v

# Run all tests
python -m pytest tests/ -v
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

## Key Conventions (MLB)

- **Enums for type safety.** Hand, Outcome, WindDirection are `str, Enum` — interoperable with plain strings.
- **League averages validated at import.** Config asserts each matchup sums to 1.0 (0.001 tolerance).
- **Player rates validated in __post_init__.** BatterStats and PitcherStats warn if rates don't sum to ~1.0 (0.02 tolerance).
- **Switch hitters resolve opposite pitcher.** `resolve_batter_hand(S, R) → L`, `resolve_batter_hand(S, L) → R`.
- **Park factors split by batter hand.** `ParkFactors.get_factors(hand)` returns the correct multiplier dict.
- **BaseState is frozen.** Each PA produces a new BaseState; never mutate in place.
- **MLB runtime depends on numpy plus data packages.** The engine still avoids web/service dependencies, but the CLI/data layer uses `pybaseball` and `MLB-StatsAPI`.
- **numpy RNG threading.** All randomness flows through `np.random.Generator` parameters — no global state. Enables reproducible simulations via seed.
- **Approximate pitch counting.** 4 pitches per PA as a rough average for pitch count tracking.
- **Extra-inning ghost runner.** 10th+ inning half-innings start with `BaseState(second=True)` per MLB rules (2020+).
- **Mercy rule at 15 innings.** Games capped to prevent infinite loops in edge cases.
- **Bullpen in order.** Relievers used sequentially from `lineup.bullpen`. Handedness-based selection is a future enhancement.
- **No GIDP in v1.** Ground-ball double plays not modeled; runners hold on generic outs (except sac fly).
- **MLB data layer uses external sources.** `pybaseball` provides season stats; `MLB-StatsAPI` provides schedules, lineups, and rosters.
- **Lazy imports protect tests.** Data fetch modules import external packages only when fetch functions are called, so mocked tests run without live network usage.
- **Overall stats stand in for true splits in v1.** `mlb/data/stats.py` uses season-level overall rates for all handedness matchups. Proper split scraping is a future enhancement.
- **Name matching is v1 identity resolution.** pybaseball and MLB Stats API players are matched by normalized player name for now; no persistent ID map yet.
- **Early season uses prior-year coverage.** In April, 2026 batting/pitching thresholds drop to 20 PA / 10 IP. Players still below those thresholds fall back to 2025 season stats before league-average fallback.
- **Player source tags are explicit.** Data-layer player payloads now carry `source` values of `2026`, `2025`, or `league_avg`, and the CLI reports batter source counts in data notes.
- **Missing real-player data falls back to league average.** Unknown batters/pitchers log warnings and use league-average rates so game-context assembly does not fail.
- **Park factors are coarse annual inputs.** `mlb/data/park_factors.py` uses hardcoded approximate 2025 factors that should be refreshed annually.
- **CLI has two output modes.** `mlb/scripts/simulate_game.py` prints human-readable summaries by default and emits clean JSON arrays on stdout with `--json`.
- **CLI rich output is isolated from orchestration.** Terminal rendering lives in `mlb/scripts/format_output.py`; `simulate_game.py` stays focused on fetch/filter/run/serialize flow.
- **JSON mode must keep stdout clean.** Logging/progress go to stderr or are suppressed so Yggdrasil can consume stdout directly.
- **Verbose simulation uses chunked execution.** The CLI batches simulations in 1000-sim chunks when `--verbose` is enabled so progress can be reported without changing engine internals.
- **SimulatedGame now carries inning scores.** `inning_scores = {'away': [...], 'home': [...]}` is populated during simulation so the CLI can render a sample line score without re-simulating.
- **Player projections include enough data for batting lines.** Aggregation now tracks doubles and HBP alongside hits/HR/BB/K so the Rich player tables can show 2B and compute AVG from simulated AB.
- **Synthetic smoke coverage is the fast end-to-end check.** `python -m mlb.scripts.test_smoke` validates the full pipeline without network access.
- **Calibration bug fixed in inning state.** Non-out events must preserve the current out count; resetting outs to zero made half-innings continue until three consecutive outs and inflated scoring to ~26 runs/game.
- **League-average neutral check is now the calibration baseline.** `python -m mlb.scripts.diagnose_calibration` should land around 8.5-9.5 total runs/game, ~70-80 PAs, and ~54 outs.
- **Stat caches must be non-empty to be trusted.** Empty merged or seasonal player caches are discarded and rebuilt instead of freezing the system at all-fallback output.
- **pybaseball K%/BB% are already decimals.** `batting_stats()` and `pitching_stats()` return `K%` and `BB%` as decimals (e.g. 0.182), not percentages (18.2). Do not divide by 100.
- **pybaseball batter handedness column is `Bat` (singular).** The column is `Bat`, not `Bats`. `stats.py` tries `Bat` then `Bats` as fallback.
- **Stats load errors propagate.** `load_schedule_and_stats` in the CLI does not swallow fetch exceptions — if pybaseball or MLB-StatsAPI fails, the error surfaces rather than silently producing all-league-average output.
- **PitcherSimStats extends PlayerSimStats for prop markets.** `aggregate.py` tracks pitcher outs/K/ER per simulation and computes IP, 5+K%, and QS% alongside the base fields. The formatter uses these directly; do not recompute from raw `pa_results`.
- **Pitcher display uses prop-market columns.** The player projections panel shows IP, K, 5+K%, ER, QS% — not rate stats like K/9 or ERA. This matches how sportsbooks present pitcher props.
- **Probable pitchers preferred over roster fallback.** `fetch_todays_games()` captures `away_probable_pitcher` / `home_probable_pitcher` from the schedule API. `_resolve_lineup()` uses these names when the boxscore lineup is not yet confirmed; it only falls back to the first roster arm when the probable field is empty. Schedule caches that predate these keys are invalidated and re-fetched automatically.

## Future Work

- Multi-objective optimization (beat_rate + roi simultaneously)
- Model architecture search (different formulas, not just parameters)
- MLB Monte Carlo simulation engine (odds ratio, baserunner tracking, pitcher substitutions)
- Jupyter notebooks for EDA
