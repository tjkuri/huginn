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
tests/mlb/                       pytest test suite (201 tests)
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
- **Outcome-based pitch counting.** Pitches per PA vary by outcome: K/BB=5, OUT=4, hits=3, HBP=2. Defined in `PITCHES_PER_OUTCOME` in `mlb/config.py`.
- **Extra-inning ghost runner.** 10th+ inning half-innings start with `BaseState(second=True)` per MLB rules (2020+).
- **Mercy rule at 15 innings.** Games capped to prevent infinite loops in edge cases.
- **Single aggregate bullpen arm.** Each team's bullpen is one `PitcherStats` built from team-level relief aggregate stats (GS==0 or GS/G<0.2). Individual reliever selection is a future enhancement.
- **No GIDP in v1.** Ground-ball double plays not modeled; runners hold on generic outs (except sac fly).
- **MLB data layer uses external sources.** `pybaseball` provides season stats; `MLB-StatsAPI` provides schedules, lineups, and rosters.
- **Lazy imports protect tests.** Data fetch modules import external packages only when fetch functions are called, so mocked tests run without live network usage.
- **True handedness splits are resolved per PA.** `mlb/data/stats.py` fetches vs-LHP/vs-RHP batting splits and vs-LHB/vs-RHB pitching splits from FanGraphs leaderboards using month parameters (13=vs-left, 14=vs-right). In the game engine, batters facing a starter use the handedness split matching the starter's throwing hand; batters facing the aggregate bullpen arm use their overall rates instead of a platoon split. Pitchers always use their handedness-allowed split based on the batter's effective side (switch hitters bat opposite the pitcher). Minimum 30 PA (batter) or 20 BF (pitcher) required to trust a split; otherwise falls back to overall rates.
- **Name matching is v1 identity resolution.** pybaseball and MLB Stats API players are matched by normalized player name for now; no persistent ID map yet. The canonical normalizer is `normalize_name` in `mlb/utils/normalize.py` — imported everywhere, no local copies. Steps: strip Unicode accents (Agustín → agustin), remove periods from initials (J.C. → JC), drop trailing suffixes (Jr./Sr./II/III/IV).
- **Name lookup failures log a warning with the normalized form.** When a player lookup falls back to league average, `builder.py` logs: `No stats found for "<raw name>" (normalized: "<key>") — using league average`. This makes it easy to diagnose FanGraphs vs MLB API name mismatches.
- **Early season uses prior-year coverage.** In April, 2026 batting/pitching thresholds drop to 20 PA / 10 IP. Players still below those thresholds fall back to 2025 season stats before league-average fallback.
- **Player source tags are explicit.** Data-layer player payloads now carry split-aware `source` values such as `2026_split`, `2026_overall`, `2025_split`, `2025_overall`, or `league_avg`, and the CLI reports batter source counts in data notes.
- **GameContext carries lineup/starter provenance.** `GameContext` includes `away_lineup_source`, `home_lineup_source`, `away_starter_source`, and `home_starter_source` so the formatter can distinguish confirmed lineups from active-roster fallback and probable starters from roster-arm fallback.
- **Missing real-player data falls back to league average.** Unknown batters/pitchers log warnings and use league-average rates so game-context assembly does not fail.
- **Park factors are coarse annual inputs.** `mlb/data/park_factors.py` uses hardcoded approximate 2025 factors that should be refreshed annually.
- **CLI has two output modes.** `mlb/scripts/simulate_game.py` prints human-readable summaries by default and emits clean JSON arrays on stdout with `--json`.
- **CLI rich output is isolated from orchestration.** Terminal rendering lives in `mlb/scripts/format_output.py`; `simulate_game.py` stays focused on fetch/filter/run/serialize flow.
- **Data quality panel surfaces input provenance, not just stat fallbacks.** Besides prior-year / league-average player data and placeholder weather, the panel now warns when a lineup used fallback active-roster order or when a starter fell back to the first roster arm because no probable pitcher was listed.
- **JSON mode must keep stdout clean.** Logging/progress go to stderr or are suppressed so Yggdrasil can consume stdout directly.
- **Verbose simulation uses chunked execution.** The CLI batches simulations in 1000-sim chunks when `--verbose` is enabled so progress can be reported without changing engine internals.
- **SimulatedGame now carries inning scores.** `inning_scores = {'away': [...], 'home': [...]}` is populated during simulation so the CLI can render a sample line score without re-simulating.
- **Player projections include enough data for batting lines.** Aggregation now tracks doubles and HBP alongside hits/HR/BB/K so the Rich player tables can show 2B and compute AVG from simulated AB.
- **Synthetic smoke coverage is the fast end-to-end check.** `python -m mlb.scripts.test_smoke` validates the full pipeline without network access.
- **Calibration bug fixed in inning state.** Non-out events must preserve the current out count; resetting outs to zero made half-innings continue until three consecutive outs and inflated scoring to ~26 runs/game.
- **League-average neutral check is now the calibration baseline.** `python -m mlb.scripts.diagnose_calibration` should land around 8.5-9.5 total runs/game, ~70-80 PAs, and ~54 outs.
- **pybaseball is called with qual=1.** `batting_stats(season, qual=1)` and `pitching_stats(season, qual=1)` return all players with any PA/IP. The FanGraphs qualified-leaderboard default would silently drop most of the league. Our own PA/IP thresholds handle filtering.
- **pybaseball K%/BB% are already decimals.** `batting_stats()` and `pitching_stats()` return `K%` and `BB%` as decimals (e.g. 0.182), not percentages (18.2). Do not divide by 100.
- **FanGraphs leaderboard `Bat` is not batter handedness.** In pybaseball/FanGraphs batting tables, `Bat` is batting runs, not L/R/S. `stats.py` enriches cached player rows with real handedness from MLB's people API keyed off Fangraphs IDs.
- **Stats load errors propagate.** `load_schedule_and_stats` in the CLI does not swallow fetch exceptions — if pybaseball or MLB-StatsAPI fails, the error surfaces rather than silently producing all-league-average output.
- **PitcherSimStats extends PlayerSimStats for prop markets.** `aggregate.py` tracks pitcher outs/K/ER per simulation and computes IP, 5+K%, and QS% alongside the base fields. The formatter uses these directly; do not recompute from raw `pa_results`.
- **Pitcher display uses prop-market columns.** The player projections panel shows IP, K, 5+K%, ER, QS% — not rate stats like K/9 or ERA. This matches how sportsbooks present pitcher props.
- **Probable pitchers always override boxscore.** `fetch_todays_games()` captures `away_probable_pitcher` / `home_probable_pitcher` from the schedule API. `_resolve_lineup()` injects these into the lineup regardless of what the boxscore pitcher field says — the boxscore pre-game pitcher entries are unreliable (may list warmup/administrative arms, not the actual starter). Falls back to first roster arm only when the probable field is empty and there is no boxscore.
- **Runtime league averages are now live simulation inputs.** `ensure_runtime_league_averages()` runs lazily from `build_game_context()`. It prefers `computed_league_averages-{season}.json`, otherwise recomputes overall league rates from pybaseball `batting_stats(..., qual=1)` / `pitching_stats(..., qual=1)` and matchup rates from handedness split caches, then mutates `LEAGUE_AVERAGES` in place. If the live recompute fails, it falls back to the cached computed file if present, then to the hardcoded 2025 fallback constants in `config.py` with a warning.
- **Park factors: Savant → hardcoded → neutral fallback chain.** `fetch_park_factors(season)` makes two requests to Baseball Savant (`batSide=L` and `batSide=R`) to get handedness-specific park factors, then caches both sides together in `park_factors-{season}.json`. If one side request fails, falls back to the combined ("All") endpoint for that side. If all Savant requests fail, falls back to the hardcoded `PARK_FACTORS` table. Early in a season, Savant only has data for venues used so far; `get_park_factors(venue)` checks the hardcoded table before returning neutral 1.0. We now normalize known Savant venue aliases like `UNIQLO Field at Dodger Stadium` → `Dodger Stadium` and `loanDepot park` → `loanDepot Park`.
- **`requests` and `beautifulsoup4` are top-level imports in `park_factors.py` and `stats.py`.** Unlike pybaseball, these are not lazy-imported. Both are listed in `requirements.txt` and are expected to be installed.
- **Roster handedness comes from `person` hydration, not entry level.** `fetch_team_roster` requests `hydrate=person` so `batSide` and `pitchHand` are under `person`, not under the roster entry directly. The old entry-level lookup always returned `None` → defaulted to "R".

## MLB Caching

Only slow pybaseball scrapes are cached. Schedule, lineup, and roster data is always fetched fresh.

```
baseball_cache/
  raw_batting-2025.json           ← prior season, kept forever (historical data never changes)
  raw_pitching-2025.json          ← prior season, kept forever
  raw_batting-2026.json           ← current season, re-fetched if older than 6 hours
  raw_pitching-2026.json          ← current season, re-fetched if older than 6 hours
  raw_batting_vs_lhp-2025.json    ← LHB vs LHP batting splits, prior season, kept forever
  raw_batting_vs_lhp-2026.json    ← LHB vs LHP batting splits, current season, 6-hour TTL
  raw_batting_vs_rhp-2025.json    ← LHB vs RHP batting splits, prior season, kept forever
  raw_batting_vs_rhp-2026.json    ← LHB vs RHP batting splits, current season, 6-hour TTL
  raw_pitching_vs_lhb-2025.json   ← pitcher vs LHB splits, prior season, kept forever
  raw_pitching_vs_lhb-2026.json   ← pitcher vs LHB splits, current season, 6-hour TTL
  raw_pitching_vs_rhb-2025.json   ← pitcher vs RHB splits, prior season, kept forever
  raw_pitching_vs_rhb-2026.json   ← pitcher vs RHB splits, current season, 6-hour TTL
  computed_league_averages-2026.json  ← overall + matchup league averages, same TTL as current-season stats
  park_factors-2025.json          ← Baseball Savant park factors for prior season, kept forever
  park_factors-2026.json          ← current season Savant park factors, no TTL (delete to refresh)
  pybaseball/                     ← pybaseball's own HTTP/DataFrame cache
```

**TTL logic:** `STATS_CACHE_MAX_AGE_HOURS = 6` in `mlb/config.py`. Files for seasons prior to `SEASON` never expire. Current-season raw stat files (batting, pitching, all split variants, and computed league averages) are age-checked; prior seasons are kept forever. Park factor cache files have no TTL — delete to force a fresh Savant fetch.

**No merged/intermediate cache.** `fetch_batting_splits` and `fetch_pitching_splits` merge current + prior season data on every call (takes milliseconds). Raw overall and split files are cached separately; merging is lazy and fast.

**Cache invalidation:** Delete any of the raw stat files to force a fresh pybaseball scrape (overall or split variants). Delete `computed_league_averages-{season}.json` to force a fresh league-average recompute. Delete `park_factors-{season}.json` to force a fresh Savant fetch. Delete `baseball_cache/pybaseball/` to force pybaseball to re-download from FanGraphs.
