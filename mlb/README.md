# MLB Simulation Engine

Monte Carlo simulation engine for MLB game prediction.

## Module Structure

```
mlb/
  config.py          Enums (Hand, Outcome, WindDirection), league averages,
                     season settings, simulation defaults, cache config
  data/
    models.py        Typed dataclasses: BatterStats, PitcherStats, ParkFactors,
                     Weather, Lineup, GameContext, BaseState, GameState,
                     PAResult, SimulatedGame, SimulationResult, PlayerSimStats
    cache.py         File-based JSON caching with TTL expiry
  engine/
    probabilities.py Odds Ratio PA probability tables with park/weather adjustments
    simulate.py      Game simulation: outcome sampling, baserunner advancement,
                     pitcher substitution, half-inning loop, full 9-inning game
  scripts/           CLI entry points (future)
```

## How It Works

### Plate Appearance Probability

For each plate appearance, we combine batter and pitcher split stats using the
**Odds Ratio method** (Tom Tango, "The Book") against league-average baselines:

```
P(outcome | matchup) = odds(P_batter) * odds(P_pitcher) / odds(P_league)
```

This produces a raw probability table (K%, BB%, HBP%, 1B%, 2B%, 3B%, HR%, OUT%)
which is then adjusted for park factors (by batter handedness) and weather
(temperature and wind), then normalized to sum to 1.0.

### Game Simulation

A full game is simulated by `simulate_game(game_context, league_averages, seed)`:

1. Loop through innings 1–15 (mercy rule cap)
2. For each half-inning, cycle through the batting order until 3 outs
3. Each PA: build probability table → sample outcome → advance baserunners
4. Walk-offs end the bottom half-inning immediately when home takes the lead
5. Extra innings (10th+) start with a runner on 2nd (MLB ghost runner rule)
6. After top of 9th+: skip bottom if home leads

Pitcher substitutions happen automatically when the starter exceeds pitch
count (default 100), 9 innings, or 8 runs allowed.

### Reproducibility

All randomness flows through `np.random.Generator` passed explicitly to every
function. Set `seed` in `simulate_game()` to reproduce any game exactly. This
also enables parallel execution with independent RNG streams.

## Key Design Decisions

- **No global random state.** RNG is threaded through every function.
- **Frozen BaseState.** Each PA produces a new BaseState; never mutated.
- **Bullpen in order.** Relievers used sequentially — handedness matching is a future enhancement.
- **No GIDP in v1.** Ground-ball double plays not modeled (future enhancement).
- **Sac fly approximation.** Runner on 3rd scores 50% of the time on a generic out with < 2 outs.
- **Stdlib + numpy only.** No additional dependencies beyond what's in requirements.txt.

## Running Tests

```bash
source venv/bin/activate

# All MLB tests
python -m pytest tests/mlb/ -v

# Simulation engine only
python -m pytest tests/mlb/test_simulate.py -v

# Full suite
python -m pytest tests/ -v
```
