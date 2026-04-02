# MLB Monte Carlo Simulation Engine

Part of the Huginn sports prediction platform.

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
  engine/            Simulation engine (future)
  scripts/           CLI entry points (future)
```

## How It Works (planned)

For each plate appearance, we combine batter and pitcher split stats using the
Odds Ratio method against league-average baselines to produce a probability
table (K%, BB%, HBP%, 1B%, 2B%, 3B%, HR%, OUT%). We adjust for park factors
(by batter handedness) and weather, normalize to sum to 1.0, then sample from
the distribution. Full 9-inning games are simulated with batting orders,
baserunner tracking, and pitcher substitutions. Run 10,000 times per game to
get distributions for total runs, win probability, and player props.

## Key Design Decisions

- **Enums for type safety:** `Hand`, `Outcome`, `WindDirection` are `str` enums
  — interoperable with plain strings but catch typos at construction time.
- **Handedness-split park factors:** `ParkFactors` has separate multiplier dicts
  for LHB and RHB because park effects differ by batter handedness.
- **Two-tier rate validation:** League averages validated to 0.001 tolerance at
  import time. Player stats validated to 0.02 tolerance in `__post_init__`.
- **Switch hitter resolution:** `resolve_batter_hand()` maps switch hitters to
  the opposite side of the pitcher.
- **Immutable BaseState:** Frozen dataclass — each PA produces a new state
  rather than mutating in place.
