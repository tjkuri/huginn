# MLB Simulation Methodology

How the engine turns player stats into win probabilities and prop estimates.
Two parts: building a probability table for each plate appearance, then running a full game with that table.

---

## Part 1: Plate Appearance Probability

For every single plate appearance we need a probability for each of 8 outcomes:
`K, BB, HBP, 1B, 2B, 3B, HR, OUT`.

The challenge: we have a batter's stats and a pitcher's stats, but both were measured against
league-average opponents. We can't just average them — that would double-count the baseline.
The odds ratio removes it.

### How player profiles are built

- `pybaseball` provides overall season batting and pitching leaderboards.
  - These are per-player rows and are hand-agnostic.

- FanGraphs legacy split pages provide handedness-specific player rows.
  - Batters: `vs LHP`, `vs RHP`
  - Pitchers: `vs LHB`, `vs RHB`

- From those sources we build split-aware player profiles.
  - Batter profile:
    - overall
    - vs LHP
    - vs RHP
  - Pitcher profile:
    - overall
    - vs LHB
    - vs RHB

- The model keeps both outcome rates and opportunity volume.
  - Batters use plate appearances (`PA`)
  - Pitchers use batters faced (`BF` / `TBF`) when available
  - If a pitching row does not include `BF` / `TBF`, batters faced is estimated as:

```
BF ≈ (IP × 3) + H + BB + HBP
```

This is a rough reconstruction:
- `IP × 3` = outs recorded
- `H + BB + HBP` = hitters who reached without recording an out

Player projections are built with Marcel 3-season regression. For each rate stat independently:

```
w1 = 5 × (PA_2026 / normalizer)
w2 = 4 × (PA_2025 / normalizer)
w3 = 3 × (PA_2024 / normalizer)
w_lg = regression_constant / normalizer

projected_rate = (w1×r_2026 + w2×r_2025 + w3×r_2024 + w_lg×league_avg)
                 / (w1 + w2 + w3 + w_lg)
```

The normalizer is **200 PA** for batters and **150 BF** for pitchers. A missing season contributes
weight zero and drops out naturally.

Batter regression constants (Tango's original values, in PA equivalents):

| Stat | Constant | Interpretation |
|------|----------|----------------|
| K%   | 150      | moderately stable |
| BB%  | 200      | stable |
| HR%  | 320      | high variance, regresses a lot |
| 1B%  | 200      | stable |
| 2B%  | 400      | less reliable than K/BB |
| 3B%  | 800      | very noisy, heavily regressed |
| HBP% | 400      | noisy |

OUT% is not independently projected — it is derived as `1 − sum(other rates)`.

Pitchers use a single regression constant of **150 BF** for all rate stats (v1).

**Concrete example — HR%, two seasons:**

A hitter with 400 PA in 2025 at HR%=0.040 and 60 PA in 2026 at HR%=0.020, league avg HR%=0.033:
- w1 = 5 × (60/200) = 1.5, w2 = 4 × (400/200) = 8.0, w_lg = 320/200 = 1.6
- projected = (1.5×0.020 + 8.0×0.040 + 1.6×0.033) / (1.5 + 8.0 + 1.6)
- = (0.030 + 0.320 + 0.053) / 11.1 ≈ **0.0363**

The 2025 season dominates because it has 6.7× more weight than the thin 2026 sample.

Overall projections use overall season rows. Handedness split projections use the matching
split rows (`vs LHP`, `vs RHP`, `vs LHB`, `vs RHB`) when available, with split PA as the
sample size. If split rows are missing, the matchup selection path falls back to the
player's overall projected profile.

### Step 1 — Pick the right league average

Rates differ meaningfully by handedness matchup. We use four separate baselines, one per
batter-hand × pitcher-hand combination. These are loaded at runtime: `build_game_context()`
computes matchup rates from current-season overall batting leaderboards plus FanGraphs
split leaderboards (vs-LHP / vs-RHP), computes weighted averages across all qualifying
players, and writes the result into `LEAGUE_AVERAGES`.
If the live fetch fails, the engine falls back to a cached `computed_league_averages-{season}.json`,
then to hardcoded 2025 constants in `mlb/config.py`.

Representative 2025 values (actual runtime values may differ slightly):

| Matchup | K% | BB% | HR% | OUT% |
|---|---|---|---|---|
| LHB vs RHP | 20.7% | 8.8% | 3.5% | 45.2% |
| RHB vs LHP | 21.1% | 8.5% | 3.7% | 45.2% |
| LHB vs LHP | 22.8% | 7.8% | 2.8% | 46.6% |
| RHB vs RHP | 22.5% | 7.6% | 3.0% | 46.3% |

Switch hitters bat opposite the pitcher (face RHP → bat left, face LHP → bat right).

These matchup league averages are not fetched as finished league tables. They are derived
from player split rows.

For each handedness bucket:

1. Take all player split rows that belong in that bucket
2. Weight each player's outcome rates by that split's opportunity volume
3. Sum those weighted outcomes across players
4. Divide by total opportunity volume in the bucket

In other words:

```
league_rate[outcome] =
    sum(player_split_opportunities × player_split_rate[outcome])
    / sum(player_split_opportunities)
```

For batter-derived matchup baselines, the opportunity volume is split `PA`.

### Step 2 — Odds Ratio

For each outcome independently (Tom Tango, *The Book*):

```
odds(p)     = p / (1 - p)

o_matchup   = (odds(p_batter) × odds(p_pitcher)) / odds(p_league)

p_matchup   = o_matchup / (1 + o_matchup)
```

**Why it works:** both the batter rate and pitcher rate were measured against the same
league-average opponent. Dividing by `odds(p_league)` cancels that shared baseline out,
leaving only the batter's individual skill and the pitcher's individual skill interacting.

**Concrete example — HR, RHB vs RHP:**

| Source | Rate | Odds |
|---|---|---|
| Batter (power hitter) | 6.0% | 0.060 / 0.940 = 0.0638 |
| Pitcher (HR-suppressor) | 2.0% | 0.020 / 0.980 = 0.0204 |
| League (RHB vs RHP) | 3.0% | 0.030 / 0.970 = 0.0309 |

```
o_matchup = (0.0638 × 0.0204) / 0.0309 = 0.0421

p_matchup = 0.0421 / 1.0421 ≈ 4.0%
```

The same batter against a league-average pitcher (3.0%) would land at ~6.0%.
Against this HR-suppressor he drops to ~4.0%. The league baseline is what makes the
math meaningful — it represents "what you'd expect from a random pitcher."

This runs independently for all 8 outcomes. The results **do not sum to 1.0** at this
point — that's intentional and handled in step 5.

### Step 3 — Park factors

Each outcome rate is multiplied by the park's factor for that batter's handedness
(LHB and RHB get different factors — Yankee Stadium's short right-field porch
affects left-handed power more than right-handed power).

```
adjusted['HR'] = raw['HR'] × park_factors['HR']   # e.g. 1.20 at Yankee for LHB
adjusted['2B'] = raw['2B'] × park_factors['2B']
# K, BB, 1B, 3B same way
# HBP and OUT are left alone — not park-dependent
```

Park factors come from Baseball Savant (separate L/R requests). When Savant data is
missing for a venue (common early in the season), the hardcoded 2025 table is used.

### Step 4 — Weather adjustments

Temperature and wind directly shift HR, 2B, and 3B rates. K, BB, HBP, 1B, and OUT are
not weather-modeled inputs; after that, the full table is clamped to a small positive
floor and normalized.

**Temperature** (air density effect, baseline 70°F):

```
temp_delta = (temperature_f - 70) / 10

HR_multiplier = 1.025 ^ temp_delta
2B_multiplier = 1.01  ^ temp_delta
3B_multiplier = 1.01  ^ temp_delta
```

| Temp | temp_delta | HR multiplier |
|---|---|---|
| 50°F | −2 | 1.025^−2 = 0.952 (−4.8%) |
| 70°F | 0 | 1.025^0 = 1.000 (no change) |
| 90°F | +2 | 1.025^2 = 1.051 (+5.1%) |

**Wind** (linear with speed, direction-dependent):

| Direction | HR | 2B |
|---|---|---|
| Out to CF | × (1 + 0.008 × mph) | × (1 + 0.003 × mph) |
| In from CF | × (1 − 0.008 × mph) | × (1 − 0.003 × mph) |
| Cross / calm | no change | no change |

At 15 mph blowing out: HR × 1.12 (+12%).

Temperature and wind stack multiplicatively. A 90°F day with 15 mph out:

```
HR × 1.051 (temp) × 1.12 (wind) = HR × 1.177   (+18% total)
```

Note: the `Weather` model stores `humidity_pct` but it is not currently applied.
Humidity has a real but small effect on air density (humid air is slightly less dense
than dry air); it is a candidate for a future calibration pass.

Note: weather is currently a stub — `get_game_weather()` always returns neutral
conditions (72°F, 5 mph calm) until a real weather API is integrated.

### Step 5 — Normalize

After the odds ratio, park, and weather steps the 8 rates no longer sum to 1.0.
Dividing each by the total fixes that:

```
total = sum of all 8 rates
p[outcome] = rate[outcome] / total
```

This is the final probability table. 8 numbers, sum exactly 1.0.

### Full pipeline summary

```
build_pa_probability_table(batter, pitcher, park_factors, weather, league_averages)
  │
  ├─ resolve_batter_hand()             switch hitters bat opposite the pitcher
  ├─ league_averages[(hand, hand)]     correct baseline for this handedness matchup
  ├─ park_factors.get_factors(hand)    LHB or RHB multipliers
  │
  ├─ compute_matchup_rates()           odds ratio on all 8 outcomes → raw rates
  ├─ apply_park_factors()              × park multipliers on K/BB/1B/2B/3B/HR
  ├─ apply_weather_adjustments()       × temp/wind on HR/2B/3B
  └─ normalize()                       ÷ sum → final probability table
```

This table is built fresh for every plate appearance in every simulation.

---

## Part 2: Game Simulation

With a probability table for any batter-pitcher matchup, we can simulate a full game.

### Sampling an outcome

Given the normalized table, one outcome is sampled by drawing a uniform random number
`r ∈ [0, 1)` and walking the cumulative distribution:

```
outcomes in order: K, BB, HBP, 1B, 2B, 3B, HR, OUT

cumulative = 0
for each outcome:
    cumulative += p[outcome]
    if r < cumulative → this is the outcome
```

For example with `p = {K:0.25, BB:0.08, ..., OUT:0.45}` and `r = 0.30`:
- after K: cumulative = 0.25 → 0.30 ≥ 0.25, keep going
- after BB: cumulative = 0.33 → 0.30 < 0.33 → **BB**

### Base state

The base state is a frozen struct with three booleans: `(first, second, third)`.
It is never mutated — every plate appearance produces a new base state object.

### Baserunner advancement

Each outcome has deterministic or probabilistic runner advancement rules:

| Outcome | Batter | Runners |
|---|---|---|
| K | out (+1 out) | no movement |
| BB / HBP | to 1st | force chain only (runner advances only if the base behind is occupied) |
| OUT | out (+1 out) | sac fly: runner on 3rd scores 50% of the time with < 2 outs |
| 1B | to 1st | runner on 3rd scores; runner on 2nd scores 90% / holds 10%; runner on 1st to 2nd 70% / to 3rd 30% |
| 2B | to 2nd | runners on 2nd/3rd score; runner on 1st to 3rd 60% / scores 40% |
| 3B | to 3rd | all runners score |
| HR | scores | all runners score |

The percentages (0.9, 0.7, 0.6, etc.) are fixed approximations of MLB base-running
tendencies. They are not player-specific — a future version could use individual
sprint speed or base-running grades.

**What is not modeled:** ground-ball double plays (GIDP). All outs produce exactly
one out regardless of base state. This slightly inflates baserunner counts in
situations where a GIDP would clear the bases.

### Half-inning loop

```
outs = 0
bases = empty (or 2nd occupied in extra innings)

while outs < 3:
    1. check if starter should be pulled (pitch count / outs / runs)
    2. get current pitcher (starter or bullpen)
    3. build PA probability table for this batter vs this pitcher
    4. sample outcome
    5. advance runners, add runs to score
    6. record PA result
    7. advance batting order (mod 9)
    8. update pitcher's pitch count, outs, runs allowed
    9. walk-off check: bottom 9th+, home leads → end immediately
```

### Pitcher substitution

The starter is replaced when any threshold is crossed:

| Threshold | Value |
|---|---|
| Pitch count | 100 |
| Runs allowed | 6 |
| Innings pitched | 8.0 |

Innings pitched is computed from actual outs recorded (`pitcher_outs / 3`), not
estimated from pitch count. When the starter is pulled, all three counters reset
to zero for the incoming bullpen arm.

Each team has a single aggregate bullpen arm built from team-level relief stats.
Once the starter exits, the bullpen arm pitches the rest of the game.

### Full game loop

```
inning = 1

while inning ≤ 15:                         15-inning mercy rule cap

    top half:
        bases = empty (or ghost runner on 2nd if inning ≥ 10)
        simulate_half_inning(away batting)
        record away runs this inning

        if inning ≥ 9 and home leads → game over (no bottom needed)

    bottom half:
        bases = empty (or ghost runner on 2nd if inning ≥ 10)
        simulate_half_inning(home batting)
        record home runs this inning

        walk-off fires inside the half-inning if home takes the lead

    after full inning:
        if inning ≥ 9 and scores not tied → game over

    inning += 1
```

### Aggregation across simulations

After N simulations (default 10,000):

- **Win probability** — fraction of games each team won
- **Run distributions** — mean, std, median, min/max, and discrete score frequencies for each team and the total
- **Betting lines** — derived from simulated game scores: no-vig moneyline, run-line cover probabilities, total probabilities, and team-total probabilities
- **Player stats** — each player's outcomes are tallied across every PA they appeared in across all simulations, then divided by games played to get per-game rates (AVG, OBP, HR%, K%, 2+H%, TB, etc.)
- **Pitcher props** — innings pitched, Ks, earned runs, 5+K probability, quality start probability
