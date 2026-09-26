# NFL QB Verdict Backtest

Backtest engine for the NFL QB passing-yards verdict model in [Yggdrasil](../../yggdrasil). Reads logged prediction snapshots, grades each one against actual passing yards fetched from nflverse, and reports hit rates, confidence-tier breakdowns, and mean absolute error.

> **No dollar EV or ROI anywhere in this engine — on purpose.** Unlike the NBA engine, this backtest deliberately reports only hit rates and mean absolute error, never a dollar figure. The underlying verdict model (in Yggdrasil) is a season-average-plus-opponent-adjustment model: it has no game-script, weather, or injury signal. Attaching a dollar EV or ROI number to that would imply a level of confidence the model can't actually back up. This was a deliberate call made when the model itself was built, not an oversight here — see Yggdrasil's own `JOURNAL.md` for the full reasoning if you want more context. If you're tempted to "helpfully" add a dollar figure back in, please don't — read the journal entry first and understand why it was left out.

## Usage

```bash
source venv/bin/activate

# Backtest — all logged predictions
python nfl/scripts/run_backtest.py

# Filtered views
python nfl/scripts/run_backtest.py --weeks-back 2    # last 2 distinct (season, week)s
python nfl/scripts/run_backtest.py --player mahomes  # filter by player name substring
python nfl/scripts/run_backtest.py --json            # raw JSON output

# Override cache location
python nfl/scripts/run_backtest.py --cache /path/to/cache

# Run tests
python -m pytest tests/nfl/ -v
```

## Methodology

The backtest grades each logged prediction snapshot against the QB's actual passing yards for that week:

- **Real bets (`OVER`/`UNDER` recommendations)** — graded `hit`/`miss` against whichever side of the line actually happened.
- **`NO_BET` predictions** — not graded as real bets (excluded from the hit-rate math), but a `hypothetical_hit` is still computed by recovering which side the model actually would have leaned, so you can see whether the NO_BET gate is filtering out genuine noise (hypothetical hit rate near 50%) or throwing away real edge.
- **Pushes** (`actual_yards == line` exactly) — excluded from both `hit` and `hypothetical_hit` (neither a win nor a loss for either side), but still counted as gradeable.
- **`abs_error`** — mean absolute error between `adjusted_mean` (the model's point projection) and actual yards, independent of the line or recommendation.

### NO_BET hypothetical side recovery

`computeQbVerdict` (Yggdrasil's `utils/nflMath.js`) normally picks the side with the larger de-vigged edge between the book's fair-over/fair-under probabilities, only falling back to the sign of the z-score when it can't de-vig the book's prices. Since only `z_score` and `win_probability` are persisted in each logged snapshot (not the raw book edges), the chosen side is recovered as: the chosen side was `OVER` if and only if `(z_score < 0) == (win_probability > 50)`. `win_probability` is always the model's probability of whichever side was actually chosen, so this recovers the true chosen side even when it disagrees with the z-score's sign alone.

## Module Structure

```
nfl/
  config.py                Yggdrasil cache path, nflverse stats URL template
  data/
    loader.py              Reads logged prediction snapshots (nflLogger.js), filters by weeks_back/player
    results.py             Fetches nflverse weekly player stats CSV, finds actual passing yards
  backtest/
    grader.py              Grades predictions: hit, hypothetical NO_BET hit, abs_error
    metrics.py             Aggregates stats: hit rate, by-confidence, NO_BET hypothetical, mean abs error
    report.py               Rich terminal output
  scripts/
    run_backtest.py        CLI entry point (--weeks-back, --player, --json, --cache)
tests/nfl/                 pytest suite (30 tests)
```

## Yggdrasil Interface

Huginn reads logged prediction snapshots from Yggdrasil's cache directory and fetches actual results directly from nflverse (no separate "results" file from Yggdrasil, unlike NBA):

- **Input:** `<cache>/{season}-wk{week}-{player_id}.json` — one snapshot per logged prediction, written by Yggdrasil's `services/nflLogger.js`.
- **Actuals:** fetched live per season from nflverse's `stats_player_week_{season}.csv` release asset (see `nfl/config.py`) — not read from Yggdrasil's cache at all.
