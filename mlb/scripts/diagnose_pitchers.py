"""Diagnostic: inspect probable-pitcher data and the current starter resolution path.

Usage:
    python -m mlb.scripts.diagnose_pitchers [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date


def _today_str() -> str:
    return date.today().isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose starting pitcher assignment.")
    parser.add_argument("--date", default=_today_str(), help="Date to check (YYYY-MM-DD).")
    args = parser.parse_args(argv)

    try:
        import statsapi
    except ImportError:
        print("ERROR: MLB-StatsAPI not installed. Run: pip install MLB-StatsAPI", file=sys.stderr)
        return 1

    print(f"\n{'='*70}")
    print(f"STEP 1: Raw statsapi.schedule() response for {args.date}")
    print(f"{'='*70}")

    raw_games = statsapi.schedule(date=args.date)
    if not raw_games:
        print(f"No games found for {args.date}.")
        return 0

    print(f"Found {len(raw_games)} games.\n")

    # Show all keys available in the first game
    print("ALL keys in game dict (first game):")
    for key in sorted(raw_games[0].keys()):
        print(f"  {key!r}: {raw_games[0][key]!r}")

    print(f"\n{'='*70}")
    print("STEP 2: Probable pitcher fields for each game")
    print(f"{'='*70}")

    for game in raw_games:
        away = game.get("away_name") or game.get("away_team", "Away")
        home = game.get("home_name") or game.get("home_team", "Home")
        print(f"\n{away} @ {home}  (game_id={game.get('game_id')})")

        # Print any field with 'pitch', 'starter', or 'probable' in the key name
        found_pitcher_fields = False
        for key in sorted(game.keys()):
            if any(term in key.lower() for term in ("pitch", "starter", "probable")):
                print(f"  {key}: {game[key]!r}")
                found_pitcher_fields = True
        if not found_pitcher_fields:
            print("  (no pitcher/starter/probable fields found)")

    print(f"\n{'='*70}")
    print("STEP 3: What the current code assigns as starting pitchers")
    print(f"{'='*70}")

    # Import lazily so this can still run without full project deps in some envs
    try:
        import logging
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

        from mlb.data.lineups import fetch_todays_games, fetch_game_lineup, build_default_lineup_from_roster
        from mlb.config import SEASON

        schedule = fetch_todays_games(date=args.date)
        for game in schedule:
            away = game.get("away_team", "Away")
            home = game.get("home_team", "Home")
            game_id = game.get("game_id")
            print(f"\n{away} @ {home}  (game_id={game_id})")

            # Show what the schedule dict carries (or doesn't) for pitchers
            print(f"  game_info['away_probable_pitcher']: {game.get('away_probable_pitcher')!r}")
            print(f"  game_info['home_probable_pitcher']: {game.get('home_probable_pitcher')!r}")

            # Show what fetch_game_lineup returns
            lineup = fetch_game_lineup(int(game_id)) if game_id else None
            if lineup is not None:
                print(f"  boxscore away_pitcher: {lineup.get('away_pitcher')}")
                print(f"  boxscore home_pitcher: {lineup.get('home_pitcher')}")
                print(f"  (lineup confirmed — boxscore data used)")
            else:
                print(f"  boxscore: no confirmed lineup yet")
                # Show what the roster fallback picks
                try:
                    _, away_p = build_default_lineup_from_roster(int(game.get("away_team_id", 0)), season=SEASON)
                    _, home_p = build_default_lineup_from_roster(int(game.get("home_team_id", 0)), season=SEASON)
                    print(f"  roster fallback away_pitcher: {away_p.get('name')!r}  (WRONG — first roster pitcher)")
                    print(f"  roster fallback home_pitcher: {home_p.get('name')!r}  (WRONG — first roster pitcher)")
                except Exception as exc:
                    print(f"  roster fallback error: {exc}")

    except Exception as exc:
        print(f"\nCould not run code-path check: {exc}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print("DIAGNOSIS COMPLETE")
    print("Use the sections above to compare:")
    print("- raw schedule probable-pitcher fields")
    print("- the normalized fetch_todays_games() payload")
    print("- confirmed-lineup vs roster-fallback starter assignment")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
