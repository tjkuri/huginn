#!/usr/bin/env python3
"""
Huginn NFL QB Verdict Backtest CLI.

Usage:
    python nfl/scripts/run_backtest.py                  # all logged predictions
    python nfl/scripts/run_backtest.py --weeks-back 2    # last 2 distinct (season, week)s
    python nfl/scripts/run_backtest.py --player mahomes  # filter by player name substring
    python nfl/scripts/run_backtest.py --json            # raw JSON output
    python nfl/scripts/run_backtest.py --cache /path      # override cache dir
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nfl.config import YGGDRASIL_CACHE
from nfl.data.loader import load_predictions
from nfl.backtest.grader import grade_all
from nfl.backtest.metrics import compute_metrics
from nfl.backtest.report import print_report


def main():
    parser = argparse.ArgumentParser(description="NFL QB Verdict Backtest")
    parser.add_argument("--weeks-back", type=int, default=None,
                        help="Limit to the last N distinct (season, week) snapshots")
    parser.add_argument("--player", type=str, default=None,
                        help="Filter by player name (case-insensitive substring)")
    parser.add_argument("--json", action="store_true", dest="json_mode",
                        help="Output raw JSON instead of a formatted report")
    parser.add_argument("--cache", type=str, default=None,
                        help="Override cache directory path")
    args = parser.parse_args()

    cache_dir = Path(args.cache) if args.cache else YGGDRASIL_CACHE

    if not cache_dir.exists():
        print(f"Error: cache directory not found: {cache_dir}", file=sys.stderr)
        sys.exit(1)

    predictions = load_predictions(cache_dir, weeks_back=args.weeks_back, player=args.player)
    graded = grade_all(predictions)
    metrics = compute_metrics(graded)

    if args.json_mode:
        print(json.dumps({"predictions": graded, "metrics": metrics}, indent=2))
    else:
        print_report(graded, metrics)


if __name__ == "__main__":
    main()
