#!/usr/bin/env python3
"""
Huginn NBA Backtest CLI.

Usage:
    python scripts/run_backtest.py                    # all data
    python scripts/run_backtest.py --days 7           # last 7 days
    python scripts/run_backtest.py --team Lakers       # filter by team
    python scripts/run_backtest.py --json              # raw JSON output
    python scripts/run_backtest.py --cache /path/to/cache  # override cache dir
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nba.config import YGGDRASIL_CACHE
from nba.data.loader import load_all_days
from nba.backtest.grader import grade_all
from nba.backtest.metrics import compute_metrics
from nba.backtest.report import print_report


def main():
    parser = argparse.ArgumentParser(description="NBA Backtest")
    parser.add_argument("--days", type=int, default=None,
                        help="Limit to last N days of data")
    parser.add_argument("--team", type=str, default=None,
                        help="Filter by team name (case-insensitive substring)")
    parser.add_argument("--json", action="store_true", dest="json_mode",
                        help="Output raw JSON instead of formatted tables")
    parser.add_argument("--cache", type=str, default=None,
                        help="Override cache directory path")
    args = parser.parse_args()

    cache_dir = Path(args.cache) if args.cache else YGGDRASIL_CACHE

    if not cache_dir.exists():
        print(f"Error: cache directory not found: {cache_dir}", file=sys.stderr)
        sys.exit(1)

    games = load_all_days(cache_dir, days=args.days, team=args.team)
    graded = grade_all(games)
    metrics = compute_metrics(graded)

    if args.json_mode:
        print(json.dumps({"games": graded, "metrics": metrics}, indent=2))
    else:
        print_report(graded, metrics)


if __name__ == "__main__":
    main()
