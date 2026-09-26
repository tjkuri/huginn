import json
from pathlib import Path


def load_predictions(cache_dir, weeks_back=None, player=None):
    """
    Load logged NFL prediction snapshots (written by Yggdrasil's services/nflLogger.js)
    from cache_dir. Each file already contains season/week/player_name/etc. — no parsing
    from the filename needed.

    Args:
        cache_dir: Path to Yggdrasil's cache/nfl-predictions directory.
        weeks_back: If set, only include predictions from the last N distinct (season, week)
            values present in cache_dir.
        player: If set, filter to player_name containing this substring (case-insensitive).
    Returns: list of prediction dicts.
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return []

    predictions = []
    for f in sorted(cache_path.glob("*.json")):
        with open(f) as fh:
            predictions.append(json.load(fh))

    if player:
        needle = player.lower()
        predictions = [p for p in predictions if needle in p["player_name"].lower()]

    if weeks_back:
        weeks = sorted(set((p["season"], p["week"]) for p in predictions))
        keep = set(weeks[-weeks_back:])
        predictions = [p for p in predictions if (p["season"], p["week"]) in keep]

    return predictions
