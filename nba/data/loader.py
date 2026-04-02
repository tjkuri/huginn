import json
from pathlib import Path


def load_all_days(cache_dir, days=None, team=None):
    """
    Scan cache_dir for days with paired prediction + results files.
    Returns list of merged game dicts.

    Args:
        cache_dir: Path to Yggdrasil cache directory.
        days: If set, only include the last N prediction files.
        team: If set, filter to games where home or away team contains
              this substring (case-insensitive).
    """
    cache_path = Path(cache_dir)
    pred_files = sorted(cache_path.glob("????-??-??-nba-predictions.json"))

    if days:
        pred_files = pred_files[-days:]

    all_games = []

    for pred_file in pred_files:
        date_str = pred_file.name[:10]
        results_file = cache_path / f"{date_str}-nba-results.json"

        if not results_file.exists():
            continue

        with open(results_file) as f:
            results = json.load(f)

        if not _is_date_complete(results):
            continue

        with open(pred_file) as f:
            predictions = json.load(f)

        if not predictions.get("games"):
            continue

        results_by_id = {str(r["game_id"]): r for r in results}

        for pred in predictions["games"]:
            result = results_by_id.get(str(pred["game_id"]))
            if not result:
                continue

            game = {
                "date": date_str,
                "game_id": str(pred["game_id"]),
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "projected_total": pred["projected_total"],
                "sd_total": pred.get("sd_total"),
                "opening_dk_line": pred["opening_dk_line"],
                "opening_z_score": pred.get("opening_z_score"),
                "opening_confidence": pred.get("opening_confidence"),
                "opening_recommendation": pred.get("opening_recommendation"),
                "opening_win_prob": pred.get("opening_win_prob"),
                "opening_ev": pred.get("opening_ev"),
                "v1_line": pred.get("v1_line"),
                "actual_total": result["actual_total"],
                "went_to_ot": result.get("went_to_ot", False),
            }

            if team:
                t = team.lower()
                if (t not in game["home_team"].lower()
                        and t not in game["away_team"].lower()):
                    continue

            all_games.append(game)

    return all_games


def _is_date_complete(results):
    """Check that all games on a date have final results."""
    if not results:
        return False
    return all(_is_record_locked(r) for r in results)


def _is_record_locked(record):
    """Check if a single game result is locked/final."""
    if not record:
        return False
    if record.get("status") == "final":
        return True
    # Legacy schema: no status field but has scores
    if (record.get("status") is None
            and record.get("home_score") is not None
            and record.get("away_score") is not None):
        return True
    return False
