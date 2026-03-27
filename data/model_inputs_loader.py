"""Load model-inputs + results from Yggdrasil cache."""
import json
from pathlib import Path
from data.loader import _is_date_complete


def load_model_inputs(cache_dir, days=None):
    """
    Load model-inputs, results, and predictions (for DK lines), merge by game ID.

    Returns list of dicts with: date, game_id, home_team, away_team,
    home_games, away_games, actual_total, went_to_ot, opening_dk_line.
    """
    cache_path = Path(cache_dir)
    input_files = sorted(cache_path.glob("????-??-??-nba-model-inputs.json"))

    if days:
        input_files = input_files[-days:]

    all_games = []

    for input_file in input_files:
        date_str = input_file.name[:10]
        results_file = cache_path / f"{date_str}-nba-results.json"
        pred_file = cache_path / f"{date_str}-nba-predictions.json"

        if not results_file.exists():
            continue

        with open(results_file) as f:
            results = json.load(f)

        if not _is_date_complete(results):
            continue

        results_by_id = {str(r["game_id"]): r for r in results}

        dk_lines = {}
        if pred_file.exists():
            with open(pred_file) as f:
                predictions = json.load(f)
            for pred in predictions.get("games", []):
                dk_lines[str(pred["game_id"])] = pred.get("opening_dk_line")

        with open(input_file) as f:
            model_inputs = json.load(f)

        for mi in model_inputs:
            game_id = str(mi["id"])
            result = results_by_id.get(game_id)
            if not result:
                continue

            dk_line = dk_lines.get(game_id)
            if dk_line is None:
                continue

            all_games.append({
                "date": date_str,
                "game_id": game_id,
                "home_team": mi["home_team"],
                "away_team": mi["away_team"],
                "home_games": mi["home_games"],
                "away_games": mi["away_games"],
                "actual_total": result["actual_total"],
                "went_to_ot": result.get("went_to_ot", False),
                "opening_dk_line": dk_line,
            })

    return all_games
