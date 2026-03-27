"""
Fitness function: evaluate a candidate config against historical data.

Loads model-inputs from cache, recomputes projections using candidate config,
grades against actuals, and returns metrics.
"""
from data.model_inputs_loader import load_model_inputs
from models.v2_current import (
    compute_my_line,
    compute_confidence_and_ev,
    compute_recommendation,
)
from backtest.grader import grade_all
from backtest.metrics import compute_metrics


def evaluate(cache_dir, cfg, days=None):
    """
    Run the full pipeline with a candidate config.
    Returns the same metrics dict as compute_metrics.
    """
    raw_games = load_model_inputs(cache_dir, days=days)

    predicted_games = []
    for raw in raw_games:
        result = compute_my_line(
            raw["home_games"], raw["away_games"], raw["date"], cfg
        )

        my_line = result["my_line"]
        sd_total = result["sd_total"]
        dk_line = raw["opening_dk_line"]

        if my_line is None:
            continue

        discrepancy = my_line - dk_line
        conf = compute_confidence_and_ev(discrepancy, sd_total, cfg)
        rec = compute_recommendation(
            my_line, dk_line, conf["z_score"], conf["expected_value"], cfg
        )

        predicted_games.append({
            "date": raw["date"],
            "game_id": raw["game_id"],
            "home_team": raw["home_team"]["name"],
            "away_team": raw["away_team"]["name"],
            "projected_total": round(my_line, 1),
            "sd_total": round(sd_total, 4) if sd_total else None,
            "opening_dk_line": dk_line,
            "opening_z_score": conf["z_score"],
            "opening_confidence": conf["confidence"],
            "opening_recommendation": rec,
            "opening_win_prob": None,
            "opening_ev": conf["expected_value"],
            "v1_line": None,
            "actual_total": raw["actual_total"],
            "went_to_ot": raw["went_to_ot"],
        })

    graded = grade_all(predicted_games)
    return compute_metrics(graded)
