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


def _compute_and_grade(raw_games, cfg):
    """
    Run the projection/grading pipeline over pre-loaded raw games.
    Returns the same metrics dict as compute_metrics.
    """
    sample_size = cfg.get("sample_size")

    predicted_games = []
    for raw in raw_games:
        home_games = raw["home_games"]
        away_games = raw["away_games"]

        if sample_size:
            home_games = home_games[:sample_size]
            away_games = away_games[:sample_size]

        result = compute_my_line(home_games, away_games, raw["date"], cfg)

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


def _extract_fitness(metrics):
    """
    Extract a flat dict of optimization-relevant values from a metrics dict.
    Returns penalty values for degenerate cases (no games or no v2 data).
    """
    if metrics["total_games"] == 0 or metrics["v2"] is None:
        return {
            "beat_rate": 0.0, "avg_miss": 999.0, "avg_advantage": -999.0,
            "roi": -100.0, "win_rate": 0.0, "total_bets": 0, "total_games": 0,
        }
    v2 = metrics["v2"]
    book = metrics.get("book_comparison") or {}
    book_v2 = book.get("v2") or {}
    return {
        "beat_rate": book_v2.get("beat_rate") if book_v2.get("beat_rate") is not None else 0.0,
        "avg_miss": v2.get("avg_miss") if v2.get("avg_miss") is not None else 999.0,
        "avg_advantage": (book_v2.get("avg_advantage")
                          if book_v2.get("avg_advantage") is not None else -999.0),
        "roi": v2.get("roi") if v2.get("roi") is not None else -100.0,
        "win_rate": v2.get("win_rate") if v2.get("win_rate") is not None else 0.0,
        "total_bets": v2.get("total_bets", 0),
        "total_games": metrics["total_games"],
    }


def evaluate(cache_dir, cfg, days=None):
    """
    Run the full pipeline with a candidate config.
    Returns the same metrics dict as compute_metrics.
    """
    raw_games = load_model_inputs(cache_dir, days=days)
    return _compute_and_grade(raw_games, cfg)


def evaluate_config(cfg, cache_dir, days=None):
    """
    Run the full pipeline with a candidate config and return a flat fitness dict.
    cfg is first (the thing that varies in optimization).
    Returns penalty values for degenerate cases.
    """
    metrics = evaluate(cache_dir, cfg, days=days)
    return _extract_fitness(metrics)
