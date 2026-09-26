from nfl.data.results import fetch_season_rows, find_actual_yards


def grade_prediction(prediction, actual_yards):
    """
    Grade one prediction against its actual result.
    If actual_yards is None (game not yet played, or nflverse hasn't published it yet),
    every graded field comes back None — callers must exclude these from hit-rate math,
    they are "not yet gradeable," not losses.
    """
    verdict = prediction["verdict"]
    line = prediction["line"]
    recommendation = verdict.get("recommendation")

    result = dict(prediction)
    result["actual_yards"] = actual_yards

    if actual_yards is None:
        result["hit"] = None
        result["hypothetical_hit"] = None
        result["abs_error"] = None
        return result

    adjusted_mean = prediction.get("adjusted_mean")
    result["abs_error"] = abs(actual_yards - adjusted_mean) if adjusted_mean is not None else None

    actual_side = "OVER" if actual_yards > line else "UNDER"

    if recommendation in ("OVER", "UNDER"):
        result["hit"] = (actual_side == recommendation)
        result["hypothetical_hit"] = None
    elif recommendation == "NO_BET":
        # Same side logic computeQbVerdict itself falls back to when it can't de-vig: line
        # below the model's mean (negative z) implies OVER, above implies UNDER. This
        # recovers which way the model leaned even though NO_BET means the edge/confidence
        # bar wasn't cleared to actually recommend it.
        z = verdict.get("z_score")
        implied_side = ("OVER" if z < 0 else "UNDER") if z is not None else None
        result["hit"] = None
        result["hypothetical_hit"] = (actual_side == implied_side) if implied_side else None
    else:
        result["hit"] = None
        result["hypothetical_hit"] = None

    return result


def grade_all(predictions):
    """
    Grade a list of predictions, fetching each distinct season's nflverse stats once
    (not once per prediction) to avoid redundant network calls within a single run.
    """
    seasons = sorted(set(p["season"] for p in predictions))
    rows_by_season = {season: fetch_season_rows(season) for season in seasons}

    graded = []
    for p in predictions:
        rows = rows_by_season.get(p["season"], [])
        actual = find_actual_yards(rows, p["player_id"], p["week"])
        graded.append(grade_prediction(p, actual))
    return graded
