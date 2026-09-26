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

    if actual_yards == line:
        # Push: an exact tie is neither a win nor a loss for either side. Excluded from
        # win/loss math (real_bets in metrics.py filters on hit is not None) while still
        # counting as gradeable (filtered on actual_yards is not None) — same convention
        # nba/backtest/grader.py establishes with its explicit "PUSH" state, adapted here
        # without needing a separate string state since this module has no v2_result/v1_result
        # field to carry it in.
        result["hit"] = None
        result["hypothetical_hit"] = None
        return result

    actual_side = "OVER" if actual_yards > line else "UNDER"

    if recommendation in ("OVER", "UNDER"):
        result["hit"] = (actual_side == recommendation)
        result["hypothetical_hit"] = None
    elif recommendation == "NO_BET":
        # Recover which side computeQbVerdict actually chose even though NO_BET means the
        # edge/confidence bar wasn't cleared to recommend it. On the normal path the verdict
        # picks the side with the larger de-vigged edge, which can disagree with z_score's
        # sign alone (that sign-only logic is only what computeQbVerdict itself falls back to
        # when it can't de-vig the book's prices). Both z_score and win_probability are always
        # persisted in the snapshot, and win_probability is always the model's probability of
        # whichever side was actually chosen — so the chosen side was OVER iff a below-mean
        # line (z < 0, which per z-sign alone implies OVER) agrees with win_probability > 50
        # (i.e. the chosen side is the more-likely-per-z one).
        z = verdict.get("z_score")
        win_prob = verdict.get("win_probability")
        implied_side = None
        if z is not None and win_prob is not None:
            implied_side = "OVER" if (z < 0) == (win_prob > 50) else "UNDER"
        elif z is not None:
            # Defensive fallback for an older logged snapshot missing win_probability.
            implied_side = "OVER" if z < 0 else "UNDER"
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
