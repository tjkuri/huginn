def _hit_rate(results):
    """results: list of True/False. Returns None if empty."""
    if not results:
        return None
    return round(sum(1 for r in results if r) / len(results), 4)


def compute_metrics(graded_predictions):
    """
    Aggregate graded NFL predictions. No ROI or dollar figures anywhere — deliberate,
    matches the verdict card's own no-EV framing (this model has no game-script/weather/
    injury signal to back a real-money confidence claim; see Yggdrasil JOURNAL 2026-09-26).
    """
    if not graded_predictions:
        return {
            "total_predictions": 0, "gradeable": 0,
            "overall_hit_rate": {"n": 0, "hit_rate": None},
            "by_confidence": {}, "no_bet_hypothetical": {"n": 0, "hit_rate": None},
            "mean_abs_error": None, "mean_abs_error_by_scope": {},
        }

    gradeable = [g for g in graded_predictions if g["actual_yards"] is not None]
    real_bets = [g for g in gradeable if g["hit"] is not None]

    by_confidence = {}
    conf_levels = sorted({g["verdict"].get("confidence") for g in real_bets if g["verdict"].get("confidence")})
    for level in conf_levels:
        group = [g["hit"] for g in real_bets if g["verdict"].get("confidence") == level]
        by_confidence[level] = {"n": len(group), "hit_rate": _hit_rate(group)}

    no_bet_group = [g["hypothetical_hit"] for g in gradeable if g["hypothetical_hit"] is not None]

    abs_errors = [g["abs_error"] for g in gradeable if g["abs_error"] is not None]
    mean_abs_error = round(sum(abs_errors) / len(abs_errors), 2) if abs_errors else None

    mean_abs_error_by_scope = {}
    scopes = sorted({g.get("scope_used") for g in gradeable if g.get("scope_used")})
    for scope in scopes:
        errs = [g["abs_error"] for g in gradeable if g.get("scope_used") == scope and g["abs_error"] is not None]
        mean_abs_error_by_scope[scope] = round(sum(errs) / len(errs), 2) if errs else None

    return {
        "total_predictions": len(graded_predictions),
        "gradeable": len(gradeable),
        "overall_hit_rate": {"n": len(real_bets), "hit_rate": _hit_rate([g["hit"] for g in real_bets])},
        "by_confidence": by_confidence,
        "no_bet_hypothetical": {"n": len(no_bet_group), "hit_rate": _hit_rate(no_bet_group)},
        "mean_abs_error": mean_abs_error,
        "mean_abs_error_by_scope": mean_abs_error_by_scope,
    }
