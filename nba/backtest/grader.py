def grade_game(game):
    """
    Grade a merged game dict from the loader.

    Adds: v2_result, v2_miss, v1_direction, v1_result, v1_miss.
    Returns a new dict (does not mutate input).
    """
    graded = dict(game)

    # -- V2 grading ----------------------------------------------------------
    rec = game.get("opening_recommendation")
    v2_result = None

    if rec in ("O", "U"):
        actual = game["actual_total"]
        dk_line = game["opening_dk_line"]
        if actual == dk_line:
            v2_result = "PUSH"
        elif rec == "O":
            v2_result = "WIN" if actual > dk_line else "LOSS"
        else:
            v2_result = "WIN" if actual < dk_line else "LOSS"

    graded["v2_result"] = v2_result
    graded["v2_miss"] = (
        game["actual_total"] - game["projected_total"]
        if game.get("projected_total") is not None
        else None
    )

    # -- V1 grading ----------------------------------------------------------
    v1_line = game.get("v1_line")
    dk_line = game["opening_dk_line"]
    v1_gap = (v1_line - dk_line) if v1_line is not None else None

    v1_direction = None
    v1_result = None

    if v1_gap is not None and v1_gap != 0:
        v1_direction = "O" if v1_gap > 0 else "U"
        actual = game["actual_total"]
        if actual == dk_line:
            v1_result = "PUSH"
        elif v1_direction == "O":
            v1_result = "WIN" if actual > dk_line else "LOSS"
        else:
            v1_result = "WIN" if actual < dk_line else "LOSS"

    graded["v1_direction"] = v1_direction
    graded["v1_result"] = v1_result
    graded["v1_miss"] = (
        game["actual_total"] - v1_line
        if v1_line is not None
        else None
    )

    # -- Beat-the-book comparison ----------------------------------------------
    dk_miss = game["actual_total"] - game["opening_dk_line"]
    dk_abs = abs(dk_miss)

    v2_abs_miss = abs(graded["v2_miss"]) if graded["v2_miss"] is not None else None
    v1_abs_miss = abs(graded["v1_miss"]) if graded["v1_miss"] is not None else None

    graded["dk_miss"] = dk_miss
    graded["v2_abs_miss"] = v2_abs_miss
    graded["v1_abs_miss"] = v1_abs_miss
    graded["v2_beat_book"] = (v2_abs_miss < dk_abs) if v2_abs_miss is not None else None
    graded["v1_beat_book"] = (v1_abs_miss < dk_abs) if v1_abs_miss is not None else None

    return graded


def grade_all(games):
    """Grade every game in the list. Returns list of enriched dicts."""
    return [grade_game(g) for g in games]
