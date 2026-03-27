import math
from config import VIG_RISK, VIG_WIN


def compute_metrics(graded_games):
    """
    Aggregate graded games into stats matching Yggdrasil's computeMetrics.
    Returns dict with: date_range, total_games, v2 (overall + breakdowns), v1.
    """
    if not graded_games:
        return {"date_range": None, "total_games": 0, "v2": None, "v1": None}

    dates = sorted(g["date"] for g in graded_games)
    date_range = {"from": dates[0], "to": dates[-1]}

    # ── V2 ─────────────────────────────────────────────────────────────────────
    v2_bets = [g for g in graded_games if g["v2_result"] is not None]
    v2_no_bets = [g for g in graded_games if g["v2_result"] is None]
    v2_overall = _record_stats(v2_bets, lambda g: g["v2_result"])

    # By confidence
    by_confidence = {}
    conf_levels = sorted(set(g["opening_confidence"] for g in v2_bets))
    for lvl in conf_levels:
        group = [g for g in v2_bets if g["opening_confidence"] == lvl]
        by_confidence[lvl] = _record_stats(group, lambda g: g["v2_result"])

    # NO_BET hypothetical
    no_bet_results = []
    for g in v2_no_bets:
        z = g.get("opening_z_score")
        if z is None or z == 0:
            continue
        hyp_dir = "O" if z > 0 else "U"
        actual = g["actual_total"]
        dk = g["opening_dk_line"]
        if actual == dk:
            no_bet_results.append("PUSH")
        elif hyp_dir == "O":
            no_bet_results.append("WIN" if actual > dk else "LOSS")
        else:
            no_bet_results.append("WIN" if actual < dk else "LOSS")

    wins = no_bet_results.count("WIN")
    losses = no_bet_results.count("LOSS")
    pushes = no_bet_results.count("PUSH")
    total_bets = wins + losses
    by_confidence["NO_BET_hypothetical"] = {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "total_bets": total_bets,
        "win_rate": round(wins / total_bets, 4) if total_bets > 0 else None,
        "roi": (round(((wins * VIG_WIN) - (losses * VIG_RISK))
                      / (total_bets * VIG_RISK) * 100, 2)
                if total_bets > 0 else None),
        "note": "NO BET (hypothetical)",
    }

    # By direction (O / U)
    by_direction = {}
    for direction in ("O", "U"):
        group = [g for g in v2_bets if g["opening_recommendation"] == direction]
        by_direction[direction] = _record_stats(group, lambda g: g["v2_result"])

    # By gap size (|projected - dk|)
    gap_buckets = [
        ("0-2", 0, 2),
        ("2-5", 2, 5),
        ("5+", 5, float("inf")),
    ]
    by_gap_size = {}
    for label, lo, hi in gap_buckets:
        group = [
            g for g in v2_bets
            if lo <= abs(
                (g["projected_total"] - g["opening_dk_line"])
                if g.get("projected_total") is not None
                   and g.get("opening_dk_line") is not None
                else 0
            ) < hi
        ]
        by_gap_size[label] = _record_stats(group, lambda g: g["v2_result"])

    # Average miss (all games, absolute values)
    v2_misses = [g["v2_miss"] for g in graded_games if g.get("v2_miss") is not None]
    avg_miss = (round(sum(abs(m) for m in v2_misses) / len(v2_misses), 2)
                if v2_misses else None)

    # Calibration (z-score buckets, actionable bets only)
    z_buckets = [
        ("0.5-0.8", 0.5, 0.8),
        ("0.8-1.0", 0.8, 1.0),
        ("1.0-1.5", 1.0, 1.5),
        ("1.5-2.0", 1.5, 2.0),
        ("2.0+", 2.0, float("inf")),
    ]
    calibration = {}
    for label, lo, hi in z_buckets:
        group = [
            g for g in v2_bets
            if lo <= abs(g.get("opening_z_score") or 0) < hi
        ]
        if not group:
            continue
        stats = _record_stats(group, lambda g: g["v2_result"])
        predicted = round(
            sum(_normal_cdf(abs(g["opening_z_score"])) for g in group)
            / len(group), 4
        )
        calibration[label] = {
            "predicted_win_prob": predicted,
            "actual_win_rate": stats["win_rate"],
            "count": len(group),
        }

    # ── V1 ─────────────────────────────────────────────────────────────────────
    v1_bets = [g for g in graded_games if g.get("v1_result") is not None]
    v1_overall = _record_stats(v1_bets, lambda g: g["v1_result"])

    v1_misses = [g["v1_miss"] for g in graded_games if g.get("v1_miss") is not None]
    avg_miss_v1 = (round(sum(abs(m) for m in v1_misses) / len(v1_misses), 2)
                   if v1_misses else None)

    return {
        "date_range": date_range,
        "total_games": len(graded_games),
        "v2": {
            **v2_overall,
            "by_confidence": by_confidence,
            "by_direction": by_direction,
            "by_gap_size": by_gap_size,
            "avg_miss": avg_miss,
            "calibration": calibration,
        },
        "v1": {
            **v1_overall,
            "avg_miss": avg_miss_v1,
        },
    }


def _record_stats(games, get_result):
    """Compute W/L/P counts, win rate, and ROI for a group of games."""
    wins = sum(1 for g in games if get_result(g) == "WIN")
    losses = sum(1 for g in games if get_result(g) == "LOSS")
    pushes = sum(1 for g in games if get_result(g) == "PUSH")
    total_bets = wins + losses
    win_rate = round(wins / total_bets, 4) if total_bets > 0 else None
    roi = (
        round(((wins * VIG_WIN) - (losses * VIG_RISK))
              / (total_bets * VIG_RISK) * 100, 2)
        if total_bets > 0 else None
    )
    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "total_bets": total_bets,
        "win_rate": win_rate,
        "roi": roi,
    }


def _normal_cdf(z):
    """
    Normal CDF via Abramowitz & Stegun approximation.
    Matches Yggdrasil's normalCDFApprox for exact calibration parity.
    """
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989422820 * math.exp(-0.5 * z * z)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.7814779
                 + t * (-1.8212560 + t * 1.3302744))))
    return 1.0 - p if z >= 0 else p
