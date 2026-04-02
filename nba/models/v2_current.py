"""
V2 model math -- port of Yggdrasil's utils/nbaMath.js.

Produces identical outputs for weighted stats, projected totals,
z-scores, confidence tiers, and recommendations.
"""
import math
from datetime import datetime


def _days_since(item_date, today):
    """Days between two YYYY-MM-DD date strings."""
    t = datetime.strptime(today, "%Y-%m-%d")
    d = datetime.strptime(item_date, "%Y-%m-%d")
    return (t - d).total_seconds() / 86400


def weighted_mean(items, today, decay):
    """
    Exponential recency-weighted mean.
    items: list of {"date": "YYYY-MM-DD", "value": number}
    """
    if not items:
        return None
    weighted_sum = 0.0
    weight_sum = 0.0
    for item in items:
        days = _days_since(item["date"], today)
        w = decay ** days
        weighted_sum += item["value"] * w
        weight_sum += w
    return weighted_sum / weight_sum if weight_sum > 0 else None


def weighted_variance(items, today, decay, w_mean):
    """Exponential recency-weighted population variance."""
    if not items or len(items) < 2 or w_mean is None:
        return None
    weighted_sum_sq = 0.0
    weight_sum = 0.0
    for item in items:
        days = _days_since(item["date"], today)
        w = decay ** days
        weighted_sum_sq += w * (item["value"] - w_mean) ** 2
        weight_sum += w
    return weighted_sum_sq / weight_sum if weight_sum > 0 else None


def normal_cdf(z):
    """
    Normal CDF via Abramowitz & Stegun rational approximation.
    Coefficient precision matches Yggdrasil's normalCDF in utils/nbaMath.js.
    """
    sign = -1 if z < 0 else 1
    abs_z = abs(z)
    t = 1.0 / (1.0 + 0.2316419 * abs_z)
    poly = t * (0.319381530
                + t * (-0.356563782
                + t * (1.781477937
                + t * (-1.821255978
                + t * 1.330274429))))
    pdf = math.exp(-0.5 * abs_z * abs_z) / math.sqrt(2 * math.pi)
    cdf = 1.0 - pdf * poly
    return 0.5 * (1.0 + sign * (2 * cdf - 1))


def _to_items(games, field):
    """Convert game dicts to [{date, value}] for weighted stats."""
    return [{"date": g["date"], "value": g[field]} for g in games]


def compute_my_line(home_games, away_games, today, cfg):
    """
    Compute projected total using O/D splits, recency weighting, home court.
    Returns dict with: my_line, proj_home, proj_away, sd_total, components.
    """
    decay = cfg["decay_factor"]
    home_boost_default = cfg["home_boost"]
    min_ha = cfg["min_home_away_games"]

    home_off_items = _to_items(home_games, "pointsScored")
    home_def_items = _to_items(home_games, "pointsAllowed")
    away_off_items = _to_items(away_games, "pointsScored")
    away_def_items = _to_items(away_games, "pointsAllowed")

    home_off = weighted_mean(home_off_items, today, decay)
    home_def = weighted_mean(home_def_items, today, decay)
    away_off = weighted_mean(away_off_items, today, decay)
    away_def = weighted_mean(away_def_items, today, decay)

    if any(v is None for v in (home_off, home_def, away_off, away_def)):
        return {"my_line": None, "proj_home": None, "proj_away": None,
                "sd_total": None, "components": None}

    home_boost = home_boost_default
    home_at_home = [g for g in home_games if g["isHome"]]
    home_away = [g for g in home_games if not g["isHome"]]
    if len(home_at_home) >= min_ha and len(home_away) >= min_ha:
        at_home_mean = weighted_mean(
            _to_items(home_at_home, "pointsScored"), today, decay)
        away_mean = weighted_mean(
            _to_items(home_away, "pointsScored"), today, decay)
        if at_home_mean is not None and away_mean is not None:
            home_boost = at_home_mean - away_mean

    proj_home = (home_off + away_def) / 2 + home_boost / 2
    proj_away = (away_off + home_def) / 2 - home_boost / 2
    my_line = proj_home + proj_away

    var_home_off = weighted_variance(home_off_items, today, decay, home_off)
    var_home_def = weighted_variance(home_def_items, today, decay, home_def)
    var_away_off = weighted_variance(away_off_items, today, decay, away_off)
    var_away_def = weighted_variance(away_def_items, today, decay, away_def)

    var_total = (((var_home_off or 0) + (var_away_def or 0)) / 4
                + ((var_away_off or 0) + (var_home_def or 0)) / 4)
    sd_total = math.sqrt(var_total) if var_total > 0 else None

    return {
        "my_line": my_line,
        "proj_home": proj_home,
        "proj_away": proj_away,
        "sd_total": sd_total,
        "components": {
            "home_off": home_off,
            "home_def": home_def,
            "away_off": away_off,
            "away_def": away_def,
            "home_boost": home_boost,
        },
    }


def compute_confidence_and_ev(discrepancy, sd_total, cfg):
    """Compute z-score, confidence tier, and vig-adjusted EV."""
    if sd_total is None or sd_total == 0 or discrepancy is None:
        return {"z_score": None, "confidence": None, "expected_value": None}

    z = discrepancy / sd_total
    abs_z = abs(z)

    if abs_z >= cfg["z_high"]:
        confidence = "HIGH"
    elif abs_z >= cfg["z_medium"]:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    win_prob = normal_cdf(abs_z)
    ev = win_prob * cfg["vig_win"] - (1 - win_prob) * cfg["vig_risk"]

    return {
        "z_score": round(z, 3),
        "confidence": confidence,
        "expected_value": round(ev, 4),
    }


def compute_recommendation(my_line, dk_line, z_score, expected_value, cfg):
    """Determine recommendation based on edge thresholds."""
    if my_line is None or dk_line is None:
        return None
    if (z_score is None
            or abs(z_score) < cfg["min_z_threshold"]
            or expected_value <= 0):
        return "NO_BET"
    if my_line > dk_line:
        return "O"
    if my_line < dk_line:
        return "U"
    return "P"
