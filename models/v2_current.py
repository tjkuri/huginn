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
