"""PA probability engine.

Pure math — no I/O, no side effects. Takes stats and context in,
returns probability tables out. This is the core of every plate
appearance in the Monte Carlo simulation.

Pipeline: odds_ratio → compute_matchup_rates → apply_park_factors
        → apply_weather_adjustments → normalize
Assembled by: build_pa_probability_table
"""
from mlb.config import Hand, Outcome, WindDirection
from mlb.data.models import (
    BatterStats,
    PitcherStats,
    ParkFactors,
    Weather,
    resolve_batter_hand,
)


def odds_ratio(p_batter: float, p_pitcher: float, p_league: float) -> float:
    """Combine batter and pitcher rates using the Odds Ratio method.

    Removes league-average baseline to avoid double-counting.
    From Tom Tango, "The Book".
    """
    if p_batter == 0.0 or p_pitcher == 0.0 or p_league == 0.0:
        return 0.0

    # Clamp inputs at 0.999 to avoid division by zero
    p_batter = min(p_batter, 0.999)
    p_pitcher = min(p_pitcher, 0.999)
    p_league = min(p_league, 0.999)

    # Convert to odds
    o_b = p_batter / (1.0 - p_batter)
    o_p = p_pitcher / (1.0 - p_pitcher)
    o_l = p_league / (1.0 - p_league)

    # Combine in odds space
    o_matchup = (o_b * o_p) / o_l

    # Convert back to probability, clamped to [0, 1]
    p_matchup = o_matchup / (1.0 + o_matchup)
    return max(0.0, min(1.0, p_matchup))


def compute_matchup_rates(
    batter: BatterStats,
    pitcher: PitcherStats,
    league_averages: dict[str, float],
) -> dict[str, float]:
    """Apply Odds Ratio to every outcome. Returns raw rates (not yet normalized)."""
    return {
        outcome.value: odds_ratio(
            batter.rates.get(outcome.value, 0.0),
            pitcher.rates.get(outcome.value, 0.0),
            league_averages.get(outcome.value, 0.0),
        )
        for outcome in Outcome
    }


# Outcomes adjusted by park factors. HBP is not park-dependent; OUT is derived.
_PARK_ADJUSTED = {'K', 'BB', '1B', '2B', '3B', 'HR'}


def apply_park_factors(
    rates: dict[str, float],
    park_factors: dict[str, float],
) -> dict[str, float]:
    """Multiply outcome rates by park factor multipliers.

    Only adjusts outcomes in _PARK_ADJUSTED. HBP and OUT are left unchanged.
    Missing factors default to 1.0 (no adjustment).
    """
    return {
        outcome: rate * park_factors.get(outcome, 1.0)
        if outcome in _PARK_ADJUSTED
        else rate
        for outcome, rate in rates.items()
    }


# ── Weather adjustment constants ─────────────────────────────────────────────
# Based on Alan Nathan's batted ball physics research.

_TEMP_BASELINE_F = 70.0
_TEMP_HR_BASE = 1.025     # per 10°F above baseline
_TEMP_XBH_BASE = 1.01     # 2B/3B per 10°F above baseline
_WIND_HR_COEFF = 0.008    # per mph
_WIND_2B_COEFF = 0.003    # per mph
_WEATHER_FLOOR = 0.001    # minimum rate (weather can't make outcomes impossible)


def apply_weather_adjustments(
    rates: dict[str, float],
    weather: Weather | None,
) -> dict[str, float]:
    """Adjust rates for temperature and wind effects.

    Returns rates unchanged (as a copy) if weather is None or is_indoor.
    Temperature affects HR, 2B, 3B. Wind affects HR, 2B.
    All results clamped to >= 0.001.
    """
    if weather is None or weather.is_indoor:
        return dict(rates)

    adjusted = dict(rates)

    # Temperature adjustments (exponential per 10°F deviation)
    temp_delta = (weather.temperature_f - _TEMP_BASELINE_F) / 10.0
    adjusted['HR'] = rates.get('HR', 0.0) * (_TEMP_HR_BASE ** temp_delta)
    adjusted['2B'] = rates.get('2B', 0.0) * (_TEMP_XBH_BASE ** temp_delta)
    adjusted['3B'] = rates.get('3B', 0.0) * (_TEMP_XBH_BASE ** temp_delta)

    # Wind adjustments (linear with speed, direction-dependent)
    if weather.wind_direction == WindDirection.OUT_TO_CF:
        adjusted['HR'] *= (1.0 + _WIND_HR_COEFF * weather.wind_speed_mph)
        adjusted['2B'] *= (1.0 + _WIND_2B_COEFF * weather.wind_speed_mph)
    elif weather.wind_direction == WindDirection.IN_FROM_CF:
        adjusted['HR'] *= (1.0 - _WIND_HR_COEFF * weather.wind_speed_mph)
        adjusted['2B'] *= (1.0 - _WIND_2B_COEFF * weather.wind_speed_mph)

    # Clamp: weather shouldn't make any outcome impossible
    for outcome in adjusted:
        adjusted[outcome] = max(_WEATHER_FLOOR, adjusted[outcome])

    return adjusted


def normalize(rates: dict[str, float]) -> dict[str, float]:
    """Normalize rates to sum to 1.0. Uniform fallback if sum is 0."""
    total = sum(rates.values())
    if total == 0:
        n = len(rates)
        return {k: 1.0 / n for k in rates}
    return {k: v / total for k, v in rates.items()}
