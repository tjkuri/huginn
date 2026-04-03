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
