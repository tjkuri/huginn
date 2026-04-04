"""MLB Monte Carlo simulation configuration.

Enums, league-average baselines, season settings, and simulation defaults.
"""
from enum import Enum
from pathlib import Path


# ── Enums ────────────────────────────────────────────────────────────────────

class Hand(str, Enum):
    LEFT = 'L'
    RIGHT = 'R'
    SWITCH = 'S'


class Outcome(str, Enum):
    K = 'K'
    BB = 'BB'
    HBP = 'HBP'
    SINGLE = '1B'
    DOUBLE = '2B'
    TRIPLE = '3B'
    HR = 'HR'
    OUT = 'OUT'


class WindDirection(str, Enum):
    OUT_TO_CF = 'out_to_cf'
    IN_FROM_CF = 'in_from_cf'
    CROSS = 'cross'
    CALM = 'calm'


# ── Outcome categories (ordered) ────────────────────────────────────────────

OUTCOMES = [o.value for o in Outcome]


# ── Season & simulation defaults ─────────────────────────────────────────────

SEASON = 2026
NUM_SIMULATIONS = 10_000
DEFAULT_PITCH_COUNT_LIMIT = 100


# ── Cache configuration ─────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent.parent / "baseball_cache"

# Current-season raw stats are re-fetched if older than this many hours.
# Prior-season files are kept forever (historical data doesn't change).
STATS_CACHE_MAX_AGE_HOURS = 6


# ── League average PA outcome rates (2025 approx) ───────────────────────────
# Keys: (batter_hand, pitcher_hand) → outcome rates
# These are baselines for the Odds Ratio method.
# Will be updated once 2026 data stabilizes.

LEAGUE_AVERAGES = {
    (Hand.LEFT, Hand.RIGHT): {   # LHB vs RHP (platoon advantage)
        'K': 0.207, 'BB': 0.088, 'HBP': 0.012,
        '1B': 0.153, '2B': 0.048, '3B': 0.005, 'HR': 0.035,
        'OUT': 0.452,
    },
    (Hand.RIGHT, Hand.LEFT): {   # RHB vs LHP (platoon advantage)
        'K': 0.211, 'BB': 0.085, 'HBP': 0.011,
        '1B': 0.150, '2B': 0.050, '3B': 0.004, 'HR': 0.037,
        'OUT': 0.452,
    },
    (Hand.LEFT, Hand.LEFT): {    # LHB vs LHP (same side)
        'K': 0.228, 'BB': 0.078, 'HBP': 0.010,
        '1B': 0.142, '2B': 0.044, '3B': 0.004, 'HR': 0.028,
        'OUT': 0.466,
    },
    (Hand.RIGHT, Hand.RIGHT): {  # RHB vs RHP (same side)
        'K': 0.225, 'BB': 0.076, 'HBP': 0.011,
        '1B': 0.145, '2B': 0.046, '3B': 0.004, 'HR': 0.030,
        'OUT': 0.463,
    },
}


# ── Validate league averages at import time ──────────────────────────────────

for _key, _rates in LEAGUE_AVERAGES.items():
    _total = sum(_rates.values())
    assert abs(_total - 1.0) < 0.001, (
        f"LEAGUE_AVERAGES[{_key}] sums to {_total:.4f}, expected 1.0"
    )
