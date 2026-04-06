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
PITCHES_PER_OUTCOME = {
    Outcome.K: 5,
    Outcome.BB: 5,
    Outcome.HBP: 2,
    Outcome.OUT: 4,
    Outcome.SINGLE: 3,
    Outcome.DOUBLE: 3,
    Outcome.TRIPLE: 3,
    Outcome.HR: 3,
}


# ── Season & simulation defaults ─────────────────────────────────────────────

SEASON = 2026
NUM_SIMULATIONS = 10_000
DEFAULT_PITCH_COUNT_LIMIT = 100


# ── Cache configuration ─────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent.parent / "baseball_cache"

# Current-season raw stats are re-fetched if older than this many hours.
# Prior-season files are kept forever (historical data doesn't change).
STATS_CACHE_MAX_AGE_HOURS = 6


# ── League average PA outcome rates (2025 hardcoded fallback only) ──────────
# Keys: (batter_hand, pitcher_hand) → outcome rates
# These are baselines for the Odds Ratio method and the last-resort fallback
# only. The runtime path now tries to fetch fresh league averages when
# `build_game_context()` runs and mutates this mapping in place if successful.
# Fallback values below are from 2025 full-season matchup recomputation.

LEAGUE_AVERAGES = {
    (Hand.LEFT, Hand.RIGHT): {   # LHB vs RHP (platoon advantage)
        'K': 0.210879, 'BB': 0.098247, 'HBP': 0.009000,
        '1B': 0.137730, '2B': 0.043644, '3B': 0.004368, 'HR': 0.034682,
        'OUT': 0.461449,
    },
    (Hand.RIGHT, Hand.LEFT): {   # RHB vs LHP (platoon advantage)
        'K': 0.225171, 'BB': 0.083435, 'HBP': 0.008495,
        '1B': 0.144848, '2B': 0.043476, '3B': 0.002651, 'HR': 0.030192,
        'OUT': 0.461732,
    },
    (Hand.LEFT, Hand.LEFT): {    # LHB vs LHP (same side)
        'K': 0.234301, 'BB': 0.078885, 'HBP': 0.013108,
        '1B': 0.142229, '2B': 0.036264, '3B': 0.004396, 'HR': 0.024961,
        'OUT': 0.465856,
    },
    (Hand.RIGHT, Hand.RIGHT): {  # RHB vs RHP (same side)
        'K': 0.226454, 'BB': 0.075364, 'HBP': 0.012099,
        '1B': 0.145286, '2B': 0.042001, '3B': 0.003009, 'HR': 0.029551,
        'OUT': 0.466236,
    },
}


# ── Validate league averages at import time ──────────────────────────────────

for _key, _rates in LEAGUE_AVERAGES.items():
    _total = sum(_rates.values())
    assert abs(_total - 1.0) < 0.001, (
        f"LEAGUE_AVERAGES[{_key}] sums to {_total:.4f}, expected 1.0"
    )
