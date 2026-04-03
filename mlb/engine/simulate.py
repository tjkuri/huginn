"""Game simulation engine.

Simulates a complete baseball game using the probability engine for each
plate appearance. Manages innings, outs, baserunners, batting order,
and pitcher substitutions. All randomness flows through an explicit
numpy.random.Generator for reproducibility.
"""
import numpy as np

from mlb.config import Outcome, OUTCOMES, DEFAULT_PITCH_COUNT_LIMIT
from mlb.data.models import (
    BaseState,
    BatterStats,
    GameContext,
    GameState,
    Lineup,
    PAResult,
    PitcherStats,
    SimulatedGame,
)
from mlb.engine.probabilities import build_pa_probability_table


def resolve_pa_outcome(
    probability_table: dict[str, float],
    rng: np.random.Generator,
) -> Outcome:
    """Sample one PA outcome from a normalized probability table.

    Uses rng.random() to draw a uniform [0, 1) value and walks the
    cumulative distribution to select the outcome.
    """
    r = rng.random()
    cumulative = 0.0
    for key in OUTCOMES:
        cumulative += probability_table.get(key, 0.0)
        if r < cumulative:
            return Outcome(key)
    # Floating-point edge case: return last outcome
    return Outcome(OUTCOMES[-1])
