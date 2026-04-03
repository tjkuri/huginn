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


def advance_runners(
    bases: BaseState,
    outcome: Outcome,
    outs: int,
    rng: np.random.Generator,
) -> tuple[BaseState, int, int]:
    """Advance baserunners for a PA outcome.

    Returns (new_base_state, runs_scored, new_outs).
    """
    if outcome == Outcome.K:
        return bases, 0, outs + 1

    if outcome in (Outcome.BB, Outcome.HBP):
        return _advance_walk(bases)

    if outcome == Outcome.OUT:
        return _advance_out(bases, outs, rng)

    if outcome == Outcome.SINGLE:
        return _advance_single(bases, rng)

    if outcome == Outcome.DOUBLE:
        return _advance_double(bases, rng)

    if outcome == Outcome.TRIPLE:
        return _advance_triple(bases)

    if outcome == Outcome.HR:
        return _advance_hr(bases)

    # Should not reach here with valid Outcome enum
    return bases, 0, outs + 1


def _advance_walk(bases: BaseState) -> tuple[BaseState, int, int]:
    """Walk/HBP: batter to 1st, force advances only."""
    runs = 0
    third = bases.third
    second = bases.second

    # Force chain: only advance if base behind is occupied
    if bases.first:
        if bases.second:
            if bases.third:
                runs = 1  # runner on 3rd forced home
            third = True  # runner on 2nd forced to 3rd
        second = True  # runner on 1st forced to 2nd

    return BaseState(first=True, second=second, third=third), runs, 0


def _advance_out(
    bases: BaseState, outs: int, rng: np.random.Generator,
) -> tuple[BaseState, int, int]:
    """Generic out. Sac fly: runner on 3rd scores ~50% with < 2 outs.

    Note: does not model ground-ball double plays — future enhancement.
    """
    runs = 0
    new_outs = outs + 1
    third = bases.third

    if bases.third and outs < 2:
        # Sac fly approximation: 50% chance runner on 3rd scores
        if rng.random() < 0.5:
            runs = 1
            third = False

    return BaseState(first=bases.first, second=bases.second, third=third), runs, new_outs


def _advance_single(bases: BaseState, rng: np.random.Generator) -> tuple[BaseState, int, int]:
    raise NotImplementedError


def _advance_double(bases: BaseState, rng: np.random.Generator) -> tuple[BaseState, int, int]:
    raise NotImplementedError


def _advance_triple(bases: BaseState) -> tuple[BaseState, int, int]:
    raise NotImplementedError


def _advance_hr(bases: BaseState) -> tuple[BaseState, int, int]:
    raise NotImplementedError
