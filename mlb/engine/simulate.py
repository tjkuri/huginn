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


def _advance_single(
    bases: BaseState, rng: np.random.Generator,
) -> tuple[BaseState, int, int]:
    """Single: batter to 1st. Runners advance with probabilistic outcomes."""
    runs = 0

    # Runner on 3rd scores
    if bases.third:
        runs += 1

    # Runner on 2nd: scores (90%) or holds at 3rd (10%)
    third = False
    if bases.second:
        if rng.random() < 0.9:
            runs += 1
        else:
            third = True

    # Runner on 1st: to 2nd (70%) or to 3rd (30%)
    second = False
    if bases.first:
        if rng.random() < 0.7:
            second = True
        else:
            third = True

    return BaseState(first=True, second=second, third=third), runs, 0


def _advance_double(
    bases: BaseState, rng: np.random.Generator,
) -> tuple[BaseState, int, int]:
    """Double: batter to 2nd. Runner on 2nd/3rd score. Runner on 1st to 3rd or scores."""
    runs = 0

    # Runner on 3rd scores
    if bases.third:
        runs += 1

    # Runner on 2nd scores
    if bases.second:
        runs += 1

    # Runner on 1st: to 3rd (60%) or scores (40%)
    third = False
    if bases.first:
        if rng.random() < 0.6:
            third = True
        else:
            runs += 1

    return BaseState(first=False, second=True, third=third), runs, 0


def _advance_triple(bases: BaseState) -> tuple[BaseState, int, int]:
    """Triple: batter to 3rd. All runners score."""
    runs = sum([bases.first, bases.second, bases.third])
    return BaseState(third=True), runs, 0


def _advance_hr(bases: BaseState) -> tuple[BaseState, int, int]:
    """Home run: batter scores. All runners score."""
    runs = 1 + sum([bases.first, bases.second, bases.third])
    return BaseState(), runs, 0


def should_pull_starter(
    pitch_count: int,
    innings_pitched: float,
    runs_allowed: int,
    config: dict | None = None,
) -> bool:
    """Decide whether to replace the starting pitcher.

    Checks pitch count, innings pitched, and runs allowed against
    configurable thresholds.
    """
    cfg = config or {}
    pitch_limit = cfg.get('pitch_count_limit', DEFAULT_PITCH_COUNT_LIMIT)
    innings_limit = cfg.get('innings_limit', 9.0)
    runs_limit = cfg.get('runs_limit', 8)

    if pitch_count >= pitch_limit:
        return True
    if innings_pitched >= innings_limit:
        return True
    if runs_allowed >= runs_limit:
        return True
    return False


def get_current_pitcher(
    lineup: Lineup,
    game_state: GameState,
    is_home: bool,
) -> PitcherStats:
    """Return the pitcher currently on the mound for the given team.

    Uses bullpen_index from GameState: -1 means starter is still in,
    0+ indexes into lineup.bullpen. If bullpen is exhausted, reuses
    the last available arm (emergency pitcher).

    Note: bullpen arms are used in order — handedness-based bullpen
    usage is a future enhancement.
    """
    bullpen_idx = game_state.home_bullpen_index if is_home else game_state.away_bullpen_index

    if bullpen_idx < 0:
        return lineup.starting_pitcher

    if not lineup.bullpen:
        return lineup.starting_pitcher

    # Clamp to last available arm if bullpen exhausted
    idx = min(bullpen_idx, len(lineup.bullpen) - 1)
    return lineup.bullpen[idx]


# Approximate pitches per PA for pitch count tracking
_PITCHES_PER_PA = 4


def simulate_half_inning(
    game_context: GameContext,
    game_state: GameState,
    is_top: bool,
    league_averages: dict,
    rng: np.random.Generator,
) -> list[PAResult]:
    """Simulate one half-inning (until 3 outs or walk-off).

    Mutates game_state in place (score, batting index, pitch count,
    bullpen index). Returns the list of PAResults for this half-inning.
    """
    results: list[PAResult] = []
    outs = 0
    bases = BaseState()

    # Determine batting team and defensive team
    if is_top:
        batting_lineup = game_context.away_lineup
        defensive_lineup = game_context.home_lineup
        is_defense_home = True
    else:
        batting_lineup = game_context.home_lineup
        defensive_lineup = game_context.away_lineup
        is_defense_home = False

    while outs < 3:
        # Get current batter
        if is_top:
            batter_idx = game_state.away_batting_index
        else:
            batter_idx = game_state.home_batting_index
        batter = batting_lineup.batting_order[batter_idx]

        # Retrieve pitch count and bullpen state for the defensive team
        if is_defense_home:
            pitch_count = game_state.home_pitch_count
            bullpen_idx = game_state.home_bullpen_index
        else:
            pitch_count = game_state.away_pitch_count
            bullpen_idx = game_state.away_bullpen_index

        # Approximate innings pitched from pitch count
        innings_pitched = pitch_count / (_PITCHES_PER_PA * 3)

        # Track runs allowed (runs scored by the batting team so far)
        if is_defense_home:
            runs_allowed = game_state.away_score
        else:
            runs_allowed = game_state.home_score

        # Check if starter should be pulled
        if bullpen_idx < 0 and should_pull_starter(pitch_count, innings_pitched, runs_allowed):
            if is_defense_home:
                game_state.home_bullpen_index = 0
                game_state.home_pitch_count = 0
            else:
                game_state.away_bullpen_index = 0
                game_state.away_pitch_count = 0

        pitcher = get_current_pitcher(defensive_lineup, game_state, is_home=is_defense_home)

        # Build probability table and sample outcome
        prob_table = build_pa_probability_table(
            batter, pitcher, game_context.park_factors,
            game_context.weather, league_averages,
        )
        outcome = resolve_pa_outcome(prob_table, rng)

        # Record base state before advancement
        runners_before = bases

        # Advance runners
        bases, runs_scored, outs = advance_runners(bases, outcome, outs, rng)

        # Update score
        if is_top:
            game_state.away_score += runs_scored
        else:
            game_state.home_score += runs_scored

        # Record PA result
        results.append(PAResult(
            outcome=outcome,
            batter_id=batter.player_id,
            pitcher_id=pitcher.player_id,
            inning=game_state.inning,
            runners_before=runners_before,
            runs_scored=runs_scored,
        ))

        # Advance batting order (mod 9)
        if is_top:
            game_state.away_batting_index = (batter_idx + 1) % 9
        else:
            game_state.home_batting_index = (batter_idx + 1) % 9

        # Increment pitch count for defensive pitcher
        if is_defense_home:
            game_state.home_pitch_count += _PITCHES_PER_PA
        else:
            game_state.away_pitch_count += _PITCHES_PER_PA

        # Walk-off check: bottom of 9th+, home leads → end immediately
        if not is_top and game_state.inning >= 9 and game_state.home_score > game_state.away_score:
            break

    return results
