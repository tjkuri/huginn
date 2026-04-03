"""Tests for mlb.engine.aggregate."""
import pytest
import numpy as np

from mlb.config import Hand, Outcome, LEAGUE_AVERAGES
from mlb.data.models import (
    BatterStats, PitcherStats, Lineup, ParkFactors, GameContext, SimulatedGame,
)
from mlb.engine.aggregate import run_simulations


# ── Shared fixtures ──────────────────────────────────────────────────────────

_REALISTIC_RATES = {
    'K': 0.225, 'BB': 0.076, 'HBP': 0.011,
    '1B': 0.145, '2B': 0.046, '3B': 0.004, 'HR': 0.030, 'OUT': 0.463,
}


def _make_pitcher(name: str, throws: str = 'R') -> PitcherStats:
    return PitcherStats(
        player_id=name, name=name, throws=Hand(throws),
        pa_against=500, rates=dict(_REALISTIC_RATES),
    )


def _make_lineup(
    team_id: str = 'TST',
    starter_name: str = 'Starter',
    rates: dict | None = None,
) -> Lineup:
    r = rates or dict(_REALISTIC_RATES)
    batters = [
        BatterStats(
            player_id=f'{team_id}_b{i}', name=f'{team_id}_Batter{i}',
            bats=Hand.RIGHT, pa=500, rates=dict(r),
        )
        for i in range(9)
    ]
    return Lineup(
        team_id=team_id, team_name=f'Team {team_id}',
        batting_order=batters,
        starting_pitcher=_make_pitcher(starter_name),
        bullpen=[_make_pitcher(f'{team_id}_R{i}') for i in range(3)],
    )


def _make_park_factors() -> ParkFactors:
    neutral = {o.value: 1.0 for o in Outcome}
    return ParkFactors(
        venue_id='TEST', venue_name='Test Park',
        factors_vs_lhb=dict(neutral),
        factors_vs_rhb=dict(neutral),
    )


def _make_game_context(away_lineup=None, home_lineup=None) -> GameContext:
    return GameContext(
        game_id='test-001',
        date='2026-04-01',
        away_lineup=away_lineup or _make_lineup('AWY', 'AwaySP'),
        home_lineup=home_lineup or _make_lineup('HME', 'HomeSP'),
        park_factors=_make_park_factors(),
    )


# ── run_simulations ──────────────────────────────────────────────────────────

class TestRunSimulations:

    def test_returns_correct_count(self):
        """Requesting 100 sims returns exactly 100 SimulatedGame objects."""
        ctx = _make_game_context()
        games = run_simulations(ctx, LEAGUE_AVERAGES, n_simulations=100, base_seed=42)
        assert len(games) == 100
        assert all(isinstance(g, SimulatedGame) for g in games)

    def test_reproducible_with_same_seed(self):
        """Same base_seed produces identical game results."""
        ctx = _make_game_context()
        games_a = run_simulations(ctx, LEAGUE_AVERAGES, n_simulations=50, base_seed=7)
        games_b = run_simulations(ctx, LEAGUE_AVERAGES, n_simulations=50, base_seed=7)
        scores_a = [(g.away_runs, g.home_runs) for g in games_a]
        scores_b = [(g.away_runs, g.home_runs) for g in games_b]
        assert scores_a == scores_b

    def test_different_seeds_produce_different_results(self):
        """Different base_seed values produce different game sequences."""
        ctx = _make_game_context()
        games_a = run_simulations(ctx, LEAGUE_AVERAGES, n_simulations=50, base_seed=1)
        games_b = run_simulations(ctx, LEAGUE_AVERAGES, n_simulations=50, base_seed=99)
        scores_a = [(g.away_runs, g.home_runs) for g in games_a]
        scores_b = [(g.away_runs, g.home_runs) for g in games_b]
        assert scores_a != scores_b
