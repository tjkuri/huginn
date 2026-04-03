"""Tests for mlb.engine.aggregate."""
import pytest
import numpy as np

from mlb.config import Hand, Outcome, LEAGUE_AVERAGES
from mlb.data.models import (
    BatterStats, PitcherStats, Lineup, ParkFactors, GameContext, SimulatedGame,
)
from mlb.engine.aggregate import run_simulations, compute_run_distributions, compute_win_probability, compute_player_stats, compute_betting_lines


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


class TestComputeRunDistributions:

    def _make_games(self, n: int = 300) -> list[SimulatedGame]:
        ctx = _make_game_context()
        return run_simulations(ctx, LEAGUE_AVERAGES, n_simulations=n, base_seed=42)

    def test_mean_total_runs_in_range(self):
        """Mean total runs is positive and finite."""
        games = self._make_games()
        dist = compute_run_distributions(games)
        mean = dist['total_runs']['mean']
        assert mean > 0 and mean < float('inf')

    def test_std_is_positive(self):
        """Standard deviation of total runs is positive."""
        games = self._make_games()
        dist = compute_run_distributions(games)
        assert dist['total_runs']['std'] > 0

    def test_min_le_mean_le_max(self):
        """min <= mean <= max for all three run distributions."""
        games = self._make_games()
        dist = compute_run_distributions(games)
        for key in ('away_runs', 'home_runs', 'total_runs'):
            d = dist[key]
            assert d['min'] <= d['mean'] <= d['max'], f"failed for {key}"

    def test_distribution_pairs_cover_all_values(self):
        """Distribution list covers the full range of observed values."""
        games = self._make_games(100)
        dist = compute_run_distributions(games)
        values = [v for v, _c in dist['total_runs']['distribution']]
        all_totals = [g.away_runs + g.home_runs for g in games]
        assert set(values) == set(all_totals)

    def test_run_diff_keys_present(self):
        """run_diff contains mean and std."""
        games = self._make_games(100)
        dist = compute_run_distributions(games)
        assert 'mean' in dist['run_diff']
        assert 'std' in dist['run_diff']


class TestComputeWinProbability:

    def test_sums_to_one(self):
        """home_win_pct + away_win_pct + tie_pct == 1.0."""
        ctx = _make_game_context()
        games = run_simulations(ctx, LEAGUE_AVERAGES, n_simulations=200, base_seed=42)
        wp = compute_win_probability(games)
        total = wp['home_win_pct'] + wp['away_win_pct'] + wp['tie_pct']
        assert abs(total - 1.0) < 1e-9

    def test_rigged_game_home_dominates(self):
        """Away team with K=1.0 batters should almost never win (home_win_pct > 0.90)."""
        k_only_rates = {o.value: 0.0 for o in Outcome}
        k_only_rates['K'] = 1.0
        away_lineup = _make_lineup('AWY', 'AwaySP', rates=k_only_rates)
        home_lineup = _make_lineup('HME', 'HomeSP')
        ctx = _make_game_context(away_lineup=away_lineup, home_lineup=home_lineup)
        games = run_simulations(ctx, LEAGUE_AVERAGES, n_simulations=500, base_seed=1)
        wp = compute_win_probability(games)
        assert wp['home_win_pct'] > 0.90

    def test_all_keys_present(self):
        """Result contains home_win_pct, away_win_pct, tie_pct."""
        ctx = _make_game_context()
        games = run_simulations(ctx, LEAGUE_AVERAGES, n_simulations=50, base_seed=5)
        wp = compute_win_probability(games)
        assert set(wp.keys()) == {'home_win_pct', 'away_win_pct', 'tie_pct'}


class TestComputePlayerStats:

    def test_all_players_present(self):
        """Every player ID from both lineups appears in the output."""
        ctx = _make_game_context()
        games = run_simulations(ctx, LEAGUE_AVERAGES, n_simulations=100, base_seed=42)
        stats = compute_player_stats(games)

        away_ids = {b.player_id for b in ctx.away_lineup.batting_order}
        home_ids = {b.player_id for b in ctx.home_lineup.batting_order}
        all_ids = away_ids | home_ids

        assert all_ids.issubset(set(stats.keys()))

    def test_hr_hitter_has_higher_hr_rate(self):
        """Batter with HR=0.20 shows higher hr_per_game than batter with HR=0.03."""
        hr_rates = dict(_REALISTIC_RATES)
        hr_rates['HR'] = 0.20
        hr_rates['OUT'] = _REALISTIC_RATES['OUT'] - (0.20 - _REALISTIC_RATES['HR'])

        hr_lineup = _make_lineup('HR_', 'HRSp', rates=hr_rates)
        normal_lineup = _make_lineup('NRM', 'NrmSP')
        ctx = _make_game_context(away_lineup=hr_lineup, home_lineup=normal_lineup)
        games = run_simulations(ctx, LEAGUE_AVERAGES, n_simulations=500, base_seed=42)
        stats = compute_player_stats(games)

        hr_player_avg = np.mean([stats[pid].hr_per_game for pid in stats if pid.startswith('HR_')])
        normal_player_avg = np.mean([stats[pid].hr_per_game for pid in stats if pid.startswith('NRM')])

        assert hr_player_avg > normal_player_avg

    def test_stats_have_expected_fields(self):
        """PlayerSimStats objects have all required fields with positive pa_per_game."""
        ctx = _make_game_context()
        games = run_simulations(ctx, LEAGUE_AVERAGES, n_simulations=50, base_seed=1)
        stats = compute_player_stats(games)

        pid = next(iter(stats))
        s = stats[pid]
        assert hasattr(s, 'pa_per_game')
        assert hasattr(s, 'hits_per_game')
        assert hasattr(s, 'hr_per_game')
        assert hasattr(s, 'bb_per_game')
        assert hasattr(s, 'k_per_game')
        assert hasattr(s, 'total_bases_per_game')
        assert hasattr(s, 'hits_per_game_std')
        assert s.pa_per_game > 0


class TestComputeBettingLines:

    def _setup(self, n: int = 300) -> tuple[list[SimulatedGame], dict]:
        ctx = _make_game_context()
        games = run_simulations(ctx, LEAGUE_AVERAGES, n_simulations=n, base_seed=42)
        from mlb.engine.aggregate import compute_run_distributions
        run_dists = compute_run_distributions(games)
        return games, run_dists

    def test_over_under_structure(self):
        """Totals output contains lines from 5.5 to 12.5; each sums to 1.0."""
        games, run_dists = self._setup()
        lines = compute_betting_lines(games, run_dists)

        expected_lines = {l / 2 for l in range(11, 26)}  # 5.5 to 12.5 step 0.5
        assert set(lines['totals'].keys()) == expected_lines

        for line, entry in lines['totals'].items():
            total = entry['over_pct'] + entry['under_pct'] + entry['push_pct']
            assert abs(total - 1.0) < 1e-9, f"totals line {line} does not sum to 1.0"

    def test_over_under_monotonicity(self):
        """over_pct decreases (or stays equal) as the line increases."""
        games, run_dists = self._setup(500)
        lines = compute_betting_lines(games, run_dists)
        assert lines['totals'][6.5]['over_pct'] >= lines['totals'][7.5]['over_pct']
        assert lines['totals'][7.5]['over_pct'] >= lines['totals'][8.5]['over_pct']
        assert lines['totals'][8.5]['over_pct'] >= lines['totals'][9.5]['over_pct']

    def test_moneyline_favorite_negative(self):
        """Favorite has negative American odds; underdog has positive."""
        # Use a rigged game to guarantee clear favorite/underdog
        k_only_rates = {o.value: 0.0 for o in Outcome}
        k_only_rates['K'] = 1.0
        away_lineup = _make_lineup('AWY', 'AwaySP', rates=k_only_rates)
        home_lineup = _make_lineup('HME', 'HomeSP')
        ctx = _make_game_context(away_lineup=away_lineup, home_lineup=home_lineup)
        games = run_simulations(ctx, LEAGUE_AVERAGES, n_simulations=200, base_seed=7)
        from mlb.engine.aggregate import compute_run_distributions
        run_dists = compute_run_distributions(games)
        lines = compute_betting_lines(games, run_dists)
        # Home is heavy favorite → negative american odds
        assert lines['moneyline']['home']['american'] < 0
        # Away is underdog → positive american odds
        assert lines['moneyline']['away']['american'] > 0

    def test_all_sections_present(self):
        """betting_lines contains totals, moneyline, run_line, team_totals."""
        games, run_dists = self._setup(100)
        lines = compute_betting_lines(games, run_dists)
        assert set(lines.keys()) == {'totals', 'moneyline', 'run_line', 'team_totals'}
        assert 'away' in lines['team_totals']
        assert 'home' in lines['team_totals']

    def test_team_totals_structure(self):
        """team_totals contains lines from 2.5 to 8.5 and each sums to 1.0."""
        games, run_dists = self._setup(200)
        lines = compute_betting_lines(games, run_dists)
        expected = {l / 2 for l in range(5, 18)}  # 2.5 to 8.5 step 0.5
        assert set(lines['team_totals']['home'].keys()) == expected
        for line, entry in lines['team_totals']['away'].items():
            total = entry['over_pct'] + entry['under_pct'] + entry['push_pct']
            assert abs(total - 1.0) < 1e-9
