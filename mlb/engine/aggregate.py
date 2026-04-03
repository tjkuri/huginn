"""Simulation aggregation engine.

Runs simulate_game N times and aggregates results into projections and
betting-relevant outputs. One GameContext in, one SimulationResult out.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from mlb.config import Outcome
from mlb.data.models import (
    GameContext,
    PlayerSimStats,
    SimulatedGame,
    SimulationResult,
)
from mlb.engine.simulate import simulate_game

_HIT_OUTCOMES = {Outcome.SINGLE, Outcome.DOUBLE, Outcome.TRIPLE, Outcome.HR}
_TOTAL_BASES = {
    Outcome.SINGLE: 1,
    Outcome.DOUBLE: 2,
    Outcome.TRIPLE: 3,
    Outcome.HR: 4,
}


def run_simulations(
    game_context: GameContext,
    league_averages: dict,
    n_simulations: int = 10000,
    base_seed: int | None = None,
) -> list[SimulatedGame]:
    """Run the game simulation N times and return all SimulatedGame results.

    Each simulation receives a unique seed derived from base_seed + i, making
    the full batch reproducible from a single seed. When base_seed is None,
    a random seed is used.

    # NOTE: This loop is embarrassingly parallel and could be sped up with
    # multiprocessing.Pool or numpy vectorization in a future optimization pass.
    """
    if base_seed is None:
        base_seed = random.randint(0, 2**31 - 1)

    return [
        simulate_game(game_context, league_averages, seed=base_seed + i)
        for i in range(n_simulations)
    ]


def compute_run_distributions(games: list[SimulatedGame]) -> dict:
    """Compute run score distributions from a batch of simulated games.

    Returns a nested dict with summary stats and frequency distributions for
    away_runs, home_runs, total_runs, and run_diff (home minus away).
    """
    away = np.array([g.away_runs for g in games])
    home = np.array([g.home_runs for g in games])
    total = away + home
    diff = home - away  # positive = home leads

    def _summarize(arr: np.ndarray) -> dict:
        values, counts = np.unique(arr, return_counts=True)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'min': int(np.min(arr)),
            'max': int(np.max(arr)),
            'distribution': [(int(v), int(c)) for v, c in zip(values, counts)],
        }

    return {
        'away_runs': _summarize(away),
        'home_runs': _summarize(home),
        'total_runs': _summarize(total),
        'run_diff': {
            'mean': float(np.mean(diff)),
            'std': float(np.std(diff)),
        },
    }


def compute_win_probability(games: list[SimulatedGame]) -> dict:
    """Compute win/loss/tie fractions from a batch of simulated games."""
    n = len(games)
    home_wins = sum(1 for g in games if g.home_runs > g.away_runs)
    away_wins = sum(1 for g in games if g.away_runs > g.home_runs)
    ties = sum(1 for g in games if g.away_runs == g.home_runs)
    return {
        'home_win_pct': home_wins / n,
        'away_win_pct': away_wins / n,
        'tie_pct': ties / n,
    }


def compute_player_stats(games: list[SimulatedGame]) -> dict[str, PlayerSimStats]:
    """Aggregate individual player performance across all simulations.

    Returns a dict keyed by player_id. Per-game stats (e.g. hits_per_game) are
    the mean across all N simulations, with standard deviations alongside them.
    This powers player prop bets: 'Will Player X get over 1.5 hits?' is a direct
    lookup against the hits distribution.
    """
    # per_game[player_id] -> list of per-game stat dicts, one entry per simulation
    per_game: dict[str, list[dict]] = defaultdict(list)

    for game in games:
        # Tally stats for each player within this one simulation
        game_stats: dict[str, dict] = defaultdict(
            lambda: {'pa': 0, 'hits': 0, 'hr': 0, 'bb': 0, 'k': 0, 'tb': 0, 'rbi': 0}
        )
        for pa in game.pa_results:
            pid = pa.batter_id
            s = game_stats[pid]
            s['pa'] += 1
            if pa.outcome in _HIT_OUTCOMES:
                s['hits'] += 1
                s['tb'] += _TOTAL_BASES[pa.outcome]
            if pa.outcome == Outcome.HR:
                s['hr'] += 1
            if pa.outcome in (Outcome.BB, Outcome.HBP):
                s['bb'] += 1
            if pa.outcome == Outcome.K:
                s['k'] += 1
            s['rbi'] += pa.runs_scored

        for pid, s in game_stats.items():
            per_game[pid].append(s)

    result: dict[str, PlayerSimStats] = {}
    for pid, game_stats_list in per_game.items():
        hits = np.array([s['hits'] for s in game_stats_list], dtype=float)
        hr = np.array([s['hr'] for s in game_stats_list], dtype=float)
        bb = np.array([s['bb'] for s in game_stats_list], dtype=float)
        k = np.array([s['k'] for s in game_stats_list], dtype=float)
        tb = np.array([s['tb'] for s in game_stats_list], dtype=float)
        pa = np.array([s['pa'] for s in game_stats_list], dtype=float)
        rbi = np.array([s['rbi'] for s in game_stats_list], dtype=float)

        result[pid] = PlayerSimStats(
            player_id=pid,
            name=pid,
            pa_per_game=float(np.mean(pa)),
            hits_per_game=float(np.mean(hits)),
            hr_per_game=float(np.mean(hr)),
            bb_per_game=float(np.mean(bb)),
            k_per_game=float(np.mean(k)),
            runs_per_game=float(np.mean(rbi)),
            total_bases_per_game=float(np.mean(tb)),
            hits_per_game_std=float(np.std(hits)),
            hr_per_game_std=float(np.std(hr)),
            bb_per_game_std=float(np.std(bb)),
            k_per_game_std=float(np.std(k)),
            total_bases_per_game_std=float(np.std(tb)),
        )

    return result
