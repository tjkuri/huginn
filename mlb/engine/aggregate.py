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
