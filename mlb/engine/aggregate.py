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
