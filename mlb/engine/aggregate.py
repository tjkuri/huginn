"""Simulation aggregation engine.

Runs simulate_game N times and aggregates results into projections and
betting-relevant outputs. One GameContext in, one SimulationResult out.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

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


@dataclass
class PitcherSimStats(PlayerSimStats):
    """Extended aggregate fields for pitcher prop markets."""
    innings_pitched_per_game: float = 0.0
    k_5_plus_pct: float = 0.0
    quality_start_pct: float = 0.0


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
    pitcher_per_game: dict[str, list[dict]] = defaultdict(list)

    for sim_index, game in enumerate(games):
        # Tally stats for each player within this one simulation
        game_stats: dict[str, dict] = defaultdict(
            lambda: {
                'pa': 0, 'hits': 0, '2b': 0, 'hr': 0,
                'bb': 0, 'hbp': 0, 'k': 0, 'tb': 0, 'rbi': 0,
            }
        )
        pitcher_game_stats: dict[str, dict] = defaultdict(
            lambda: {
                'outs': 0, 'k': 0, 'er': 0,
            }
        )
        for pa in game.pa_results:
            pid = pa.batter_id
            s = game_stats[pid]
            s['pa'] += 1
            if pa.outcome in _HIT_OUTCOMES:
                s['hits'] += 1
                s['tb'] += _TOTAL_BASES[pa.outcome]
            if pa.outcome == Outcome.DOUBLE:
                s['2b'] += 1
            if pa.outcome == Outcome.HR:
                s['hr'] += 1
            if pa.outcome == Outcome.BB:
                s['bb'] += 1
            if pa.outcome == Outcome.HBP:
                s['hbp'] += 1
            if pa.outcome == Outcome.K:
                s['k'] += 1
            s['rbi'] += pa.runs_scored

            pitcher_id = pa.pitcher_id
            pitcher_stats = pitcher_game_stats[pitcher_id]
            if pa.outcome in {Outcome.K, Outcome.OUT}:
                pitcher_stats['outs'] += 1
            if pa.outcome == Outcome.K:
                pitcher_stats['k'] += 1
            pitcher_stats['er'] += pa.runs_scored

        for pid, s in game_stats.items():
            per_game[pid].append(s)
        for pid in list(per_game.keys()):
            if len(per_game[pid]) < sim_index + 1:
                per_game[pid].append(
                    {'pa': 0, 'hits': 0, '2b': 0, 'hr': 0, 'bb': 0, 'hbp': 0, 'k': 0, 'tb': 0, 'rbi': 0}
                )

        for pid, s in pitcher_game_stats.items():
            if len(pitcher_per_game[pid]) < sim_index:
                pitcher_per_game[pid].extend(
                    [{'outs': 0, 'k': 0, 'er': 0}] * (sim_index - len(pitcher_per_game[pid]))
                )
            pitcher_per_game[pid].append(s)
        for pid in list(pitcher_per_game.keys()):
            if len(pitcher_per_game[pid]) < sim_index + 1:
                pitcher_per_game[pid].append({'outs': 0, 'k': 0, 'er': 0})

    result: dict[str, PlayerSimStats] = {}
    for pid, game_stats_list in per_game.items():
        hits = np.array([s['hits'] for s in game_stats_list], dtype=float)
        doubles = np.array([s['2b'] for s in game_stats_list], dtype=float)
        hr = np.array([s['hr'] for s in game_stats_list], dtype=float)
        bb = np.array([s['bb'] for s in game_stats_list], dtype=float)
        hbp = np.array([s['hbp'] for s in game_stats_list], dtype=float)
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
            doubles_per_game=float(np.mean(doubles)),
            hbp_per_game=float(np.mean(hbp)),
            total_bases_per_game=float(np.mean(tb)),
            hits_per_game_std=float(np.std(hits)),
            hr_per_game_std=float(np.std(hr)),
            bb_per_game_std=float(np.std(bb)),
            k_per_game_std=float(np.std(k)),
            total_bases_per_game_std=float(np.std(tb)),
        )

    for pid, game_stats_list in pitcher_per_game.items():
        outs = np.array([s['outs'] for s in game_stats_list], dtype=float)
        strikeouts = np.array([s['k'] for s in game_stats_list], dtype=float)
        earned_runs = np.array([s['er'] for s in game_stats_list], dtype=float)
        innings_pitched = outs / 3.0

        result[pid] = PitcherSimStats(
            player_id=pid,
            name=pid,
            pa_per_game=0.0,
            hits_per_game=0.0,
            hr_per_game=0.0,
            bb_per_game=0.0,
            k_per_game=float(np.mean(strikeouts)),
            runs_per_game=float(np.mean(earned_runs)),
            innings_pitched_per_game=float(np.mean(innings_pitched)),
            k_per_game_std=float(np.std(strikeouts)),
            k_5_plus_pct=float(np.mean(strikeouts >= 5.0)),
            quality_start_pct=float(np.mean((innings_pitched >= 6.0) & (earned_runs <= 3.0))),
        )

    return result


def _american_odds(probability: float) -> float:
    """Convert a win probability to no-vig American odds."""
    probability = max(0.001, min(0.999, probability))
    if probability > 0.5:
        return -(probability / (1.0 - probability)) * 100
    elif probability < 0.5:
        return ((1.0 - probability) / probability) * 100
    return 100.0  # even


def _over_under_table(run_values: np.ndarray, lines: list[float]) -> dict:
    """Build an over/under table for a sequence of run values."""
    n = len(run_values)
    return {
        line: {
            'over_pct': float(np.sum(run_values > line) / n),
            'under_pct': float(np.sum(run_values < line) / n),
            'push_pct': float(np.sum(run_values == line) / n),
        }
        for line in lines
    }


def compute_betting_lines(games: list[SimulatedGame], run_distributions: dict) -> dict:
    """Compute model-implied probabilities for common bet types.

    Args:
        games: list of SimulatedGame from run_simulations.
        run_distributions: output of compute_run_distributions (passed for context;
            raw game data is used directly for probability calculations).

    Returns a nested dict with keys: totals, moneyline, run_line, team_totals.
    """
    n = len(games)
    away_arr = np.array([g.away_runs for g in games], dtype=float)
    home_arr = np.array([g.home_runs for g in games], dtype=float)
    total_arr = away_arr + home_arr

    # ── Game totals ──────────────────────────────────────────────────────────
    total_lines = [l / 2 for l in range(11, 26)]  # 5.5 to 12.5 in 0.5 steps
    totals = _over_under_table(total_arr, total_lines)

    # ── Moneyline ────────────────────────────────────────────────────────────
    home_win_pct = float(np.sum(home_arr > away_arr) / n)
    away_win_pct = float(np.sum(away_arr > home_arr) / n)
    moneyline = {
        'home': {
            'probability': home_win_pct,
            'american': _american_odds(home_win_pct),
            'no_vig_line': _american_odds(home_win_pct),
        },
        'away': {
            'probability': away_win_pct,
            'american': _american_odds(away_win_pct),
            'no_vig_line': _american_odds(away_win_pct),
        },
    }

    # ── Run line (-1.5 / +1.5) ───────────────────────────────────────────────
    diff = home_arr - away_arr  # positive = home leads
    if home_win_pct >= away_win_pct:
        favorite_cover = float(np.sum(diff >= 2) / n)    # home wins by 2+
        underdog_cover = float(np.sum(diff <= 1) / n)    # home wins by <=1 or away wins
    else:
        favorite_cover = float(np.sum(diff <= -2) / n)   # away wins by 2+
        underdog_cover = float(np.sum(diff >= -1) / n)   # away wins by <=1 or home wins

    run_line = {
        'favorite': 'home' if home_win_pct >= away_win_pct else 'away',
        'favorite_cover_pct': favorite_cover,
        'underdog_cover_pct': underdog_cover,
    }

    # ── Team totals ──────────────────────────────────────────────────────────
    team_lines = [l / 2 for l in range(5, 18)]  # 2.5 to 8.5 in 0.5 steps
    team_totals = {
        'away': _over_under_table(away_arr, team_lines),
        'home': _over_under_table(home_arr, team_lines),
    }

    return {
        'totals': totals,
        'moneyline': moneyline,
        'run_line': run_line,
        'team_totals': team_totals,
    }


def aggregate_simulations(
    game_context: GameContext,
    league_averages: dict,
    n_simulations: int = 10000,
    base_seed: int | None = None,
) -> SimulationResult:
    """Run N simulations and return a fully populated SimulationResult.

    This is the main entry point for the aggregation module. All bet-relevant
    outputs (run distributions, win probabilities, player stats, betting lines)
    are computed here and packed into a single SimulationResult.
    """
    games = run_simulations(game_context, league_averages, n_simulations, base_seed)

    run_dists = compute_run_distributions(games)
    win_probs = compute_win_probability(games)
    player_stats = compute_player_stats(games)
    betting_lines = compute_betting_lines(games, run_dists)

    return SimulationResult(
        game_id=game_context.game_id,
        n_simulations=n_simulations,
        away_team=game_context.away_lineup.team_name,
        home_team=game_context.home_lineup.team_name,
        away_runs_mean=run_dists['away_runs']['mean'],
        away_runs_std=run_dists['away_runs']['std'],
        home_runs_mean=run_dists['home_runs']['mean'],
        home_runs_std=run_dists['home_runs']['std'],
        total_runs_mean=run_dists['total_runs']['mean'],
        total_runs_std=run_dists['total_runs']['std'],
        home_win_pct=win_probs['home_win_pct'],
        away_win_pct=win_probs['away_win_pct'],
        player_stats=player_stats,
        betting_lines=betting_lines,
        run_distributions=run_dists,
    )
