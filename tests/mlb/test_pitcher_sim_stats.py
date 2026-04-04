from __future__ import annotations

from mlb.config import Hand, Outcome, WindDirection
from mlb.data.models import (
    BaseState,
    BatterStats,
    GameContext,
    Lineup,
    PAResult,
    ParkFactors,
    PitcherStats,
    SimulatedGame,
    Weather,
)
from mlb.engine.aggregate import compute_player_stats


def _batter(player_id: str, name: str) -> BatterStats:
    return BatterStats(
        player_id=player_id,
        name=name,
        bats=Hand.RIGHT,
        pa=100,
        rates={"K": 0.2, "BB": 0.1, "HBP": 0.0, "1B": 0.2, "2B": 0.1, "3B": 0.0, "HR": 0.05, "OUT": 0.35},
    )


def _pitcher(player_id: str, name: str) -> PitcherStats:
    return PitcherStats(
        player_id=player_id,
        name=name,
        throws=Hand.RIGHT,
        pa_against=100,
        rates={"K": 0.25, "BB": 0.08, "HBP": 0.0, "1B": 0.15, "2B": 0.05, "3B": 0.01, "HR": 0.03, "OUT": 0.43},
        avg_pitch_count=90.0,
    )


def _context() -> GameContext:
    away_pitcher = _pitcher("p1", "Starter One")
    home_pitcher = _pitcher("p2", "Starter Two")
    away_batter = _batter("b1", "Batter One")
    home_batter = _batter("b2", "Batter Two")
    return GameContext(
        game_id="g1",
        date="2026-04-04",
        away_lineup=Lineup(
            team_id="away",
            team_name="Synthetic Away",
            batting_order=[away_batter] * 9,
            starting_pitcher=away_pitcher,
            bullpen=[],
        ),
        home_lineup=Lineup(
            team_id="home",
            team_name="Synthetic Home",
            batting_order=[home_batter] * 9,
            starting_pitcher=home_pitcher,
            bullpen=[],
        ),
        park_factors=ParkFactors(
            venue_id="v1",
            venue_name="Test Park",
            factors_vs_lhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
            factors_vs_rhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
        ),
        weather=Weather(72.0, 5.0, WindDirection.CALM, 50.0, is_indoor=False),
    )


def _pa(outcome: Outcome, batter_id: str, pitcher_id: str, runs_scored: int = 0) -> PAResult:
    return PAResult(
        outcome=outcome,
        batter_id=batter_id,
        pitcher_id=pitcher_id,
        inning=1,
        runners_before=BaseState(),
        runs_scored=runs_scored,
    )


def test_compute_player_stats_total_bases_and_pitcher_threshold_props():
    games: list[SimulatedGame] = []
    for sim in range(100):
        pa_results = [_pa(Outcome.SINGLE, "b1", "p2")]
        if sim < 20:
            pa_results.append(_pa(Outcome.DOUBLE, "b1", "p2"))
        if sim < 5:
            pa_results.append(_pa(Outcome.TRIPLE, "b1", "p2"))
        if sim < 10:
            pa_results.append(_pa(Outcome.HR, "b1", "p2", runs_scored=1))

        starter_strikeouts = 5 if sim < 45 else 4
        starter_outs = 18 if sim < 60 else 15
        starter_er = 2 if sim < 60 else 4
        pa_results.extend(_pa(Outcome.K, "b2", "p1") for _ in range(starter_strikeouts))
        pa_results.extend(_pa(Outcome.OUT, "b2", "p1") for _ in range(starter_outs - starter_strikeouts))
        if starter_er:
            pa_results.append(_pa(Outcome.HR, "b2", "p1", runs_scored=starter_er))

        games.append(
            SimulatedGame(
                game_id=f"g-{sim}",
                away_runs=0,
                home_runs=0,
                away_hits=0,
                home_hits=0,
                pa_results=pa_results,
                innings_played=9,
                inning_scores={"away": [0] * 9, "home": [0] * 9},
            )
        )

    stats = compute_player_stats(games)

    batter_stats = stats["b1"]
    assert batter_stats.total_bases_per_game == 1.95

    pitcher_stats = stats["p1"]
    assert pitcher_stats.k_per_game == 4.45
    assert pitcher_stats.runs_per_game == 2.8
    assert pitcher_stats.innings_pitched_per_game == 5.6
    assert pitcher_stats.k_5_plus_pct == 0.45
    assert pitcher_stats.quality_start_pct == 0.60
