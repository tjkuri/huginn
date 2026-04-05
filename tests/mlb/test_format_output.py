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
    SimulationResult,
    Weather,
)
from mlb.engine.aggregate import compute_betting_lines, compute_player_stats, compute_run_distributions, compute_win_probability
from mlb.scripts import format_output
from rich.console import Console


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


def _bullpen(player_id: str, name: str) -> PitcherStats:
    return PitcherStats(
        player_id=player_id,
        name=name,
        throws=Hand.RIGHT,
        pa_against=300,
        rates={"K": 0.24, "BB": 0.09, "HBP": 0.0, "1B": 0.14, "2B": 0.05, "3B": 0.01, "HR": 0.03, "OUT": 0.44},
        avg_pitch_count=120.0,
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


def test_plain_report_uses_prop_market_projection_columns():
    away_pitcher = _pitcher("ap", "Away Starter")
    home_pitcher = _pitcher("hp", "Home Starter")
    away_batter = _batter("ab", "Away Batter")
    home_batter = _batter("hb", "Home Batter")
    context = GameContext(
        game_id="g1",
        date="2026-04-04",
        away_lineup=Lineup(
            team_id="away",
            team_name="Synthetic Away",
            batting_order=[away_batter] + [_batter(f"ab{i}", f"Away Batter {i}") for i in range(2, 10)],
            starting_pitcher=away_pitcher,
            bullpen=[_bullpen("abp", "Synthetic Away Bullpen")],
        ),
        home_lineup=Lineup(
            team_id="home",
            team_name="Synthetic Home",
            batting_order=[home_batter] + [_batter(f"hb{i}", f"Home Batter {i}") for i in range(2, 10)],
            starting_pitcher=home_pitcher,
            bullpen=[_bullpen("hbp", "Synthetic Home Bullpen")],
        ),
        park_factors=ParkFactors(
            venue_id="v1",
            venue_name="Test Park",
            factors_vs_lhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
            factors_vs_rhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
        ),
        weather=Weather(72.0, 5.0, WindDirection.CALM, 50.0, is_indoor=False),
    )

    games = []
    for sim in range(10):
        pa_results = [_pa(Outcome.SINGLE, "ab", "hp")]
        pa_results.extend(_pa(Outcome.K, "hb", "ap") for _ in range(5 if sim < 5 else 4))
        pa_results.extend(_pa(Outcome.OUT, "hb", "ap") for _ in range(13 if sim < 5 else 11))
        pa_results.append(_pa(Outcome.HR, "hb", "ap", runs_scored=2 if sim < 5 else 4))
        games.append(
            SimulatedGame(
                game_id=f"g-{sim}",
                away_runs=1,
                home_runs=2,
                away_hits=1,
                home_hits=1,
                pa_results=pa_results,
                innings_played=9,
                inning_scores={"away": [0] * 9, "home": [0] * 9},
            )
        )

    run_dists = compute_run_distributions(games)
    win_probs = compute_win_probability(games)
    player_stats = compute_player_stats(games)
    result = SimulationResult(
        game_id=context.game_id,
        n_simulations=len(games),
        away_team=context.away_lineup.team_name,
        home_team=context.home_lineup.team_name,
        away_runs_mean=run_dists["away_runs"]["mean"],
        away_runs_std=run_dists["away_runs"]["std"],
        home_runs_mean=run_dists["home_runs"]["mean"],
        home_runs_std=run_dists["home_runs"]["std"],
        total_runs_mean=run_dists["total_runs"]["mean"],
        total_runs_std=run_dists["total_runs"]["std"],
        home_win_pct=win_probs["home_win_pct"],
        away_win_pct=win_probs["away_win_pct"],
        player_stats=player_stats,
        betting_lines=compute_betting_lines(games, run_dists),
        run_distributions=run_dists,
    )

    original_has_rich = format_output.HAS_RICH
    format_output.HAS_RICH = False
    try:
        report = format_output.build_terminal_output(result, context, None, None, [], games)
    finally:
        format_output.HAS_RICH = original_has_rich

    assert "PLAYER PROJECTIONS" not in report
    assert "TB " in report
    assert " SLG " not in report
    assert "Starting pitchers:" in report
    assert "Bullpen:" in report
    assert "5+K%" in report
    assert "QS%" in report
    assert "K/9" not in report
    assert "ERA*" not in report


def test_rich_report_hides_quality_panel_when_there_are_no_warnings():
    away_pitcher = _pitcher("ap", "Away Starter")
    home_pitcher = _pitcher("hp", "Home Starter")
    away_batter = _batter("ab", "Away Batter")
    home_batter = _batter("hb", "Home Batter")
    context = GameContext(
        game_id="g2",
        date="2026-04-04",
        away_lineup=Lineup(
            team_id="away",
            team_name="Synthetic Away",
            batting_order=[away_batter] + [_batter(f"ab{i}", f"Away Batter {i}") for i in range(2, 10)],
            starting_pitcher=away_pitcher,
            bullpen=[_bullpen("abp", "Synthetic Away Bullpen")],
        ),
        home_lineup=Lineup(
            team_id="home",
            team_name="Synthetic Home",
            batting_order=[home_batter] + [_batter(f"hb{i}", f"Home Batter {i}") for i in range(2, 10)],
            starting_pitcher=home_pitcher,
            bullpen=[_bullpen("hbp", "Synthetic Home Bullpen")],
        ),
        park_factors=ParkFactors(
            venue_id="v1",
            venue_name="Test Park",
            factors_vs_lhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
            factors_vs_rhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
        ),
        weather=Weather(70.0, 4.0, WindDirection.CROSS, 45.0, is_indoor=False),
    )

    games = [
        SimulatedGame(
            game_id="g-1",
            away_runs=1,
            home_runs=2,
            away_hits=1,
            home_hits=1,
            pa_results=[_pa(Outcome.SINGLE, "ab", "hp"), _pa(Outcome.OUT, "hb", "ap")],
            innings_played=9,
            inning_scores={"away": [0] * 9, "home": [0] * 9},
        )
    ]
    run_dists = compute_run_distributions(games)
    win_probs = compute_win_probability(games)
    player_stats = compute_player_stats(games)
    result = SimulationResult(
        game_id=context.game_id,
        n_simulations=len(games),
        away_team=context.away_lineup.team_name,
        home_team=context.home_lineup.team_name,
        away_runs_mean=run_dists["away_runs"]["mean"],
        away_runs_std=run_dists["away_runs"]["std"],
        home_runs_mean=run_dists["home_runs"]["mean"],
        home_runs_std=run_dists["home_runs"]["std"],
        total_runs_mean=run_dists["total_runs"]["mean"],
        total_runs_std=run_dists["total_runs"]["std"],
        home_win_pct=win_probs["home_win_pct"],
        away_win_pct=win_probs["away_win_pct"],
        player_stats=player_stats,
        betting_lines=compute_betting_lines(games, run_dists),
        run_distributions=run_dists,
    )

    renderable = format_output.build_terminal_output(result, context, None, None, [], games)
    console = Console(width=90, record=True)
    console.print(renderable)
    output = console.export_text()

    assert "DATA QUALITY" not in output
    assert "Bullpen" in output


def test_rich_report_shows_quality_panel_for_missing_players_only():
    away_pitcher = _pitcher("ap", "Away Starter")
    home_pitcher = _pitcher("hp", "Home Starter")
    away_batter = _batter("ab", "Away Batter")
    home_batter = _batter("hb", "Home Batter")
    context = GameContext(
        game_id="g3",
        date="2026-04-04",
        away_lineup=Lineup(
            team_id="away",
            team_name="Synthetic Away",
            batting_order=[away_batter] + [_batter(f"ab{i}", f"Away Batter {i}") for i in range(2, 10)],
            starting_pitcher=away_pitcher,
            bullpen=[_bullpen("abp", "Synthetic Away Bullpen")],
        ),
        home_lineup=Lineup(
            team_id="home",
            team_name="Synthetic Home",
            batting_order=[home_batter] + [_batter(f"hb{i}", f"Home Batter {i}") for i in range(2, 10)],
            starting_pitcher=home_pitcher,
            bullpen=[_bullpen("hbp", "Synthetic Home Bullpen")],
        ),
        park_factors=ParkFactors(
            venue_id="v1",
            venue_name="Test Park",
            factors_vs_lhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
            factors_vs_rhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
        ),
        weather=Weather(72.0, 5.0, WindDirection.CALM, 50.0, is_indoor=False),
    )

    games = [
        SimulatedGame(
            game_id="g-1",
            away_runs=1,
            home_runs=2,
            away_hits=1,
            home_hits=1,
            pa_results=[_pa(Outcome.SINGLE, "ab", "hp"), _pa(Outcome.OUT, "hb", "ap")],
            innings_played=9,
            inning_scores={"away": [0] * 9, "home": [0] * 9},
        )
    ]
    run_dists = compute_run_distributions(games)
    win_probs = compute_win_probability(games)
    player_stats = compute_player_stats(games)
    result = SimulationResult(
        game_id=context.game_id,
        n_simulations=len(games),
        away_team=context.away_lineup.team_name,
        home_team=context.home_lineup.team_name,
        away_runs_mean=run_dists["away_runs"]["mean"],
        away_runs_std=run_dists["away_runs"]["std"],
        home_runs_mean=run_dists["home_runs"]["mean"],
        home_runs_std=run_dists["home_runs"]["std"],
        total_runs_mean=run_dists["total_runs"]["mean"],
        total_runs_std=run_dists["total_runs"]["std"],
        home_win_pct=win_probs["home_win_pct"],
        away_win_pct=win_probs["away_win_pct"],
        player_stats=player_stats,
        betting_lines=compute_betting_lines(games, run_dists),
        run_distributions=run_dists,
    )

    renderable = format_output.build_terminal_output(
        result,
        context,
        None,
        None,
        ["Missing batting data for Test Batter; using league-average fallback"],
        games,
    )
    console = Console(width=90, record=True)
    console.print(renderable)
    output = console.export_text()

    assert "DATA QUALITY" in output
    assert "Weather" in output
