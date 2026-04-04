"""Synthetic end-to-end smoke test for the MLB simulation pipeline."""
from __future__ import annotations

from mlb.config import Hand, LEAGUE_AVERAGES, WindDirection
from mlb.data.models import BatterStats, GameContext, Lineup, ParkFactors, PitcherStats, Weather
from mlb.engine.aggregate import aggregate_simulations, run_simulations
from mlb.scripts.simulate_game import format_terminal_report, serialize_simulation_result


def _batter(player_id: str, name: str, bats: Hand) -> BatterStats:
    rates = {
        "K": 0.22,
        "BB": 0.09,
        "HBP": 0.01,
        "1B": 0.15,
        "2B": 0.05,
        "3B": 0.01,
        "HR": 0.04,
        "OUT": 0.43,
    }
    return BatterStats(player_id=player_id, name=name, bats=bats, pa=600, rates=rates)


def _pitcher(player_id: str, name: str, throws: Hand, avg_pitch_count: float) -> PitcherStats:
    rates = {
        "K": 0.24,
        "BB": 0.08,
        "HBP": 0.01,
        "1B": 0.14,
        "2B": 0.04,
        "3B": 0.01,
        "HR": 0.03,
        "OUT": 0.45,
    }
    return PitcherStats(
        player_id=player_id,
        name=name,
        throws=throws,
        pa_against=700,
        rates=rates,
        avg_pitch_count=avg_pitch_count,
    )


def build_synthetic_game_context() -> GameContext:
    """Create a complete synthetic game context for smoke testing."""
    away_lineup = Lineup(
        team_id="away",
        team_name="Synthetic Away",
        batting_order=[_batter(f"a{i}", f"Away Batter {i}", Hand.RIGHT) for i in range(1, 10)],
        starting_pitcher=_pitcher("ap", "Away Starter", Hand.RIGHT, 92.0),
        bullpen=[_pitcher(f"arp{i}", f"Away RP{i}", Hand.RIGHT, 26.0) for i in range(1, 5)],
    )
    home_lineup = Lineup(
        team_id="home",
        team_name="Synthetic Home",
        batting_order=[_batter(f"h{i}", f"Home Batter {i}", Hand.LEFT) for i in range(1, 10)],
        starting_pitcher=_pitcher("hp", "Home Starter", Hand.LEFT, 94.0),
        bullpen=[_pitcher(f"hrp{i}", f"Home RP{i}", Hand.LEFT if i % 2 else Hand.RIGHT, 26.0) for i in range(1, 5)],
    )
    return GameContext(
        game_id="smoke-game",
        date="2026-04-03",
        away_lineup=away_lineup,
        home_lineup=home_lineup,
        park_factors=ParkFactors(
            venue_id="smoke-park",
            venue_name="Smoke Test Park",
            factors_vs_lhb={"HR": 1.02, "2B": 1.01, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
            factors_vs_rhb={"HR": 1.01, "2B": 1.01, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
        ),
        weather=Weather(72.0, 5.0, WindDirection.CALM, 50.0, is_indoor=False),
    )


def main() -> int:
    """Run the full synthetic pipeline and validate the result shape."""
    context = build_synthetic_game_context()
    sample_games = run_simulations(context, LEAGUE_AVERAGES, n_simulations=100, base_seed=7)
    result = aggregate_simulations(context, LEAGUE_AVERAGES, n_simulations=100, base_seed=7)
    payload = serialize_simulation_result(result, context, seed=7, data_warnings=[])

    required_keys = {
        "game_id",
        "date",
        "venue",
        "away_team",
        "home_team",
        "total_runs_mean",
        "home_win_pct",
        "away_win_pct",
        "betting_lines",
        "player_stats",
        "metadata",
    }
    missing = required_keys - payload.keys()
    assert not missing, f"Smoke payload missing keys: {sorted(missing)}"

    renderable = format_terminal_report(
        result,
        context,
        [],
        sample_game=sample_games[6],
        sample_index=7,
        simulated_games=sample_games,
    )
    if isinstance(renderable, str):
        print(renderable)
    else:
        from rich.console import Console

        Console().print(renderable)
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
