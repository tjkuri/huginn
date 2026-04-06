"""Diagnostic script for league-average run production calibration."""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from mlb.config import Hand, LEAGUE_AVERAGES, Outcome, SEASON
from mlb.data.builder import build_game_context
from mlb.data.lineups import fetch_todays_games
from mlb.data.models import BatterStats, GameContext, Lineup, ParkFactors, PitcherStats
from mlb.data.stats import fetch_batting_splits, fetch_pitching_splits
from mlb.engine.aggregate import run_simulations
from mlb.engine.probabilities import (
    apply_park_factors,
    apply_weather_adjustments,
    build_pa_probability_table,
    compute_matchup_rates,
)


def _neutral_park() -> ParkFactors:
    neutral = {outcome.value: 1.0 for outcome in Outcome}
    return ParkFactors(
        venue_id="neutral",
        venue_name="Neutral Park",
        factors_vs_lhb=dict(neutral),
        factors_vs_rhb=dict(neutral),
    )


def _league_average_batter(player_id: str, hand: Hand, pitcher_hand: Hand) -> BatterStats:
    effective_hand = Hand.LEFT if hand == Hand.SWITCH and pitcher_hand == Hand.RIGHT else (
        Hand.RIGHT if hand == Hand.SWITCH else hand
    )
    rates = LEAGUE_AVERAGES[(effective_hand, pitcher_hand)]
    return BatterStats(player_id=player_id, name=player_id, bats=hand, pa=600, rates=dict(rates))


def _league_average_pitcher(player_id: str, throws: Hand, batter_hand: Hand) -> PitcherStats:
    effective_batter = Hand.LEFT if batter_hand == Hand.SWITCH and throws == Hand.RIGHT else (
        Hand.RIGHT if batter_hand == Hand.SWITCH else batter_hand
    )
    rates = LEAGUE_AVERAGES[(effective_batter, throws)]
    return PitcherStats(
        player_id=player_id,
        name=player_id,
        throws=throws,
        pa_against=700,
        rates=dict(rates),
        avg_pitch_count=95.0,
    )


def build_league_average_context() -> GameContext:
    """Build a neutral context using league-average rates for all players."""
    away_lineup = Lineup(
        team_id="AWY",
        team_name="League Avg Away",
        batting_order=[_league_average_batter(f"ab{i}", Hand.RIGHT, Hand.RIGHT) for i in range(1, 10)],
        starting_pitcher=_league_average_pitcher("asp", Hand.RIGHT, Hand.RIGHT),
        bullpen=[_league_average_pitcher("away-bullpen", Hand.RIGHT, Hand.RIGHT)],
    )
    home_lineup = Lineup(
        team_id="HME",
        team_name="League Avg Home",
        batting_order=[_league_average_batter(f"hb{i}", Hand.RIGHT, Hand.RIGHT) for i in range(1, 10)],
        starting_pitcher=_league_average_pitcher("hsp", Hand.RIGHT, Hand.RIGHT),
        bullpen=[_league_average_pitcher("home-bullpen", Hand.RIGHT, Hand.RIGHT)],
    )
    return GameContext(
        game_id="calibration",
        date="2026-04-03",
        away_lineup=away_lineup,
        home_lineup=home_lineup,
        park_factors=_neutral_park(),
        weather=None,
    )


def summarize_games(games) -> dict[str, float]:
    """Compute aggregate diagnostics from a batch of simulated games."""
    total_runs = []
    away_runs = []
    home_runs = []
    total_pas = []
    total_hits = []
    total_walks = []
    total_ks = []
    total_hrs = []
    total_outs = []

    for game in games:
        counts = Counter(pa.outcome for pa in game.pa_results)
        total_runs.append(game.away_runs + game.home_runs)
        away_runs.append(game.away_runs)
        home_runs.append(game.home_runs)
        total_pas.append(len(game.pa_results))
        total_hits.append(
            counts[Outcome.SINGLE] + counts[Outcome.DOUBLE] + counts[Outcome.TRIPLE] + counts[Outcome.HR]
        )
        total_walks.append(counts[Outcome.BB] + counts[Outcome.HBP])
        total_ks.append(counts[Outcome.K])
        total_hrs.append(counts[Outcome.HR])
        total_outs.append(counts[Outcome.K] + counts[Outcome.OUT])

    return {
        "mean_total_runs": float(np.mean(total_runs)),
        "mean_away_runs": float(np.mean(away_runs)),
        "mean_home_runs": float(np.mean(home_runs)),
        "mean_total_pas": float(np.mean(total_pas)),
        "mean_hits": float(np.mean(total_hits)),
        "mean_walks": float(np.mean(total_walks)),
        "mean_ks": float(np.mean(total_ks)),
        "mean_hrs": float(np.mean(total_hrs)),
        "mean_outs": float(np.mean(total_outs)),
    }


def _today_str() -> str:
    from datetime import date

    return date.today().isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MLB calibration and bullpen diagnostics.")
    parser.add_argument("--team", help="Print bullpen-rate diagnostics for a real game involving this team.")
    parser.add_argument("--date", default=_today_str(), help="Date for the real-game bullpen diagnostic (YYYY-MM-DD).")
    return parser


def _format_rate(value: float) -> str:
    return f"{value:.3f}"


def print_bullpen_diagnostic(team: str, target_date: str) -> None:
    games = fetch_todays_games(date=target_date)
    matching_games = [
        game for game in games
        if team.strip().lower() in str(game.get("away_team", "")).lower()
        or team.strip().lower() in str(game.get("home_team", "")).lower()
    ]
    if not matching_games:
        raise ValueError(f"No game found for {team} on {target_date}")

    game = matching_games[0]
    batting_data = fetch_batting_splits(season=SEASON)
    pitching_data = fetch_pitching_splits(season=SEASON)
    context = build_game_context(game, batting_data, pitching_data)

    league = LEAGUE_AVERAGES[(Hand.RIGHT, Hand.RIGHT)]
    bullpens = [
        (context.away_lineup.team_name, context.away_lineup.bullpen[0]),
        (context.home_lineup.team_name, context.home_lineup.bullpen[0]),
    ]

    print(f"\nBullpen rate diagnostic for {context.away_lineup.team_name} @ {context.home_lineup.team_name}")
    print(f"Date: {target_date}")
    print("Rates shown: bullpen vs league-average RHP baseline")
    for team_name, bullpen in bullpens:
        rates = bullpen.rates
        print(f"\n{team_name} bullpen")
        print(
            "  "
            f"K {_format_rate(rates['K'])} vs {_format_rate(league['K'])} | "
            f"BB {_format_rate(rates['BB'])} vs {_format_rate(league['BB'])} | "
            f"HR {_format_rate(rates['HR'])} vs {_format_rate(league['HR'])} | "
            f"OUT {_format_rate(rates['OUT'])} vs {_format_rate(league['OUT'])}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the calibration diagnostics."""
    args = build_parser().parse_args(argv)
    league = LEAGUE_AVERAGES[(Hand.RIGHT, Hand.RIGHT)]
    batter = BatterStats("b", "LeagueAvgBatter", Hand.RIGHT, 600, dict(league))
    pitcher = PitcherStats("p", "LeagueAvgPitcher", Hand.RIGHT, 700, dict(league), 95.0)
    park = _neutral_park()

    raw = compute_matchup_rates(batter, pitcher, league)
    park_adjusted = apply_park_factors(raw, park.get_factors(Hand.RIGHT))
    weather_adjusted = apply_weather_adjustments(park_adjusted, None)
    final = build_pa_probability_table(batter, pitcher, park, None, LEAGUE_AVERAGES)

    print("League-average matchup diagnostic")
    print("league:", league)
    print("raw:", raw)
    print(f"pre-normalization sum: {sum(weather_adjusted.values()):.6f}")
    print("final:", final)

    context = build_league_average_context()
    games = run_simulations(context, LEAGUE_AVERAGES, n_simulations=1000, base_seed=42)
    summary = summarize_games(games)

    print("\nSimulation summary (1000 games)")
    for key, value in summary.items():
        print(f"{key}: {value:.3f}")

    if args.team:
        print_bullpen_diagnostic(args.team, args.date)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
