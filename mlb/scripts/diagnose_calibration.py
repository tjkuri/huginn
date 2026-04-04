"""Diagnostic script for league-average run production calibration."""
from __future__ import annotations

from collections import Counter

import numpy as np

from mlb.config import Hand, LEAGUE_AVERAGES, Outcome
from mlb.data.models import BatterStats, GameContext, Lineup, ParkFactors, PitcherStats
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


def _league_average_batter(player_id: str, hand: Hand) -> BatterStats:
    rates = LEAGUE_AVERAGES[(hand, Hand.RIGHT if hand == Hand.LEFT else Hand.RIGHT)]
    return BatterStats(player_id=player_id, name=player_id, bats=hand, pa=600, rates=dict(rates))


def _league_average_pitcher(player_id: str, throws: Hand) -> PitcherStats:
    rates = (
        LEAGUE_AVERAGES[(Hand.RIGHT, Hand.LEFT)]
        if throws == Hand.LEFT
        else LEAGUE_AVERAGES[(Hand.RIGHT, Hand.RIGHT)]
    )
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
        batting_order=[_league_average_batter(f"ab{i}", Hand.RIGHT) for i in range(1, 10)],
        starting_pitcher=_league_average_pitcher("asp", Hand.RIGHT),
        bullpen=[_league_average_pitcher(f"arp{i}", Hand.RIGHT) for i in range(1, 5)],
    )
    home_lineup = Lineup(
        team_id="HME",
        team_name="League Avg Home",
        batting_order=[_league_average_batter(f"hb{i}", Hand.RIGHT) for i in range(1, 10)],
        starting_pitcher=_league_average_pitcher("hsp", Hand.RIGHT),
        bullpen=[_league_average_pitcher(f"hrp{i}", Hand.RIGHT) for i in range(1, 5)],
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


def main() -> int:
    """Run the calibration diagnostics."""
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
