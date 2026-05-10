from __future__ import annotations

from collections import Counter
from functools import lru_cache
from statistics import mean, pstdev

import numpy as np

from mlb.config import SEASON, Outcome
from mlb.data.park_factors import TEAM_TO_VENUE, get_park_factors
from mlb.data.stats import (
    build_batter_stats,
    build_pitcher_stats,
    fetch_batting_splits,
    fetch_pitching_splits,
    fetch_runtime_league_averages,
    fetch_team_bullpen_stats,
)
from mlb.engine.probabilities import build_pa_probability_table
from mlb.engine.simulate import resolve_pa_outcome

MIN_BATTER_PA = 75
MIN_STARTER_PA_AGAINST = 100
STARTER_SAMPLE_COUNT = 20
TOP_POWER_HITTER_COUNT = 20
ELITE_STARTER_COUNT = 5
PA_SAMPLES_PER_MATCHUP = 100
# Independent 2025 MLB season-end PA outcome rates from Baseball Reference.
INDEPENDENT_2025_SEASON_RATES = {
    "K": 0.224,
    "BB": 0.082,
    "HR": 0.031,
    "1B": 0.151,
    "2B": 0.043,
    "3B": 0.004,
    "HBP": 0.011,
    "OUT": 0.454,
}


def _is_starter_profile(pitcher_profile: dict[str, object]) -> bool:
    player_id = str(pitcher_profile.get("player_id") or "")
    if player_id.endswith("-bullpen"):
        return False
    return int(pitcher_profile.get("pa_against") or 0) >= MIN_STARTER_PA_AGAINST


@lru_cache(maxsize=1)
def _build_matchup_fixture() -> dict[str, object]:
    batting_data = fetch_batting_splits(season=SEASON, use_cache=True)
    pitching_data = fetch_pitching_splits(season=SEASON, use_cache=True)
    bullpen_data = fetch_team_bullpen_stats(season=SEASON, use_cache=True)
    matchup_league_averages = fetch_runtime_league_averages(season=SEASON, use_cache=True)

    qualified_batters = [
        batter
        for batter in batting_data.values()
        if int(batter.get("pa") or 0) >= MIN_BATTER_PA
    ]
    top_starter_profiles = sorted(
        (profile for profile in pitching_data.values() if _is_starter_profile(profile)),
        key=lambda profile: int(profile.get("pa_against") or 0),
        reverse=True,
    )[:STARTER_SAMPLE_COUNT]
    # League K%/HR% are season-wide and include reliever PAs (~40% of league),
    # so the pitcher pool must include bullpen aggregates to calibrate against them.
    pitcher_profiles = top_starter_profiles + list(bullpen_data.values())

    venue_rotation = sorted(set(TEAM_TO_VENUE.values()))
    starter_contexts = [
        {
            "pitcher": build_pitcher_stats(profile),
            "park_factors": get_park_factors(venue_rotation[i % len(venue_rotation)]),
        }
        for i, profile in enumerate(pitcher_profiles)
    ]

    rows: list[dict[str, object]] = []
    for batter_profile in qualified_batters:
        for starter_context in starter_contexts:
            pitcher = starter_context["pitcher"]
            batter = build_batter_stats(
                batter_profile,
                batter_profile.get("bats", "R"),
                pitcher_hand=pitcher.throws.value,
            )
            probabilities = build_pa_probability_table(
                batter,
                pitcher,
                starter_context["park_factors"],
                None,  # isolate player+park calibration; weather adds slate-specific noise
                matchup_league_averages,
            )
            rows.append(
                {
                    "batter_id": batter.player_id,
                    "pitcher_id": pitcher.player_id,
                    "observed_batter_hr": batter.rates["HR"],
                    "probabilities": probabilities,
                }
            )

    power_hitter_ids = {
        str(batter.get("player_id") or batter.get("id") or batter.get("name"))
        for batter in sorted(
            qualified_batters,
            key=lambda item: float((item.get("rates") or {}).get("HR", 0.0)),
            reverse=True,
        )[:TOP_POWER_HITTER_COUNT]
    }

    unique_starters = {}
    for starter_context in starter_contexts:
        starter = starter_context["pitcher"]
        unique_starters[starter.player_id] = starter
    elite_pitcher_ids = {
        starter.player_id
        for starter in sorted(
            unique_starters.values(),
            key=lambda item: item.rates["K"],
            reverse=True,
        )[:ELITE_STARTER_COUNT]
    }

    return {
        "rows": rows,
        "actual_rates": dict(INDEPENDENT_2025_SEASON_RATES),
        "power_hitter_ids": power_hitter_ids,
        "elite_pitcher_ids": elite_pitcher_ids,
        "starter_count": len(unique_starters),
    }


def _simulate_aggregate_outcomes(rows: list[dict[str, object]]) -> dict[str, float]:
    rng = np.random.default_rng(20260420)
    counts: Counter[str] = Counter()
    for row in rows:
        probabilities = row["probabilities"]
        for _ in range(PA_SAMPLES_PER_MATCHUP):
            outcome = resolve_pa_outcome(probabilities, rng)
            counts[outcome.value] += 1

    total = sum(counts.values())
    return {
        outcome.value: counts[outcome.value] / total
        for outcome in Outcome
    }


def _format_summary_table(simulated_rates: dict[str, float], actual_rates: dict[str, float]) -> str:
    outcomes = [outcome.value for outcome in Outcome]
    header = f"{'Outcome':<7} {'Simulated':>10} {'Actual':>10} {'Diff':>10}"
    lines = [header, "-" * len(header)]
    for outcome in outcomes:
        simulated = simulated_rates[outcome]
        actual = actual_rates[outcome]
        diff = simulated - actual
        lines.append(
            f"{outcome:<7} {simulated:>9.2%} {actual:>9.2%} {diff:>+9.2%}"
        )
    return "\n".join(lines)


def test_full_league_day_matchup_calibration():
    fixture = _build_matchup_fixture()
    rows = fixture["rows"]
    actual_rates = fixture["actual_rates"]
    simulated_rates = _simulate_aggregate_outcomes(rows)

    simulated_hr_std = pstdev(row["probabilities"]["HR"] for row in rows)
    observed_batter_hr_std = pstdev(row["observed_batter_hr"] for row in rows)

    league_matchup_hr = mean(row["probabilities"]["HR"] for row in rows)
    league_matchup_k = mean(row["probabilities"]["K"] for row in rows)
    power_hitter_hr = mean(
        row["probabilities"]["HR"]
        for row in rows
        if row["batter_id"] in fixture["power_hitter_ids"]
    )
    elite_pitcher_k = mean(
        row["probabilities"]["K"]
        for row in rows
        if row["pitcher_id"] in fixture["elite_pitcher_ids"]
    )

    print(
        "\n"
        f"League-day matchup summary "
        f"({len(rows)} matchups, {fixture['starter_count']} starters, "
        f"{PA_SAMPLES_PER_MATCHUP} sampled PA per matchup)\n"
        "Comparison baseline: independent 2025 season-end MLB rates "
        "(Baseball Reference)\n"
        f"{_format_summary_table(simulated_rates, actual_rates)}"
    )

    assert abs(simulated_rates["K"] - actual_rates["K"]) <= 0.01
    assert abs(simulated_rates["HR"] - actual_rates["HR"]) <= 0.005
    assert observed_batter_hr_std * 0.90 <= simulated_hr_std <= observed_batter_hr_std * 1.35
    assert power_hitter_hr >= league_matchup_hr * 1.50
    assert elite_pitcher_k >= league_matchup_k * 1.15
