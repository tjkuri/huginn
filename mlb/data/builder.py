"""Assembly helpers that turn fetched data into simulation-ready game contexts."""
from __future__ import annotations

import logging

from mlb.config import Hand, LEAGUE_AVERAGES, SEASON
from mlb.data.lineups import build_default_lineup_from_roster, fetch_game_lineup, fetch_team_roster
from mlb.data.models import BatterStats, GameContext, Lineup
from mlb.data.park_factors import get_park_factors, get_venue_for_team
from mlb.data.stats import (
    build_batter_stats,
    build_pitcher_stats,
    ensure_runtime_league_averages,
    fangraphs_team_code,
    fetch_team_bullpen_stats,
)
from mlb.data.weather import get_game_weather
from mlb.utils.normalize import normalize_name as _normalize_name

logger = logging.getLogger(__name__)


def _league_average_batter(name: str, bats: str, pitcher_throws: str | None = None) -> dict:
    bats_text = str(bats).upper()
    hand = Hand.LEFT if bats_text == Hand.LEFT.value else Hand.RIGHT
    opposing_hand = Hand.LEFT if str(pitcher_throws).upper() == Hand.LEFT.value else Hand.RIGHT
    effective_hand = Hand.LEFT if bats_text == Hand.SWITCH.value and opposing_hand == Hand.RIGHT else (
        Hand.RIGHT if bats_text == Hand.SWITCH.value else hand
    )
    return {
        "player_id": f"avg-batter-{_normalize_name(name) or 'unknown'}",
        "name": name or "League Average Batter",
        "bats": hand.value,
        "pa": 600,
        "rates": dict(LEAGUE_AVERAGES[(effective_hand, opposing_hand)]),
        "source": "league_avg",
    }


def _league_average_pitcher(name: str, throws: str, batter_hand: str | None = None) -> dict:
    hand = Hand.LEFT if str(throws).upper() == Hand.LEFT.value else Hand.RIGHT
    effective_batter_hand = Hand.LEFT if str(batter_hand).upper() == Hand.LEFT.value else Hand.RIGHT
    matchup_rates = LEAGUE_AVERAGES[(effective_batter_hand, hand)]
    return {
        "player_id": f"avg-pitcher-{_normalize_name(name) or 'unknown'}",
        "name": name or "League Average Pitcher",
        "throws": hand.value,
        "pa_against": 700,
        "rates": dict(matchup_rates),
        "overall": {
            "pa_against": 700,
            "rates": dict(matchup_rates),
            "source": "league_avg",
        },
        "splits": {
            "vs_lhb": {
                "pa_against": 350,
                "rates": dict(LEAGUE_AVERAGES[(Hand.LEFT, hand)]),
                "source": "league_avg",
            },
            "vs_rhb": {
                "pa_against": 350,
                "rates": dict(LEAGUE_AVERAGES[(Hand.RIGHT, hand)]),
                "source": "league_avg",
            },
        },
        "avg_pitch_count": 85.0,
        "source": "league_avg",
    }


def _build_batting_order(
    players: list[dict],
    batting_data: dict[str, dict],
    opposing_pitcher: dict | None = None,
) -> list[BatterStats]:
    batting_order: list[BatterStats] = []
    opposing_throws = (opposing_pitcher or {}).get("throws")
    for player in players[:9]:
        name = player.get("name", "")
        normalized = _normalize_name(name)
        stats = batting_data.get(normalized)
        if stats is None:
            logger.warning(
                'No stats found for "%s" (normalized: "%s") — using league average', name, normalized
            )
            stats = _league_average_batter(name, player.get("bats", "R"), opposing_throws)

        batter = build_batter_stats(
            stats,
            player.get("bats", stats.get("bats", "R")),
            pitcher_hand=opposing_throws,
        )
        batter.player_id = str(player.get("id") or batter.player_id)
        batter.name = str(player.get("name") or batter.name)
        batting_order.append(batter)

    if len(batting_order) != 9:
        raise ValueError(f"Expected 9 batters, found {len(batting_order)}")
    return batting_order


def _build_pitcher(player: dict | None, pitching_data: dict[str, dict]):
    if not player:
        logger.warning("Missing starting pitcher info; using league-average fallback")
        pitcher = build_pitcher_stats(_league_average_pitcher("League Average Pitcher", "R"))
        return pitcher

    name = player.get("name", "")
    normalized = _normalize_name(name)
    stats = pitching_data.get(normalized)
    if stats is None:
        logger.warning(
            'No stats found for "%s" (normalized: "%s") — using league average', name, normalized
        )
        stats = _league_average_pitcher(name, player.get("throws", "R"))
    else:
        stats = dict(stats, throws=player.get("throws", stats.get("throws", "R")))

    pitcher = build_pitcher_stats(stats)
    pitcher.player_id = str(player.get("id") or pitcher.player_id)
    pitcher.name = str(player.get("name") or pitcher.name)
    return pitcher


def _build_bullpen(team_name: str) -> list:
    team_code = fangraphs_team_code(team_name)
    bullpen_data = fetch_team_bullpen_stats(season=SEASON).get(team_code)
    if bullpen_data is None:
        logger.warning("Missing bullpen data for %s; using league-average fallback", team_name)
        bullpen_data = _league_average_pitcher(f"{team_name} Bullpen", "R")
    else:
        bullpen_data = dict(bullpen_data, name=f"{team_name} Bullpen")

    bullpen = build_pitcher_stats(bullpen_data)
    bullpen.player_id = str(bullpen_data.get("player_id") or f"{_normalize_name(team_name)}-bullpen")
    bullpen.name = f"{team_name} Bullpen"
    return [bullpen]


def _resolve_lineup(game_info: dict) -> dict:
    lineup = fetch_game_lineup(int(game_info["game_id"]))

    # Probable pitchers from the schedule API always take priority over the boxscore.
    # The boxscore may list warm-up arms or last season's pitchers for pre-game entries.
    away_probable = str(game_info.get("away_probable_pitcher") or "").strip()
    home_probable = str(game_info.get("home_probable_pitcher") or "").strip()

    if lineup is not None:
        away_starter_source = "boxscore"
        home_starter_source = "boxscore"
        if away_probable:
            throws = (lineup.get("away_pitcher") or {}).get("throws", "R")
            lineup = dict(lineup, away_pitcher={"name": away_probable, "id": "", "throws": throws})
            away_starter_source = "probable"
            logger.info("Using probable starter %r for %s", away_probable, game_info.get("away_team", "away"))
        if home_probable:
            throws = (lineup.get("home_pitcher") or {}).get("throws", "R")
            lineup = dict(lineup, home_pitcher={"name": home_probable, "id": "", "throws": throws})
            home_starter_source = "probable"
            logger.info("Using probable starter %r for %s", home_probable, game_info.get("home_team", "home"))

        # Enrich batter handedness from roster — boxscore_data does not carry batSide.
        away_bats = {p["id"]: p["bats"] for p in fetch_team_roster(int(game_info["away_team_id"]))}
        home_bats = {p["id"]: p["bats"] for p in fetch_team_roster(int(game_info["home_team_id"]))}
        for b in lineup["away_batters"]:
            b["bats"] = away_bats.get(b["id"], b.get("bats", "R"))
        for b in lineup["home_batters"]:
            b["bats"] = home_bats.get(b["id"], b.get("bats", "R"))

        return dict(
            lineup,
            away_lineup_source="confirmed",
            home_lineup_source="confirmed",
            away_starter_source=away_starter_source,
            home_starter_source=home_starter_source,
        )

    away_batters, roster_away_pitcher = build_default_lineup_from_roster(
        int(game_info["away_team_id"]),
        season=SEASON,
    )
    home_batters, roster_home_pitcher = build_default_lineup_from_roster(
        int(game_info["home_team_id"]),
        season=SEASON,
    )

    away_pitcher = (
        {"name": away_probable, "id": "", "throws": roster_away_pitcher.get("throws", "R")}
        if away_probable
        else roster_away_pitcher
    )
    home_pitcher = (
        {"name": home_probable, "id": "", "throws": roster_home_pitcher.get("throws", "R")}
        if home_probable
        else roster_home_pitcher
    )
    away_starter_source = "probable" if away_probable else "first_roster_arm"
    home_starter_source = "probable" if home_probable else "first_roster_arm"

    if away_probable:
        logger.info("Using probable starter %r for %s", away_probable, game_info.get("away_team", "away"))
    else:
        logger.warning(
            "No probable pitcher for %s; using first roster arm %r",
            game_info.get("away_team", "away"),
            roster_away_pitcher.get("name"),
        )
    if home_probable:
        logger.info("Using probable starter %r for %s", home_probable, game_info.get("home_team", "home"))
    else:
        logger.warning(
            "No probable pitcher for %s; using first roster arm %r",
            game_info.get("home_team", "home"),
            roster_home_pitcher.get("name"),
        )

    return {
        "away_batters": away_batters,
        "home_batters": home_batters,
        "away_pitcher": away_pitcher,
        "home_pitcher": home_pitcher,
        "away_lineup_source": "fallback_roster_order",
        "home_lineup_source": "fallback_roster_order",
        "away_starter_source": away_starter_source,
        "home_starter_source": home_starter_source,
    }


def build_game_context(
    game_info: dict,
    batting_data: dict[str, dict],
    pitching_data: dict[str, dict],
) -> GameContext:
    """Build a complete simulation-ready GameContext from fetched data."""
    ensure_runtime_league_averages(season=SEASON)
    lineup_info = _resolve_lineup(game_info)
    venue_name = game_info.get("venue") or get_venue_for_team(game_info.get("home_team", "")) or "Unknown Venue"

    away_lineup = Lineup(
        team_id=str(game_info.get("away_team_id") or ""),
        team_name=str(game_info.get("away_team") or ""),
        batting_order=_build_batting_order(
            lineup_info["away_batters"],
            batting_data,
            lineup_info.get("home_pitcher"),
        ),
        starting_pitcher=_build_pitcher(lineup_info.get("away_pitcher"), pitching_data),
        bullpen=_build_bullpen(str(game_info.get("away_team") or "Away Team")),
    )
    home_lineup = Lineup(
        team_id=str(game_info.get("home_team_id") or ""),
        team_name=str(game_info.get("home_team") or ""),
        batting_order=_build_batting_order(
            lineup_info["home_batters"],
            batting_data,
            lineup_info.get("away_pitcher"),
        ),
        starting_pitcher=_build_pitcher(lineup_info.get("home_pitcher"), pitching_data),
        bullpen=_build_bullpen(str(game_info.get("home_team") or "Home Team")),
    )

    return GameContext(
        game_id=str(game_info.get("game_id") or ""),
        date=str(game_info.get("game_datetime") or ""),
        away_lineup=away_lineup,
        home_lineup=home_lineup,
        park_factors=get_park_factors(venue_name),
        weather=get_game_weather(venue_name, str(game_info.get("game_datetime") or "")),
        away_lineup_source=str(lineup_info.get("away_lineup_source") or "confirmed"),
        home_lineup_source=str(lineup_info.get("home_lineup_source") or "confirmed"),
        away_starter_source=str(lineup_info.get("away_starter_source") or "boxscore"),
        home_starter_source=str(lineup_info.get("home_starter_source") or "boxscore"),
    )
