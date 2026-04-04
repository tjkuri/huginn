"""Assembly helpers that turn fetched data into simulation-ready game contexts."""
from __future__ import annotations

import logging

from mlb.config import Hand, LEAGUE_AVERAGES, SEASON
from mlb.data.lineups import build_default_lineup_from_roster, fetch_game_lineup
from mlb.data.models import BatterStats, GameContext, Lineup
from mlb.data.park_factors import get_park_factors, get_venue_for_team
from mlb.data.stats import build_batter_stats, build_pitcher_stats
from mlb.data.weather import get_game_weather

logger = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _league_average_batter(name: str, bats: str) -> dict:
    hand = Hand.LEFT if str(bats).upper() == Hand.LEFT.value else Hand.RIGHT
    return {
        "player_id": f"avg-batter-{_normalize_name(name) or 'unknown'}",
        "name": name or "League Average Batter",
        "bats": hand.value,
        "pa": 600,
        "rates": dict(LEAGUE_AVERAGES[(hand, Hand.RIGHT)]),
        "source": "league_avg",
    }


def _league_average_pitcher(name: str, throws: str) -> dict:
    hand = Hand.LEFT if str(throws).upper() == Hand.LEFT.value else Hand.RIGHT
    matchup_rates = (
        LEAGUE_AVERAGES[(Hand.RIGHT, Hand.LEFT)]
        if hand == Hand.LEFT
        else LEAGUE_AVERAGES[(Hand.LEFT, Hand.RIGHT)]
    )
    return {
        "player_id": f"avg-pitcher-{_normalize_name(name) or 'unknown'}",
        "name": name or "League Average Pitcher",
        "throws": hand.value,
        "pa_against": 700,
        "rates": dict(matchup_rates),
        "avg_pitch_count": 85.0,
        "source": "league_avg",
    }


def _build_batting_order(players: list[dict], batting_data: dict[str, dict]) -> list[BatterStats]:
    batting_order: list[BatterStats] = []
    for player in players[:9]:
        name = player.get("name", "")
        stats = batting_data.get(_normalize_name(name))
        if stats is None:
            logger.warning("Missing batting data for %s; using league-average fallback", name)
            stats = _league_average_batter(name, player.get("bats", "R"))
        elif stats.get("source") == str(SEASON - 1):
            logger.info("Using %s batting stats for %s", SEASON - 1, name)

        batter = build_batter_stats(stats, player.get("bats", stats.get("bats", "R")))
        batter.player_id = str(player.get("id") or batter.player_id)
        batter.name = str(player.get("name") or batter.name)
        batter.data_source = str(stats.get("source") or "unknown")
        batting_order.append(batter)

    if len(batting_order) != 9:
        raise ValueError(f"Expected 9 batters, found {len(batting_order)}")
    return batting_order


def _build_pitcher(player: dict | None, pitching_data: dict[str, dict]):
    if not player:
        logger.warning("Missing starting pitcher info; using league-average fallback")
        pitcher = build_pitcher_stats(_league_average_pitcher("League Average Pitcher", "R"))
        pitcher.data_source = "league_avg"
        return pitcher

    name = player.get("name", "")
    stats = pitching_data.get(_normalize_name(name))
    if stats is None:
        logger.warning("Missing pitching data for %s; using league-average fallback", name)
        stats = _league_average_pitcher(name, player.get("throws", "R"))
    elif stats.get("source") == str(SEASON - 1):
        logger.info("Using %s pitching stats for %s", SEASON - 1, name)

    pitcher = build_pitcher_stats(stats)
    pitcher.player_id = str(player.get("id") or pitcher.player_id)
    pitcher.name = str(player.get("name") or pitcher.name)
    pitcher.data_source = str(stats.get("source") or "unknown")
    return pitcher


def _build_bullpen(team_name: str, throws: str) -> list:
    hand = Hand.LEFT if str(throws).upper() == Hand.LEFT.value else Hand.RIGHT
    bullpen = []
    for idx in range(4):
        reliever = build_pitcher_stats(
            {
                "player_id": f"{_normalize_name(team_name)}-rp-{idx + 1}",
                "name": f"{team_name} RP{idx + 1}",
                "throws": hand.value if idx % 2 == 0 else Hand.RIGHT.value,
                "pa_against": 240,
                "rates": dict(
                    LEAGUE_AVERAGES[
                        (Hand.RIGHT, Hand.LEFT) if (hand if idx % 2 == 0 else Hand.RIGHT) == Hand.LEFT
                        else (Hand.LEFT, Hand.RIGHT)
                    ]
                ),
                "avg_pitch_count": 28.0,
            }
        )
        bullpen.append(reliever)
    return bullpen


def _resolve_lineup(game_info: dict) -> dict:
    lineup = fetch_game_lineup(int(game_info["game_id"]))
    if lineup is not None:
        return lineup

    away_batters, away_pitcher = build_default_lineup_from_roster(
        int(game_info["away_team_id"]),
        season=SEASON,
    )
    home_batters, home_pitcher = build_default_lineup_from_roster(
        int(game_info["home_team_id"]),
        season=SEASON,
    )
    return {
        "away_batters": away_batters,
        "home_batters": home_batters,
        "away_pitcher": away_pitcher,
        "home_pitcher": home_pitcher,
    }


def build_game_context(
    game_info: dict,
    batting_data: dict[str, dict],
    pitching_data: dict[str, dict],
) -> GameContext:
    """Build a complete simulation-ready GameContext from fetched data."""
    lineup_info = _resolve_lineup(game_info)
    venue_name = game_info.get("venue") or get_venue_for_team(game_info.get("home_team", "")) or "Unknown Venue"

    away_lineup = Lineup(
        team_id=str(game_info.get("away_team_id") or ""),
        team_name=str(game_info.get("away_team") or ""),
        batting_order=_build_batting_order(lineup_info["away_batters"], batting_data),
        starting_pitcher=_build_pitcher(lineup_info.get("away_pitcher"), pitching_data),
        bullpen=_build_bullpen(str(game_info.get("away_team") or "Away Team"), lineup_info.get("away_pitcher", {}).get("throws", "R")),
    )
    home_lineup = Lineup(
        team_id=str(game_info.get("home_team_id") or ""),
        team_name=str(game_info.get("home_team") or ""),
        batting_order=_build_batting_order(lineup_info["home_batters"], batting_data),
        starting_pitcher=_build_pitcher(lineup_info.get("home_pitcher"), pitching_data),
        bullpen=_build_bullpen(str(game_info.get("home_team") or "Home Team"), lineup_info.get("home_pitcher", {}).get("throws", "R")),
    )

    return GameContext(
        game_id=str(game_info.get("game_id") or ""),
        date=str(game_info.get("game_datetime") or ""),
        away_lineup=away_lineup,
        home_lineup=home_lineup,
        park_factors=get_park_factors(venue_name),
        weather=get_game_weather(venue_name, str(game_info.get("game_datetime") or "")),
    )
