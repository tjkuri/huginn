"""Schedule, roster, and confirmed-lineup access via MLB Stats API."""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from mlb.config import DAILY_CACHE_TTL_DAYS, GAME_CACHE_TTL_DAYS, SEASON, SEASONAL_CACHE_TTL_DAYS
from mlb.data.cache import read_cache, write_cache

logger = logging.getLogger(__name__)


def _today_str() -> str:
    return date.today().isoformat()


def _normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _import_statsapi():
    try:
        import statsapi
    except ImportError as exc:
        raise ImportError(
            "MLB-StatsAPI is required for lineup and schedule fetching. Install it with "
            "`venv/bin/pip install MLB-StatsAPI`."
        ) from exc
    return statsapi


def _extract_team(block: dict[str, Any], side: str) -> tuple[str, str]:
    if f"{side}_id" in block or f"{side}_name" in block:
        return (
            str(block.get(f"{side}_id") or ""),
            str(block.get(f"{side}_name") or ""),
        )
    side_block = block.get(side, {})
    team = side_block.get("team", {})
    return str(team.get("id") or side_block.get("team_id") or ""), str(team.get("name") or side_block.get("name") or "")


def _schedule_cache_is_valid(games: list[dict]) -> bool:
    if not games:
        return True
    return all(game.get("away_team") and game.get("home_team") for game in games)


def fetch_todays_games(date: str | None = None) -> list[dict]:
    """Fetch the MLB schedule for one date."""
    schedule_date = date or _today_str()
    cached = read_cache("schedule", "games", date=schedule_date, max_age_days=DAILY_CACHE_TTL_DAYS)
    if cached:
        cached_games = list(cached.get("games", []))
        if _schedule_cache_is_valid(cached_games):
            return cached_games
        logger.warning("Discarding malformed cached schedule for %s", schedule_date)

    statsapi = _import_statsapi()
    raw_games = statsapi.schedule(date=schedule_date)
    games: list[dict] = []
    for game in raw_games:
        home_id, home_name = _extract_team(game, "home")
        away_id, away_name = _extract_team(game, "away")
        games.append(
            {
                "game_id": str(game.get("game_id") or game.get("gamePk") or ""),
                "away_team": away_name,
                "away_team_id": away_id,
                "home_team": home_name,
                "home_team_id": home_id,
                "game_datetime": game.get("game_datetime") or game.get("gameDate") or "",
                "venue": game.get("venue_name") or game.get("venue") or "",
                "status": game.get("status") or game.get("detailed_state") or "",
            }
        )

    write_cache("schedule", "games", {"games": games}, date=schedule_date)
    return games


def _extract_batters(team_block: dict[str, Any]) -> list[dict]:
    players = team_block.get("players", {}) or {}
    batters: list[dict] = []
    for player in players.values():
        batting_order = player.get("battingOrder")
        if not batting_order:
            continue
        position = (player.get("position") or {}).get("abbreviation")
        if position == "P":
            continue
        person = player.get("person", {})
        batters.append(
            {
                "name": person.get("fullName") or player.get("name") or "",
                "id": str(person.get("id") or player.get("id") or ""),
                "batting_position": int(str(batting_order)[:2]),
                "bats": ((player.get("batSide") or {}).get("code") or "R").upper(),
                "position": position or "",
            }
        )
    batters.sort(key=lambda item: item["batting_position"])
    return batters[:9]


def _extract_pitcher(team_block: dict[str, Any]) -> dict | None:
    for player in (team_block.get("players", {}) or {}).values():
        position = (player.get("position") or {}).get("abbreviation")
        if position != "P":
            continue
        if not player.get("battingOrder") and not player.get("stats"):
            continue
        person = player.get("person", {})
        return {
            "name": person.get("fullName") or player.get("name") or "",
            "id": str(person.get("id") or player.get("id") or ""),
            "throws": ((player.get("pitchHand") or {}).get("code") or "R").upper(),
        }

    pitchers = team_block.get("pitchers") or []
    if pitchers:
        first_pitcher_id = str(pitchers[0])
        players = team_block.get("players", {}) or {}
        player = players.get(f"ID{first_pitcher_id}") or {}
        person = player.get("person", {})
        return {
            "name": person.get("fullName") or player.get("name") or "",
            "id": first_pitcher_id,
            "throws": ((player.get("pitchHand") or {}).get("code") or "R").upper(),
        }

    return None


def fetch_game_lineup(game_id: int) -> dict | None:
    """Fetch confirmed lineups for a game if they are available."""
    cache_key = str(game_id)
    cached = read_cache("lineups", cache_key, max_age_days=GAME_CACHE_TTL_DAYS)
    if cached:
        return cached.get("lineup")

    statsapi = _import_statsapi()
    boxscore = statsapi.boxscore_data(game_id)
    away = boxscore.get("away", {})
    home = boxscore.get("home", {})
    lineup = {
        "away_batters": _extract_batters(away),
        "home_batters": _extract_batters(home),
        "away_pitcher": _extract_pitcher(away),
        "home_pitcher": _extract_pitcher(home),
    }

    if (
        len(lineup["away_batters"]) < 9
        or len(lineup["home_batters"]) < 9
        or lineup["away_pitcher"] is None
        or lineup["home_pitcher"] is None
    ):
        logger.info("Confirmed lineups not available for game %s", game_id)
        return None

    write_cache("lineups", cache_key, {"lineup": lineup})
    return lineup


def fetch_team_roster(team_id: int, season: int = SEASON) -> list[dict]:
    """Fetch and cache a team roster for one season."""
    cache_key = f"{season}-{team_id}"
    cached = read_cache("rosters", cache_key, max_age_days=SEASONAL_CACHE_TTL_DAYS)
    if cached:
        return list(cached.get("roster", []))

    statsapi = _import_statsapi()
    response = statsapi.get("team_roster", {"teamId": team_id, "season": season, "rosterType": "active"})
    roster_entries = response.get("roster", [])
    roster: list[dict] = []
    for entry in roster_entries:
        person = entry.get("person", {})
        position = entry.get("position", {})
        roster.append(
            {
                "id": str(person.get("id") or ""),
                "name": person.get("fullName") or "",
                "position": position.get("abbreviation") or "",
                "status": (entry.get("status") or {}).get("description") or "",
                "bats": ((entry.get("batSide") or {}).get("code") or "R").upper(),
                "throws": ((entry.get("pitchHand") or {}).get("code") or "R").upper(),
            }
        )

    write_cache("rosters", cache_key, {"roster": roster})
    return roster


def build_default_lineup_from_roster(team_id: int, season: int = SEASON) -> tuple[list[dict], dict]:
    """Create a simple fallback lineup from roster order when no lineup is confirmed."""
    roster = fetch_team_roster(team_id, season=season)
    batters = [player for player in roster if player.get("position") != "P"][:9]
    pitcher = next((player for player in roster if player.get("position") == "P"), None)
    if len(batters) < 9 or pitcher is None:
        raise ValueError(f"Insufficient roster data to build fallback lineup for team {team_id}")

    batting_order = [
        {
            "name": player["name"],
            "id": player["id"],
            "batting_position": index,
            "bats": player.get("bats", "R"),
            "position": player.get("position", ""),
        }
        for index, player in enumerate(batters, start=1)
    ]
    starting_pitcher = {
        "name": pitcher["name"],
        "id": pitcher["id"],
        "throws": pitcher.get("throws", "R"),
    }
    return batting_order, starting_pitcher
