"""Schedule, roster, and confirmed-lineup access via MLB Stats API."""
from __future__ import annotations

import logging
import re
import unicodedata
from datetime import date
from typing import Any

from mlb.config import SEASON

logger = logging.getLogger(__name__)


def _today_str() -> str:
    return date.today().isoformat()


_NAME_SUFFIX_RE = re.compile(r'\s+(jr|sr|ii|iii|iv)$')


def _normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize('NFD', str(name))
    stripped = ''.join(c for c in nfkd if not unicodedata.category(c).startswith('M'))
    stripped = stripped.replace('.', '')
    stripped = ' '.join(stripped.strip().lower().split())
    stripped = _NAME_SUFFIX_RE.sub('', stripped)
    return stripped


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


def fetch_todays_games(date: str | None = None) -> list[dict]:
    """Fetch the MLB schedule for one date. Always fetches fresh from the API."""
    schedule_date = date or _today_str()
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
                "away_probable_pitcher": str(game.get("away_probable_pitcher") or ""),
                "home_probable_pitcher": str(game.get("home_probable_pitcher") or ""),
            }
        )
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
    """Fetch confirmed lineups for a game if they are available. Always fetches fresh."""
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

    return lineup


def fetch_team_roster(team_id: int, season: int = SEASON) -> list[dict]:
    """Fetch a team roster for one season. Always fetches fresh."""
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
