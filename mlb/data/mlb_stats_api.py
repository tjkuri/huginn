"""Repo-local client for MLB Stats API leaderboard-style stat loads."""
from __future__ import annotations

from typing import Any, Callable

import requests

from mlb.data.team_codes import TEAM_TO_CODE

_STATS_API_URL = "https://statsapi.mlb.com/api/v1/stats"
_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
_REQUEST_TIMEOUT = 30
_PAGE_SIZE = 10000


def _fetch_all_splits(*, season: int, group: str, stats_type: str, sit_code: str | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "stats": stats_type,
        "group": group,
        "playerPool": "ALL",
        "season": season,
        "sportIds": 1,
        "limit": _PAGE_SIZE,
        "offset": 0,
    }
    if sit_code:
        params["sitCodes"] = sit_code

    all_splits: list[dict[str, Any]] = []
    while True:
        response = requests.get(_STATS_API_URL, params=params, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        stat_block = (response.json().get("stats") or [{}])[0]
        splits = stat_block.get("splits") or []
        total = int(stat_block.get("totalSplits") or len(splits))
        all_splits.extend(splits)
        params["offset"] += params["limit"]
        if len(all_splits) >= total or not splits:
            break
    return all_splits


def _chunked(values: list[int], size: int) -> list[list[int]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def _fetch_people_handedness(person_ids: list[int]) -> dict[str, dict[str, str]]:
    handedness: dict[str, dict[str, str]] = {}
    for batch in _chunked(person_ids, 50):
        response = requests.get(
            _PEOPLE_URL,
            params={"personIds": ",".join(str(person_id) for person_id in batch)},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        for person in response.json().get("people") or []:
            mlbam_id = str(person.get("id") or "").strip()
            if not mlbam_id:
                continue
            handedness[mlbam_id] = {
                "bats": str(((person.get("batSide") or {}).get("code")) or "").strip().upper(),
                "throws": str(((person.get("pitchHand") or {}).get("code")) or "").strip().upper(),
            }
    return handedness


def _extract_team_code(row: dict[str, Any]) -> str:
    team = row.get("team") or {}
    name = str(team.get("name") or "").strip()
    return TEAM_TO_CODE.get(name, name)


def _coerce_hbp(stat: dict[str, Any]) -> Any:
    if "hitByPitch" in stat:
        return stat.get("hitByPitch")
    return stat.get("hitBatsmen", 0)


def parse_baseball_innings(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    if "." not in text:
        return float(text)
    whole, frac = text.split(".", 1)
    if frac in {"0", "1", "2"}:
        return float(whole) + (int(frac) / 3.0)
    return float(text)


def _fetch_and_normalize_rows(
    *,
    season: int,
    group: str,
    stats_type: str,
    normalize_row: Callable[[dict[str, Any], dict[str, dict[str, str]]], dict[str, Any]],
    sit_code: str | None = None,
) -> list[dict[str, Any]]:
    rows = _fetch_all_splits(season=season, group=group, stats_type=stats_type, sit_code=sit_code)
    person_ids = sorted(
        {
            int(str((row.get("player") or {}).get("id") or "").strip())
            for row in rows
            if str((row.get("player") or {}).get("id") or "").strip().isdigit()
        }
    )
    handedness = _fetch_people_handedness(person_ids)
    return [normalize_row(row, handedness) for row in rows]


def _normalize_batting_row(row: dict[str, Any], handedness: dict[str, dict[str, str]]) -> dict[str, Any]:
    player = row.get("player") or {}
    stat = row.get("stat") or {}
    player_id = str(player.get("id") or "").strip()
    return {
        "ID": player_id,
        "Name": str(player.get("fullName") or "").strip(),
        "Team": _extract_team_code(row),
        "Bats": str((handedness.get(player_id) or {}).get("bats") or "R").upper(),
        "PA": stat.get("plateAppearances", 0),
        "H": stat.get("hits", 0),
        "2B": stat.get("doubles", 0),
        "3B": stat.get("triples", 0),
        "HR": stat.get("homeRuns", 0),
        "HBP": _coerce_hbp(stat),
        "SO": stat.get("strikeOuts", 0),
        "BB": stat.get("baseOnBalls", 0),
    }


def _normalize_pitching_row(row: dict[str, Any], handedness: dict[str, dict[str, str]]) -> dict[str, Any]:
    player = row.get("player") or {}
    stat = row.get("stat") or {}
    player_id = str(player.get("id") or "").strip()
    return {
        "ID": player_id,
        "Name": str(player.get("fullName") or "").strip(),
        "Team": _extract_team_code(row),
        "Throws": str((handedness.get(player_id) or {}).get("throws") or "R").upper(),
        "IP": parse_baseball_innings(stat.get("inningsPitched", 0)),
        "BF": stat.get("battersFaced", 0),
        "G": stat.get("gamesPlayed", 0),
        "GS": stat.get("gamesStarted", 0),
        "H": stat.get("hits", 0),
        "2B": stat.get("doubles", 0),
        "3B": stat.get("triples", 0),
        "HR": stat.get("homeRuns", 0),
        "BB": stat.get("baseOnBalls", 0),
        "HBP": _coerce_hbp(stat),
        "SO": stat.get("strikeOuts", 0),
    }


def fetch_batting_season_rows(season: int) -> list[dict[str, Any]]:
    return _fetch_and_normalize_rows(
        season=season,
        group="hitting",
        stats_type="season",
        normalize_row=_normalize_batting_row,
    )


def fetch_batting_split_rows(season: int, *, sit_code: str) -> list[dict[str, Any]]:
    return _fetch_and_normalize_rows(
        season=season,
        group="hitting",
        stats_type="statSplits",
        sit_code=sit_code,
        normalize_row=_normalize_batting_row,
    )


def fetch_pitching_season_rows(season: int) -> list[dict[str, Any]]:
    return _fetch_and_normalize_rows(
        season=season,
        group="pitching",
        stats_type="season",
        normalize_row=_normalize_pitching_row,
    )


def fetch_pitching_split_rows(season: int, *, sit_code: str) -> list[dict[str, Any]]:
    return _fetch_and_normalize_rows(
        season=season,
        group="pitching",
        stats_type="statSplits",
        sit_code=sit_code,
        normalize_row=_normalize_pitching_row,
    )
