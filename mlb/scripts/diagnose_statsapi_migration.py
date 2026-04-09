"""Spike script for validating MLB Stats API as a replacement for FanGraphs stat loads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests

from mlb.config import CACHE_DIR
from mlb.data.stats import _build_batter_player, _build_pitcher_player, _fetch_people_handedness
from mlb.data.team_codes import TEAM_TO_CODE
from mlb.utils.normalize import normalize_name

_STATS_API_URL = "https://statsapi.mlb.com/api/v1/stats"
_DEFAULT_TIMEOUT = 30
_BATCH_SIZE = 10000
_RATE_KEYS = ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--samples", type=int, default=5)
    return parser.parse_args()


def _fetch_stats_api_rows(*, season: int, group: str, stats_type: str, sit_code: str | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "stats": stats_type,
        "group": group,
        "playerPool": "ALL",
        "season": season,
        "sportIds": 1,
        "limit": _BATCH_SIZE,
    }
    if sit_code:
        params["sitCodes"] = sit_code
    response = requests.get(_STATS_API_URL, params=params, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    return payload.get("stats", [{}])[0].get("splits") or []


def _team_code(row: dict[str, Any]) -> str:
    team = row.get("team") or {}
    name = str(team.get("name") or "").strip()
    return TEAM_TO_CODE.get(name, name)


def _coerce_hbp(stat: dict[str, Any]) -> Any:
    if "hitByPitch" in stat:
        return stat.get("hitByPitch")
    return stat.get("hitBatsmen", 0)


def _people_handedness(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    person_ids = sorted(
        {
            int(str((row.get("player") or {}).get("id") or "").strip())
            for row in rows
            if str((row.get("player") or {}).get("id") or "").strip().isdigit()
        }
    )
    return _fetch_people_handedness(person_ids)


def _normalize_batting_rows(rows: list[dict[str, Any]], season: int, *, split_type: str) -> dict[str, dict[str, Any]]:
    handedness = _people_handedness(rows)
    players: dict[str, dict[str, Any]] = {}
    for row in rows:
        player = row.get("player") or {}
        stat = row.get("stat") or {}
        mlb_row = {
            "ID": str(player.get("id") or ""),
            "Name": str(player.get("fullName") or "").strip(),
            "Team": _team_code(row),
            "Bats": str((handedness.get(str(player.get("id") or "")) or {}).get("bats") or "R").upper(),
            "PA": stat.get("plateAppearances", 0),
            "H": stat.get("hits", 0),
            "2B": stat.get("doubles", 0),
            "3B": stat.get("triples", 0),
            "HR": stat.get("homeRuns", 0),
            "HBP": _coerce_hbp(stat),
            "SO": stat.get("strikeOuts", 0),
            "BB": stat.get("baseOnBalls", 0),
        }
        source = f"{season}_{'split' if split_type != 'overall' else 'overall'}"
        normalized = _build_batter_player(mlb_row, season, source=source, split_type=split_type)
        if normalized is None:
            continue
        normalized["notes"] = f"MLB Stats API batting stats ({split_type})."
        players[normalize_name(normalized["name"])] = normalized
    return players


def _normalize_pitching_rows(rows: list[dict[str, Any]], season: int, *, split_type: str) -> dict[str, dict[str, Any]]:
    handedness = _people_handedness(rows)
    players: dict[str, dict[str, Any]] = {}
    for row in rows:
        player = row.get("player") or {}
        stat = row.get("stat") or {}
        mlb_row = {
            "ID": str(player.get("id") or ""),
            "Name": str(player.get("fullName") or "").strip(),
            "Team": _team_code(row),
            "Throws": str((handedness.get(str(player.get("id") or "")) or {}).get("throws") or "R").upper(),
            "IP": stat.get("inningsPitched", 0),
            "BF": stat.get("battersFaced", 0),
            "H": stat.get("hits", 0),
            "2B": stat.get("doubles", 0),
            "3B": stat.get("triples", 0),
            "HR": stat.get("homeRuns", 0),
            "BB": stat.get("baseOnBalls", 0),
            "HBP": _coerce_hbp(stat),
            "SO": stat.get("strikeOuts", 0),
        }
        source = f"{season}_{'split' if split_type != 'overall' else 'overall'}"
        normalized = _build_pitcher_player(mlb_row, season, source=source, split_type=split_type)
        if normalized is None:
            continue
        normalized["notes"] = f"MLB Stats API pitching stats ({split_type})."
        players[normalize_name(normalized["name"])] = normalized
    return players


def _load_cached_players(kind: str, season: int) -> dict[str, dict[str, Any]]:
    path = Path(CACHE_DIR) / f"raw_{kind}-{season}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("players") or {}


def _sample_overlap(
    *,
    label: str,
    api_players: dict[str, dict[str, Any]],
    cached_players: dict[str, dict[str, Any]],
    sample_size: int,
    volume_key: str,
) -> None:
    overlap = sorted(set(api_players) & set(cached_players), key=lambda name: api_players[name].get(volume_key, 0), reverse=True)
    print(f"  overlap with cache: {len(overlap)} players")
    for name in overlap[:sample_size]:
        api_player = api_players[name]
        cached_player = cached_players[name]
        print(f"  sample {api_player['name']}:")
        for key in _RATE_KEYS:
            api_rate = api_player["rates"][key]
            cached_rate = cached_player["rates"][key]
            delta = api_rate - cached_rate
            print(f"    {key}: api={api_rate:.4f} cache={cached_rate:.4f} delta={delta:+.4f}")


def _report_dataset(
    *,
    label: str,
    api_players: dict[str, dict[str, Any]],
    cached_players: dict[str, dict[str, Any]],
    sample_size: int,
    volume_key: str,
) -> None:
    print(f"\n[{label}]")
    print(f"  statsapi normalized players: {len(api_players)}")
    print(f"  cached players:             {len(cached_players)}")
    print(f"  only in statsapi:           {len(set(api_players) - set(cached_players))}")
    print(f"  only in cache:              {len(set(cached_players) - set(api_players))}")
    _sample_overlap(
        label=label,
        api_players=api_players,
        cached_players=cached_players,
        sample_size=sample_size,
        volume_key=volume_key,
    )


def main() -> None:
    args = _parse_args()
    season = args.season

    batting_overall_rows = _fetch_stats_api_rows(season=season, group="hitting", stats_type="season")
    pitching_overall_rows = _fetch_stats_api_rows(season=season, group="pitching", stats_type="season")
    batting_vs_lhp_rows = _fetch_stats_api_rows(season=season, group="hitting", stats_type="statSplits", sit_code="vl")
    batting_vs_rhp_rows = _fetch_stats_api_rows(season=season, group="hitting", stats_type="statSplits", sit_code="vr")
    pitching_vs_lhb_rows = _fetch_stats_api_rows(season=season, group="pitching", stats_type="statSplits", sit_code="vl")
    pitching_vs_rhb_rows = _fetch_stats_api_rows(season=season, group="pitching", stats_type="statSplits", sit_code="vr")

    batting_overall = _normalize_batting_rows(batting_overall_rows, season, split_type="overall")
    pitching_overall = _normalize_pitching_rows(pitching_overall_rows, season, split_type="overall")
    batting_vs_lhp = _normalize_batting_rows(batting_vs_lhp_rows, season, split_type="vs_lhp")
    batting_vs_rhp = _normalize_batting_rows(batting_vs_rhp_rows, season, split_type="vs_rhp")
    pitching_vs_lhb = _normalize_pitching_rows(pitching_vs_lhb_rows, season, split_type="vs_lhb")
    pitching_vs_rhb = _normalize_pitching_rows(pitching_vs_rhb_rows, season, split_type="vs_rhb")

    print(f"MLB Stats API spike for season {season}")
    _report_dataset(
        label="batting overall",
        api_players=batting_overall,
        cached_players=_load_cached_players("batting", season),
        sample_size=args.samples,
        volume_key="pa",
    )
    _report_dataset(
        label="batting vs lhp",
        api_players=batting_vs_lhp,
        cached_players=_load_cached_players("batting_vs_lhp", season),
        sample_size=args.samples,
        volume_key="pa",
    )
    _report_dataset(
        label="batting vs rhp",
        api_players=batting_vs_rhp,
        cached_players=_load_cached_players("batting_vs_rhp", season),
        sample_size=args.samples,
        volume_key="pa",
    )
    _report_dataset(
        label="pitching overall",
        api_players=pitching_overall,
        cached_players=_load_cached_players("pitching", season),
        sample_size=args.samples,
        volume_key="pa_against",
    )
    _report_dataset(
        label="pitching vs lhb",
        api_players=pitching_vs_lhb,
        cached_players=_load_cached_players("pitching_vs_lhb", season),
        sample_size=args.samples,
        volume_key="pa_against",
    )
    _report_dataset(
        label="pitching vs rhb",
        api_players=pitching_vs_rhb,
        cached_players=_load_cached_players("pitching_vs_rhb", season),
        sample_size=args.samples,
        volume_key="pa_against",
    )


if __name__ == "__main__":
    main()
