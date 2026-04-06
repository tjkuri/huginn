"""Player statistics fetch/build helpers for MLB simulation inputs."""
from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

from mlb.config import (
    CACHE_DIR,
    Hand,
    LEAGUE_AVERAGES,
    SEASON,
    STATS_CACHE_MAX_AGE_HOURS,
)
from mlb.data.models import BatterStats, PitcherStats

logger = logging.getLogger(__name__)

_MIN_BATTER_PA = 50
_MIN_PITCHER_IP = 20.0
_MIN_SPLIT_BATTER_PA = 30
_MIN_SPLIT_PITCHER_BF = 20
_AVG_PITCHES_PER_INNING = 16.0
_EARLY_SEASON_BATTER_PA = 20
_EARLY_SEASON_PITCHER_IP = 10.0
_SPLIT_MONTH_VS_LEFT = 13
_SPLIT_MONTH_VS_RIGHT = 14

_NAME_SUFFIX_RE = re.compile(r'\s+(jr|sr|ii|iii|iv)$')
_TEAM_BULLPEN_CACHE: dict[tuple[int, bool], dict[str, dict[str, Any]]] = {}
_COMPUTED_LEAGUE_AVERAGE_CACHE: dict[tuple[int, bool], dict[str, float]] = {}
_TEAM_TO_FG_CODE = {
    "Arizona Diamondbacks": "ARI",
    "Athletics": "ATH",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}


def _normalize_name(name: str) -> str:
    # Strip diacritical marks via NFD decomposition
    nfkd = unicodedata.normalize('NFD', str(name))
    stripped = ''.join(c for c in nfkd if not unicodedata.category(c).startswith('M'))
    # Remove periods (turns "J.C." into "JC")
    stripped = stripped.replace('.', '')
    # Lowercase and collapse whitespace
    stripped = ' '.join(stripped.strip().lower().split())
    # Strip trailing name suffixes
    stripped = _NAME_SUFFIX_RE.sub('', stripped)
    return stripped


# ── Raw season stat cache (flat files, no subdirectories) ────────────────────

def _raw_cache_path(kind: str, season: int) -> Path:
    return CACHE_DIR / f"raw_{kind}-{season}.json"


def _load_raw_cache(kind: str, season: int) -> dict[str, dict] | None:
    path = _raw_cache_path(kind, season)
    if not path.exists():
        return None
    # Prior seasons never change; cache them forever.
    # Current season: re-fetch if older than STATS_CACHE_MAX_AGE_HOURS.
    if season >= SEASON:
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > STATS_CACHE_MAX_AGE_HOURS:
            logger.debug("Raw %s cache for %d is %.1fh old, re-fetching", kind, season, age_hours)
            return None
    with open(path) as f:
        return json.load(f).get("players")


def _save_raw_cache(kind: str, season: int, players: dict[str, dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_cache_path(kind, season)
    with open(path, "w") as f:
        json.dump({"players": players}, f, indent=2)
    logger.debug("Raw cache written: %s (%d players)", path, len(players))


def _computed_cache_path(kind: str, season: int) -> Path:
    return CACHE_DIR / f"computed_{kind}-{season}.json"


def _load_computed_cache(kind: str, season: int) -> dict[str, Any] | None:
    path = _computed_cache_path(kind, season)
    if not path.exists():
        return None
    if season >= SEASON:
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > STATS_CACHE_MAX_AGE_HOURS:
            logger.debug("Computed %s cache for %d is %.1fh old, re-fetching", kind, season, age_hours)
            return None
    with open(path) as f:
        return json.load(f)


def _save_computed_cache(kind: str, season: int, payload: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _computed_cache_path(kind, season)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.debug("Computed cache written: %s", path)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if value != value:
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    return int(round(_safe_float(value, default)))


def _first_present(row: dict[str, Any], *keys: str, default: Any = 0.0) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _hand_from_value(value: Any, allow_switch: bool = False) -> Hand:
    # Handle Hand enum directly
    if isinstance(value, Hand):
        return value
    text = str(value or "").strip().upper()
    if text == Hand.LEFT.value:
        return Hand.LEFT
    if text == Hand.RIGHT.value:
        return Hand.RIGHT
    if allow_switch and text == Hand.SWITCH.value:
        return Hand.SWITCH
    return Hand.RIGHT


def _season_date_today() -> date:
    return date.today()


def _is_early_season(season: int) -> bool:
    today = _season_date_today()
    return today.year == season and today < date(season, 5, 1)


def _batting_threshold_for_season(season: int) -> int:
    return _EARLY_SEASON_BATTER_PA if _is_early_season(season) else _MIN_BATTER_PA


def _pitching_threshold_for_season(season: int) -> float:
    return _EARLY_SEASON_PITCHER_IP if _is_early_season(season) else _MIN_PITCHER_IP


def _normalize_rates(rates: dict[str, float]) -> dict[str, float]:
    sanitized = {key: max(0.0, float(value)) for key, value in rates.items()}
    non_out_total = sum(value for key, value in sanitized.items() if key != "OUT")
    sanitized["OUT"] = max(0.0, 1.0 - non_out_total)
    total = sum(sanitized.values())
    if total <= 0:
        return {"K": 0.0, "BB": 0.0, "HBP": 0.0, "1B": 0.0, "2B": 0.0, "3B": 0.0, "HR": 0.0, "OUT": 1.0}
    return {key: value / total for key, value in sanitized.items()}


def _league_average_rates_for_batter(hand: Hand) -> dict[str, float]:
    return dict(LEAGUE_AVERAGES[(hand, Hand.RIGHT)])


def _league_average_rates_for_pitcher(throws: Hand) -> dict[str, float]:
    if throws == Hand.LEFT:
        return dict(LEAGUE_AVERAGES[(Hand.RIGHT, Hand.LEFT)])
    return dict(LEAGUE_AVERAGES[(Hand.LEFT, Hand.RIGHT)])


def _import_pybaseball():
    pybaseball_cache_dir = Path(CACHE_DIR) / "pybaseball"
    pybaseball_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYBASEBALL_CACHE", str(pybaseball_cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(pybaseball_cache_dir / "mplconfig"))
    try:
        from pybaseball import batting_stats, cache, pitching_stats
    except ImportError as exc:
        raise ImportError(
            "pybaseball is required for MLB stats fetching. Install it with "
            "`venv/bin/pip install pybaseball`."
        ) from exc

    cache.enable()
    return batting_stats, pitching_stats


def _frame_to_records(frame: Any) -> list[dict[str, Any]]:
    if hasattr(frame, "to_dict"):
        return frame.to_dict(orient="records")
    raise TypeError("Expected pandas DataFrame-like object with to_dict()")


def compute_league_averages(batting_df: Any, pitching_df: Any) -> dict[str, float]:
    """Compute overall league-average batting rates from raw leaderboard frames."""
    del pitching_df
    records = _frame_to_records(batting_df)
    totals = {
        "PA": 0.0,
        "H": 0.0,
        "2B": 0.0,
        "3B": 0.0,
        "HR": 0.0,
        "HBP": 0.0,
        "K": 0.0,
        "BB": 0.0,
    }
    for row in records:
        pa = _safe_float(_first_present(row, "PA"))
        if pa <= 0:
            continue
        totals["PA"] += pa
        totals["H"] += _safe_float(_first_present(row, "H"))
        totals["2B"] += _safe_float(_first_present(row, "2B"))
        totals["3B"] += _safe_float(_first_present(row, "3B"))
        totals["HR"] += _safe_float(_first_present(row, "HR"))
        totals["HBP"] += _safe_float(_first_present(row, "HBP"))
        totals["K"] += _safe_float(_first_present(row, "SO", "K"), default=pa * _safe_float(_first_present(row, "K%", "SO%", "K_pct")))
        totals["BB"] += _safe_float(_first_present(row, "BB"), default=pa * _safe_float(_first_present(row, "BB%", "BB_pct")))

    total_pa = totals["PA"]
    if total_pa <= 0:
        raise ValueError("Unable to compute league averages from empty batting frame")

    singles = max(0.0, totals["H"] - totals["2B"] - totals["3B"] - totals["HR"])
    rates = {
        "K": totals["K"] / total_pa,
        "BB": totals["BB"] / total_pa,
        "HBP": totals["HBP"] / total_pa,
        "1B": singles / total_pa,
        "2B": totals["2B"] / total_pa,
        "3B": totals["3B"] / total_pa,
        "HR": totals["HR"] / total_pa,
    }
    rates["OUT"] = max(0.0, 1.0 - sum(rates.values()))
    ab = max(1.0, total_pa - totals["BB"] - totals["HBP"])
    rates["AVG"] = totals["H"] / ab
    return rates


def _extract_batter_rates(row: dict[str, Any]) -> dict[str, float]:
    pa = _safe_float(_first_present(row, "PA"))
    if pa <= 0:
        raise ValueError("PA must be positive")

    hits = _safe_float(_first_present(row, "H"))
    doubles = _safe_float(_first_present(row, "2B"))
    triples = _safe_float(_first_present(row, "3B"))
    hr = _safe_float(_first_present(row, "HR"))
    singles = _safe_float(_first_present(row, "1B", default=hits - doubles - triples - hr))
    hbp = _safe_float(_first_present(row, "HBP"))
    # pybaseball returns K% and BB% as decimals (e.g. 0.182), not percentages
    k_rate = _safe_float(_first_present(row, "K%", "SO%", "K_pct"))
    bb_rate = _safe_float(_first_present(row, "BB%", "BB_pct"))

    return _normalize_rates(
        {
            "K": k_rate,
            "BB": bb_rate,
            "HBP": hbp / pa,
            "1B": singles / pa,
            "2B": doubles / pa,
            "3B": triples / pa,
            "HR": hr / pa,
            "OUT": 0.0,
        }
    )


def _build_batter_player(
    row: dict[str, Any],
    season: int,
    source: str,
    *,
    split_type: str = "overall",
) -> dict | None:
    pa = _safe_int(_first_present(row, "PA"))
    if pa <= 0:
        return None

    name = str(_first_present(row, "Name", default="")).strip()
    if not name:
        return None

    try:
        rates = _extract_batter_rates(row)
    except ValueError:
        return None

    return {
        "player_id": str(_first_present(row, "IDfg", "ID", default=name)),
        "name": name,
        "team": str(_first_present(row, "Team", default="")).strip(),
        "bats": str(_first_present(row, "Bat", "Bats", default="R")).strip().upper() or "R",
        "pa": pa,
        "rates": rates,
        "season": season,
        "source": source,
        "split_type": split_type,
        "notes": f"FanGraphs batting leaderboard data ({split_type}).",
    }


def _estimate_pitcher_pa(row: dict[str, Any]) -> int:
    tbf = _safe_float(_first_present(row, "TBF", "BF"))
    if tbf > 0:
        return max(1, int(round(tbf)))

    ip = _safe_float(_first_present(row, "IP"))
    hits = _safe_float(_first_present(row, "H"))
    walks = _safe_float(_first_present(row, "BB"))
    hbp = _safe_float(_first_present(row, "HBP"))
    estimated = (ip * 3.0) + hits + walks + hbp
    return max(1, int(round(estimated)))


def _extract_pitcher_rates(row: dict[str, Any]) -> tuple[dict[str, float], int]:
    pa_against = _estimate_pitcher_pa(row)
    hits = _safe_float(_first_present(row, "H"))
    doubles = _safe_float(_first_present(row, "2B", "2B_Allowed"))
    triples = _safe_float(_first_present(row, "3B", "3B_Allowed"))
    hr = _safe_float(_first_present(row, "HR"))
    singles = _safe_float(_first_present(row, "1B", default=hits - doubles - triples - hr))
    walks = _safe_float(_first_present(row, "BB"))
    hbp = _safe_float(_first_present(row, "HBP"))
    strikeouts = _safe_float(_first_present(row, "SO", "K"))

    rates = _normalize_rates(
        {
            "K": strikeouts / pa_against,
            "BB": walks / pa_against,
            "HBP": hbp / pa_against,
            "1B": singles / pa_against,
            "2B": doubles / pa_against,
            "3B": triples / pa_against,
            "HR": hr / pa_against,
            "OUT": 0.0,
        }
    )
    return rates, pa_against


def _build_pitcher_player(
    row: dict[str, Any],
    season: int,
    source: str,
    *,
    split_type: str = "overall",
) -> dict | None:
    ip = _safe_float(_first_present(row, "IP"))
    if ip <= 0:
        return None

    name = str(_first_present(row, "Name", default="")).strip()
    if not name:
        return None

    rates, pa_against = _extract_pitcher_rates(row)
    return {
        "player_id": str(_first_present(row, "IDfg", "ID", default=name)),
        "name": name,
        "team": str(_first_present(row, "Team", default="")).strip(),
        "throws": str(_first_present(row, "Throws", default="R")).strip().upper() or "R",
        "ip": ip,
        "pa_against": pa_against,
        "rates": rates,
        "avg_pitch_count": max(60.0, min(110.0, ip * _AVG_PITCHES_PER_INNING)),
        "season": season,
        "source": source,
        "split_type": split_type,
        "notes": f"FanGraphs pitching leaderboard data ({split_type}).",
    }


def _is_reliever_row(row: dict[str, Any]) -> bool:
    games = _safe_float(_first_present(row, "G", default=0.0))
    games_started = _safe_float(_first_present(row, "GS", default=0.0))
    if games <= 0:
        return False
    return games_started == 0 or (games_started / games) < 0.2


def _aggregate_team_bullpen_records(records: list[dict[str, Any]], season: int) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for row in records:
        team_code = str(_first_present(row, "Team", default="")).strip()
        if not team_code or not _is_reliever_row(row):
            continue

        team = aggregates.setdefault(
            team_code,
            {
                "ip": 0.0,
                "pa_against": 0,
                "H": 0.0,
                "2B": 0.0,
                "3B": 0.0,
                "HR": 0.0,
                "BB": 0.0,
                "HBP": 0.0,
                "SO": 0.0,
            },
        )
        team["ip"] += _safe_float(_first_present(row, "IP"))
        team["pa_against"] += _estimate_pitcher_pa(row)
        team["H"] += _safe_float(_first_present(row, "H"))
        team["2B"] += _safe_float(_first_present(row, "2B", "2B_Allowed"))
        team["3B"] += _safe_float(_first_present(row, "3B", "3B_Allowed"))
        team["HR"] += _safe_float(_first_present(row, "HR"))
        team["BB"] += _safe_float(_first_present(row, "BB"))
        team["HBP"] += _safe_float(_first_present(row, "HBP"))
        team["SO"] += _safe_float(_first_present(row, "SO", "K"))

    bullpen: dict[str, dict[str, Any]] = {}
    for team_code, totals in aggregates.items():
        pa_against = int(totals["pa_against"])
        if pa_against <= 0:
            continue
        hits = totals["H"]
        doubles = totals["2B"]
        triples = totals["3B"]
        hr = totals["HR"]
        singles = max(0.0, hits - doubles - triples - hr)
        rates = _normalize_rates(
            {
                "K": totals["SO"] / pa_against,
                "BB": totals["BB"] / pa_against,
                "HBP": totals["HBP"] / pa_against,
                "1B": singles / pa_against,
                "2B": doubles / pa_against,
                "3B": triples / pa_against,
                "HR": hr / pa_against,
                "OUT": 0.0,
            }
        )
        bullpen[team_code] = {
            "player_id": f"{team_code.lower()}-bullpen",
            "team": team_code,
            "name": f"{team_code} Bullpen",
            "throws": "R",
            "ip": totals["ip"],
            "pa_against": pa_against,
            "rates": rates,
            "avg_pitch_count": 120.0,
            "season": season,
            "source": str(season),
            "split_type": "team_relief",
            "notes": "Aggregate team bullpen using relievers only (GS == 0 or GS/G < 0.2).",
        }
    return bullpen


def _fetch_batting_season_raw(
    season: int,
    batting_stats,
    use_cache: bool,
    *,
    kind: str = "batting",
    month: int | None = None,
    split_type: str = "overall",
) -> dict[str, dict]:
    if use_cache:
        cached = _load_raw_cache(kind, season)
        if cached is not None:
            return cached

    fetch_kwargs: dict[str, Any] = {"qual": 1}
    if month is not None:
        fetch_kwargs["month"] = month
    records = _frame_to_records(batting_stats(season, **fetch_kwargs))
    players: dict[str, dict] = {}
    for row in records:
        source = f"{season}_{'split' if split_type != 'overall' else 'overall'}"
        player = _build_batter_player(row, season, source=source, split_type=split_type)
        if player is None:
            continue
        players[_normalize_name(player["name"])] = player

    _save_raw_cache(kind, season, players)
    return players


def _fetch_pitching_season_raw(
    season: int,
    pitching_stats,
    use_cache: bool,
    *,
    kind: str = "pitching",
    month: int | None = None,
    split_type: str = "overall",
) -> dict[str, dict]:
    if use_cache:
        cached = _load_raw_cache(kind, season)
        if cached is not None:
            return cached

    fetch_kwargs: dict[str, Any] = {"qual": 1}
    if month is not None:
        fetch_kwargs["month"] = month
    records = _frame_to_records(pitching_stats(season, **fetch_kwargs))
    players: dict[str, dict] = {}
    for row in records:
        source = f"{season}_{'split' if split_type != 'overall' else 'overall'}"
        player = _build_pitcher_player(row, season, source=source, split_type=split_type)
        if player is None:
            continue
        players[_normalize_name(player["name"])] = player

    _save_raw_cache(kind, season, players)
    return players


def _choose_split_record(
    current_player: dict[str, Any] | None,
    prior_player: dict[str, Any] | None,
    stat_key: str,
    threshold: int,
) -> dict[str, Any] | None:
    if current_player and _safe_int(current_player.get(stat_key)) >= threshold:
        return current_player
    if prior_player and _safe_int(prior_player.get(stat_key)) >= threshold:
        return prior_player
    return None


def _merge_batter_player(
    current_overall: dict[str, Any] | None,
    prior_overall: dict[str, Any] | None,
    current_vs_lhp: dict[str, Any] | None,
    prior_vs_lhp: dict[str, Any] | None,
    current_vs_rhp: dict[str, Any] | None,
    prior_vs_rhp: dict[str, Any] | None,
    threshold: int,
) -> dict[str, Any] | None:
    overall = None
    if current_overall and _safe_int(current_overall.get("pa")) >= threshold:
        overall = dict(current_overall)
    elif prior_overall:
        overall = dict(prior_overall)
    if overall is None:
        return None

    overall["overall"] = {
        "pa": _safe_int(overall.get("pa")),
        "rates": dict(overall.get("rates") or {}),
        "source": str(overall.get("source") or "unknown"),
        "season": _safe_int(overall.get("season")),
        "split_type": "overall",
    }
    splits: dict[str, dict[str, Any]] = {}
    vs_lhp = _choose_split_record(current_vs_lhp, prior_vs_lhp, "pa", _MIN_SPLIT_BATTER_PA)
    if vs_lhp is not None:
        splits["vs_lhp"] = {
            "pa": _safe_int(vs_lhp.get("pa")),
            "rates": dict(vs_lhp.get("rates") or {}),
            "source": str(vs_lhp.get("source") or "unknown"),
            "season": _safe_int(vs_lhp.get("season")),
            "split_type": "vs_lhp",
        }
    vs_rhp = _choose_split_record(current_vs_rhp, prior_vs_rhp, "pa", _MIN_SPLIT_BATTER_PA)
    if vs_rhp is not None:
        splits["vs_rhp"] = {
            "pa": _safe_int(vs_rhp.get("pa")),
            "rates": dict(vs_rhp.get("rates") or {}),
            "source": str(vs_rhp.get("source") or "unknown"),
            "season": _safe_int(vs_rhp.get("season")),
            "split_type": "vs_rhp",
        }
    overall["splits"] = splits
    return overall


def _merge_pitcher_player(
    current_overall: dict[str, Any] | None,
    prior_overall: dict[str, Any] | None,
    current_vs_lhb: dict[str, Any] | None,
    prior_vs_lhb: dict[str, Any] | None,
    current_vs_rhb: dict[str, Any] | None,
    prior_vs_rhb: dict[str, Any] | None,
    threshold: float,
) -> dict[str, Any] | None:
    overall = None
    if current_overall and _safe_float(current_overall.get("ip")) >= threshold:
        overall = dict(current_overall)
    elif prior_overall:
        overall = dict(prior_overall)
    if overall is None:
        return None

    overall["overall"] = {
        "pa_against": _safe_int(overall.get("pa_against")),
        "rates": dict(overall.get("rates") or {}),
        "source": str(overall.get("source") or "unknown"),
        "season": _safe_int(overall.get("season")),
        "split_type": "overall",
    }
    splits: dict[str, dict[str, Any]] = {}
    vs_lhb = _choose_split_record(current_vs_lhb, prior_vs_lhb, "pa_against", _MIN_SPLIT_PITCHER_BF)
    if vs_lhb is not None:
        splits["vs_lhb"] = {
            "pa_against": _safe_int(vs_lhb.get("pa_against")),
            "rates": dict(vs_lhb.get("rates") or {}),
            "source": str(vs_lhb.get("source") or "unknown"),
            "season": _safe_int(vs_lhb.get("season")),
            "split_type": "vs_lhb",
        }
    vs_rhb = _choose_split_record(current_vs_rhb, prior_vs_rhb, "pa_against", _MIN_SPLIT_PITCHER_BF)
    if vs_rhb is not None:
        splits["vs_rhb"] = {
            "pa_against": _safe_int(vs_rhb.get("pa_against")),
            "rates": dict(vs_rhb.get("rates") or {}),
            "source": str(vs_rhb.get("source") or "unknown"),
            "season": _safe_int(vs_rhb.get("season")),
            "split_type": "vs_rhb",
        }
    overall["splits"] = splits
    return overall


def fetch_batting_splits(season: int = SEASON, use_cache: bool = True) -> dict[str, dict]:
    """Fetch season batting stats and merge in handedness splits when supported."""
    batting_stats, _ = _import_pybaseball()
    current_players = _fetch_batting_season_raw(season, batting_stats, use_cache)
    prior_players = _fetch_batting_season_raw(season - 1, batting_stats, use_cache)
    # NOTE: leave the split-aware merge/selection pipeline in place for a future
    # supported split source. FanGraphs `month=13/14` currently fails both in
    # pybaseball's FangraphsMonth enum validation and as a live leaderboard
    # request, so we intentionally skip those requests for now and fall back to
    # overall rates downstream.
    # current_vs_lhp = _fetch_batting_season_raw(..., month=_SPLIT_MONTH_VS_LEFT, ...)
    # prior_vs_lhp = _fetch_batting_season_raw(..., month=_SPLIT_MONTH_VS_LEFT, ...)
    # current_vs_rhp = _fetch_batting_season_raw(..., month=_SPLIT_MONTH_VS_RIGHT, ...)
    # prior_vs_rhp = _fetch_batting_season_raw(..., month=_SPLIT_MONTH_VS_RIGHT, ...)
    current_vs_lhp: dict[str, dict] = {}
    prior_vs_lhp: dict[str, dict] = {}
    current_vs_rhp: dict[str, dict] = {}
    prior_vs_rhp: dict[str, dict] = {}
    players: dict[str, dict] = {}
    threshold = _batting_threshold_for_season(season)

    all_names = (
        set(current_players)
        | set(prior_players)
        | set(current_vs_lhp)
        | set(prior_vs_lhp)
        | set(current_vs_rhp)
        | set(prior_vs_rhp)
    )
    for name in all_names:
        player = _merge_batter_player(
            current_players.get(name),
            prior_players.get(name),
            current_vs_lhp.get(name),
            prior_vs_lhp.get(name),
            current_vs_rhp.get(name),
            prior_vs_rhp.get(name),
            threshold,
        )
        if player is not None:
            players[name] = player
    return players


def fetch_pitching_splits(season: int = SEASON, use_cache: bool = True) -> dict[str, dict]:
    """Fetch season pitching stats and merge in handedness splits when supported."""
    _, pitching_stats = _import_pybaseball()
    current_players = _fetch_pitching_season_raw(season, pitching_stats, use_cache)
    prior_players = _fetch_pitching_season_raw(season - 1, pitching_stats, use_cache)
    # NOTE: keep the split-profile plumbing intact so a future supported source
    # can populate these fields. For now we intentionally skip the unsupported
    # FanGraphs `month=13/14` requests and let the engine fall back to overall
    # pitcher rates when no split is present.
    # current_vs_lhb = _fetch_pitching_season_raw(..., month=_SPLIT_MONTH_VS_LEFT, ...)
    # prior_vs_lhb = _fetch_pitching_season_raw(..., month=_SPLIT_MONTH_VS_LEFT, ...)
    # current_vs_rhb = _fetch_pitching_season_raw(..., month=_SPLIT_MONTH_VS_RIGHT, ...)
    # prior_vs_rhb = _fetch_pitching_season_raw(..., month=_SPLIT_MONTH_VS_RIGHT, ...)
    current_vs_lhb: dict[str, dict] = {}
    prior_vs_lhb: dict[str, dict] = {}
    current_vs_rhb: dict[str, dict] = {}
    prior_vs_rhb: dict[str, dict] = {}
    players: dict[str, dict] = {}
    threshold = _pitching_threshold_for_season(season)

    all_names = (
        set(current_players)
        | set(prior_players)
        | set(current_vs_lhb)
        | set(prior_vs_lhb)
        | set(current_vs_rhb)
        | set(prior_vs_rhb)
    )
    for name in all_names:
        player = _merge_pitcher_player(
            current_players.get(name),
            prior_players.get(name),
            current_vs_lhb.get(name),
            prior_vs_lhb.get(name),
            current_vs_rhb.get(name),
            prior_vs_rhb.get(name),
            threshold,
        )
        if player is not None:
            players[name] = player
    return players


def fetch_computed_league_averages(season: int = SEASON, use_cache: bool = True) -> dict[str, float]:
    """Fetch and cache overall league averages computed from raw pybaseball leaderboards."""
    cache_key = (season, use_cache)
    cached = _COMPUTED_LEAGUE_AVERAGE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if use_cache:
        payload = _load_computed_cache("league_averages", season)
        if payload is not None and "rates" in payload:
            rates = dict(payload["rates"])
            _COMPUTED_LEAGUE_AVERAGE_CACHE[cache_key] = rates
            return rates

    batting_stats, pitching_stats = _import_pybaseball()
    batting_df = batting_stats(season, qual=1)
    pitching_df = pitching_stats(season, qual=1)
    rates = compute_league_averages(batting_df, pitching_df)
    _save_computed_cache("league_averages", season, {"season": season, "rates": rates})
    _COMPUTED_LEAGUE_AVERAGE_CACHE[cache_key] = rates
    return rates


def fetch_team_bullpen_stats(season: int = SEASON, use_cache: bool = True) -> dict[str, dict[str, Any]]:
    """Fetch team-level bullpen aggregate rates from reliever rows only."""
    cache_key = (season, use_cache)
    cached = _TEAM_BULLPEN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _, pitching_stats = _import_pybaseball()
    records = _frame_to_records(pitching_stats(season, qual=1))
    bullpen = _aggregate_team_bullpen_records(records, season)
    _TEAM_BULLPEN_CACHE[cache_key] = bullpen
    return bullpen


def fangraphs_team_code(team_name: str) -> str:
    """Map a full MLB team name to the Fangraphs team code used in stat rows."""
    return _TEAM_TO_FG_CODE.get(team_name, "")


def _effective_batter_hand_for_pitcher(batter_hand: Any, pitcher_throws: Hand) -> Hand:
    hand = _hand_from_value(batter_hand, allow_switch=True)
    if hand == Hand.SWITCH:
        return Hand.LEFT if pitcher_throws == Hand.RIGHT else Hand.RIGHT
    return hand


def _select_batter_rates(
    player_data: dict[str, Any],
    bats: Hand,
    pitcher_hand: Any | None,
    *,
    use_overall: bool = False,
) -> tuple[dict[str, float], int, str]:
    overall = dict(player_data.get("overall") or {})
    default_rates = dict(player_data.get("rates") or overall.get("rates") or _league_average_rates_for_batter(bats))
    default_pa = _safe_int(player_data.get("pa") or overall.get("pa"), 0)
    default_source = str(player_data.get("source") or overall.get("source") or "unknown")
    if use_overall or pitcher_hand is None:
        return default_rates, default_pa, default_source

    pitcher_throws = _hand_from_value(pitcher_hand)
    split_key = "vs_lhp" if pitcher_throws == Hand.LEFT else "vs_rhp"
    split = (player_data.get("splits") or {}).get(split_key)
    if split is None:
        return default_rates, default_pa, default_source
    return dict(split.get("rates") or default_rates), _safe_int(split.get("pa"), default_pa), str(split.get("source") or default_source)


def _select_pitcher_rates(
    player_data: dict[str, Any],
    throws: Hand,
    batter_hand: Any | None,
    *,
    use_overall: bool = False,
) -> tuple[dict[str, float], int, str]:
    overall = dict(player_data.get("overall") or {})
    default_rates = dict(player_data.get("rates") or overall.get("rates") or _league_average_rates_for_pitcher(throws))
    default_pa = _safe_int(player_data.get("pa_against") or overall.get("pa_against"), 0)
    default_source = str(player_data.get("source") or overall.get("source") or "unknown")
    if use_overall or batter_hand is None:
        return default_rates, default_pa, default_source

    effective_batter_hand = _effective_batter_hand_for_pitcher(batter_hand, throws)
    split_key = "vs_lhb" if effective_batter_hand == Hand.LEFT else "vs_rhb"
    split = (player_data.get("splits") or {}).get(split_key)
    if split is None:
        return default_rates, default_pa, default_source
    return dict(split.get("rates") or default_rates), _safe_int(split.get("pa_against"), default_pa), str(split.get("source") or default_source)


def build_batter_stats(player_data: dict, hand: str, pitcher_hand: Any | None = None, *, use_overall: bool = False) -> BatterStats:
    """Convert a raw player stats dict into BatterStats."""
    bats = _hand_from_value(hand or player_data.get("bats"), allow_switch=True)
    selected_rates, selected_pa, source = _select_batter_rates(player_data, bats, pitcher_hand, use_overall=use_overall)
    rates = _normalize_rates(selected_rates)
    return BatterStats(
        player_id=str(player_data.get("player_id") or player_data.get("id") or player_data.get("name") or "unknown"),
        name=str(player_data.get("name") or "Unknown Batter"),
        bats=bats,
        pa=selected_pa,
        rates=rates,
        data_source=source,
        split_profile=player_data,
    )


def build_pitcher_stats(player_data: dict, batter_hand: Any | None = None, *, use_overall: bool = False) -> PitcherStats:
    """Convert a raw player stats dict into PitcherStats."""
    throws = _hand_from_value(player_data.get("throws"))
    selected_rates, pa_against, source = _select_pitcher_rates(player_data, throws, batter_hand, use_overall=use_overall)
    rates = _normalize_rates(selected_rates)
    if pa_against <= 0:
        pa_against = _estimate_pitcher_pa(player_data)

    avg_pitch_count = _safe_float(player_data.get("avg_pitch_count"))
    if avg_pitch_count <= 0:
        ip = _safe_float(player_data.get("ip"))
        avg_pitch_count = max(60.0, min(110.0, ip * _AVG_PITCHES_PER_INNING)) if ip > 0 else 85.0

    return PitcherStats(
        player_id=str(player_data.get("player_id") or player_data.get("id") or player_data.get("name") or "unknown"),
        name=str(player_data.get("name") or "Unknown Pitcher"),
        throws=throws,
        pa_against=pa_against,
        rates=rates,
        avg_pitch_count=avg_pitch_count,
        data_source=source,
        split_profile=player_data,
    )
