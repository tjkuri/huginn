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
_AVG_PITCHES_PER_INNING = 16.0
_EARLY_SEASON_BATTER_PA = 20
_EARLY_SEASON_PITCHER_IP = 10.0

_NAME_SUFFIX_RE = re.compile(r'\s+(jr|sr|ii|iii|iv)$')


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


def _build_batter_player(row: dict[str, Any], season: int, source: str) -> dict | None:
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
        "split_type": "overall",
        "notes": (
            "TODO: replace with true handedness splits via pybaseball split pages "
            "or an ID-mapped split source."
        ),
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


def _build_pitcher_player(row: dict[str, Any], season: int, source: str) -> dict | None:
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
        "split_type": "overall",
        "notes": "TODO: add handedness splits and stronger ID mapping than player-name matching.",
    }


def _fetch_batting_season_raw(season: int, batting_stats, use_cache: bool) -> dict[str, dict]:
    if use_cache:
        cached = _load_raw_cache("batting", season)
        if cached is not None:
            return cached

    records = _frame_to_records(batting_stats(season, qual=1))
    players: dict[str, dict] = {}
    for row in records:
        player = _build_batter_player(row, season, source=str(season))
        if player is None:
            continue
        players[_normalize_name(player["name"])] = player

    _save_raw_cache("batting", season, players)
    return players


def _fetch_pitching_season_raw(season: int, pitching_stats, use_cache: bool) -> dict[str, dict]:
    if use_cache:
        cached = _load_raw_cache("pitching", season)
        if cached is not None:
            return cached

    records = _frame_to_records(pitching_stats(season, qual=1))
    players: dict[str, dict] = {}
    for row in records:
        player = _build_pitcher_player(row, season, source=str(season))
        if player is None:
            continue
        players[_normalize_name(player["name"])] = player

    _save_raw_cache("pitching", season, players)
    return players


def fetch_batting_splits(season: int = SEASON, use_cache: bool = True) -> dict[str, dict]:
    """Fetch season batting stats and convert them to per-PA rates.

    v1 uses overall rates for all matchups. True handedness splits require
    individual split-page requests, which are too slow for this pass.
    """
    batting_stats, _ = _import_pybaseball()
    current_players = _fetch_batting_season_raw(season, batting_stats, use_cache)
    prior_players = _fetch_batting_season_raw(season - 1, batting_stats, use_cache)
    players: dict[str, dict] = {}
    threshold = _batting_threshold_for_season(season)

    all_names = set(current_players) | set(prior_players)
    for name in all_names:
        current_player = current_players.get(name)
        prior_player = prior_players.get(name)
        if current_player and _safe_int(current_player.get("pa")) >= threshold:
            players[name] = dict(current_player, source=str(season))
        elif prior_player:
            players[name] = dict(prior_player, source=str(season - 1))
    return players


def fetch_pitching_splits(season: int = SEASON, use_cache: bool = True) -> dict[str, dict]:
    """Fetch season pitching stats and convert them to per-batter-faced rates."""
    _, pitching_stats = _import_pybaseball()
    current_players = _fetch_pitching_season_raw(season, pitching_stats, use_cache)
    prior_players = _fetch_pitching_season_raw(season - 1, pitching_stats, use_cache)
    players: dict[str, dict] = {}
    threshold = _pitching_threshold_for_season(season)

    all_names = set(current_players) | set(prior_players)
    for name in all_names:
        current_player = current_players.get(name)
        prior_player = prior_players.get(name)
        if current_player and _safe_float(current_player.get("ip")) >= threshold:
            players[name] = dict(current_player, source=str(season))
        elif prior_player:
            players[name] = dict(prior_player, source=str(season - 1))
    return players


def build_batter_stats(player_data: dict, hand: str) -> BatterStats:
    """Convert a raw player stats dict into BatterStats."""
    bats = _hand_from_value(hand or player_data.get("bats"), allow_switch=True)
    rates = _normalize_rates(dict(player_data.get("rates") or _league_average_rates_for_batter(bats)))
    return BatterStats(
        player_id=str(player_data.get("player_id") or player_data.get("id") or player_data.get("name") or "unknown"),
        name=str(player_data.get("name") or "Unknown Batter"),
        bats=bats,
        pa=_safe_int(player_data.get("pa"), 0),
        rates=rates,
        data_source=str(player_data.get("source") or "unknown"),
    )


def build_pitcher_stats(player_data: dict) -> PitcherStats:
    """Convert a raw player stats dict into PitcherStats."""
    throws = _hand_from_value(player_data.get("throws"))
    rates = _normalize_rates(dict(player_data.get("rates") or _league_average_rates_for_pitcher(throws)))
    pa_against = _safe_int(player_data.get("pa_against"), 0)
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
        data_source=str(player_data.get("source") or "unknown"),
    )
