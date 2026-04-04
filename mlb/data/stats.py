"""Player statistics fetch/build helpers for MLB simulation inputs."""
from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

from mlb.config import (
    CACHE_DIR,
    DAILY_CACHE_TTL_DAYS,
    Hand,
    LEAGUE_AVERAGES,
    SEASON,
    SEASONAL_CACHE_TTL_DAYS,
)
from mlb.data.cache import read_cache, write_cache
from mlb.data.models import BatterStats, PitcherStats

logger = logging.getLogger(__name__)

_BATTING_CACHE_KEY = "season_batting"
_PITCHING_CACHE_KEY = "season_pitching"
_MIN_BATTER_PA = 50
_MIN_PITCHER_IP = 20.0
_AVG_PITCHES_PER_INNING = 16.0
_EARLY_SEASON_BATTER_PA = 20
_EARLY_SEASON_PITCHER_IP = 10.0


def _today_str() -> str:
    return date.today().isoformat()


def _normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _cache_payload(players: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"players": players}


def _load_cached_players(category: str, key: str, use_cache: bool) -> dict[str, dict] | None:
    if not use_cache:
        return None
    cached = read_cache(
        category,
        key,
        date=_today_str(),
        max_age_days=DAILY_CACHE_TTL_DAYS,
    )
    if cached:
        players = cached.get("players", {})
        if players:
            return players
        logger.warning("Discarding empty cached %s/%s payload", category, key)
    return None


def _write_cached_players(category: str, key: str, players: dict[str, dict]) -> None:
    write_cache(category, key, _cache_payload(players), date=_today_str())


def _load_cached_season_players(
    category: str,
    key: str,
    season: int,
    use_cache: bool,
) -> dict[str, dict] | None:
    if not use_cache:
        return None
    cached = read_cache(
        category,
        f"{key}-{season}",
        max_age_days=SEASONAL_CACHE_TTL_DAYS,
    )
    if cached:
        players = cached.get("players", {})
        if players:
            return players
        logger.warning("Discarding empty cached %s/%s season payload", category, key)
    return None


def _write_cached_season_players(
    category: str,
    key: str,
    season: int,
    players: dict[str, dict],
) -> None:
    write_cache(category, f"{key}-{season}", _cache_payload(players))


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


def _fetch_batting_season_raw(
    season: int,
    batting_stats,
    use_cache: bool,
) -> dict[str, dict]:
    cached = _load_cached_season_players("batting_splits", "raw_batting", season, use_cache)
    if cached is not None:
        return cached

    records = _frame_to_records(batting_stats(season))
    players: dict[str, dict] = {}
    for row in records:
        player = _build_batter_player(row, season, source=str(season))
        if player is None:
            continue
        players[_normalize_name(player["name"])] = player

    _write_cached_season_players("batting_splits", "raw_batting", season, players)
    return players


def _fetch_pitching_season_raw(
    season: int,
    pitching_stats,
    use_cache: bool,
) -> dict[str, dict]:
    cached = _load_cached_season_players("pitching_splits", "raw_pitching", season, use_cache)
    if cached is not None:
        return cached

    records = _frame_to_records(pitching_stats(season))
    players: dict[str, dict] = {}
    for row in records:
        player = _build_pitcher_player(row, season, source=str(season))
        if player is None:
            continue
        players[_normalize_name(player["name"])] = player

    _write_cached_season_players("pitching_splits", "raw_pitching", season, players)
    return players


def fetch_batting_splits(season: int = SEASON, use_cache: bool = True) -> dict[str, dict]:
    """Fetch season batting stats and convert them to per-PA rates.

    v1 uses overall rates for all matchups. True handedness splits require
    individual split-page requests, which are too slow for this pass.
    """
    cached = _load_cached_players("batting_splits", _BATTING_CACHE_KEY, use_cache)
    if cached is not None:
        return cached

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

    _write_cached_players("batting_splits", _BATTING_CACHE_KEY, players)
    return players


def fetch_pitching_splits(season: int = SEASON, use_cache: bool = True) -> dict[str, dict]:
    """Fetch season pitching stats and convert them to per-batter-faced rates."""
    cached = _load_cached_players("pitching_splits", _PITCHING_CACHE_KEY, use_cache)
    if cached is not None:
        return cached

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

    _write_cached_players("pitching_splits", _PITCHING_CACHE_KEY, players)
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
