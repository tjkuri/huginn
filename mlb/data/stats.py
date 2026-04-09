"""Player statistics fetch/build helpers for MLB simulation inputs."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from mlb.config import (
    CACHE_DIR,
    Hand,
    LEAGUE_AVERAGES,
    SEASON,
    STATS_CACHE_MAX_AGE_HOURS,
)
from mlb.data.mlb_stats_api import (
    fetch_batting_season_rows,
    fetch_batting_split_rows,
    fetch_pitching_season_rows,
    fetch_pitching_split_rows,
)
from mlb.data.models import BatterStats, DataSourceStatus, PitcherStats
from mlb.data.team_codes import TEAM_TO_CODE as _TEAM_TO_FG_CODE
from mlb.utils.normalize import normalize_name as _normalize_name

logger = logging.getLogger(__name__)

_AVG_PITCHES_PER_INNING = 16.0
_BATTING_SPLIT_SIT_CODES = {
    "vs_lhp": "vl",
    "vs_rhp": "vr",
}
_PITCHING_SPLIT_SIT_CODES = {
    "vs_lhb": "vl",
    "vs_rhb": "vr",
}
_BATTER_REGRESSION_CONSTANTS: dict[str, float] = {
    "BB": 200.0,
    "K": 150.0,
    "HR": 320.0,
    "1B": 200.0,
    "2B": 400.0,
    "3B": 800.0,
    "HBP": 400.0,
}
# Halved constants for the split-ratio Marcel (regresses ratio toward 1.0, not an absolute rate)
_BATTER_RATIO_REGRESSION_CONSTANTS: dict[str, float] = {
    "BB": 100.0,
    "K": 75.0,
    "HR": 160.0,
    "1B": 100.0,
    "2B": 200.0,
    "3B": 400.0,
    "HBP": 200.0,
}
_PITCHER_REGRESSION_CONSTANT = 150.0
_PITCHER_RATIO_REGRESSION_CONSTANT = 75.0
_BATTER_NORMALIZER = 200.0
_PITCHER_NORMALIZER = 150.0
_MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"

_TEAM_BULLPEN_CACHE: dict[tuple[int, bool], dict[str, dict[str, Any]]] = {}
_COMPUTED_LEAGUE_AVERAGE_CACHE: dict[tuple[int, bool], dict[str, float]] = {}
_MATCHUP_LEAGUE_AVERAGE_CACHE: dict[tuple[int, bool], dict[tuple[Hand, Hand], dict[str, float]]] = {}
_PLAYER_HANDEDNESS_CACHE: dict[str, dict[str, str]] = {}
# ── Raw season stat cache (flat files, no subdirectories) ────────────────────

def _raw_cache_path(kind: str, season: int) -> Path:
    return CACHE_DIR / f"raw_{kind}-{season}.json"


def _load_raw_cache_payload(kind: str, season: int) -> dict[str, Any] | None:
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
        return json.load(f)


def _load_raw_cache(kind: str, season: int) -> dict[str, dict] | None:
    payload = _load_raw_cache_payload(kind, season)
    if payload is None:
        return None
    return payload.get("players")


def _load_raw_records_cache(kind: str, season: int) -> list[dict[str, Any]] | None:
    payload = _load_raw_cache_payload(kind, season)
    if payload is None:
        return None
    records = payload.get("records")
    return records if isinstance(records, list) else None


def _save_raw_cache(
    kind: str,
    season: int,
    players: dict[str, dict],
    *,
    records: list[dict[str, Any]] | None = None,
) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_cache_path(kind, season)
    payload: dict[str, Any] = {"players": players}
    if records is not None:
        payload["records"] = records
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
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


# ── Marcel regression ────────────────────────────────────────────────────────


def marcel_blend(
    seasons: list[tuple[int, float, float]],
    regression_constant: float,
    league_avg: float,
    normalizer: float = 200.0,
) -> float:
    """Marcel projection for a single rate stat.

    Pure function — no I/O, no side effects.

    Args:
        seasons: List of (year_coefficient, observed_rate, sample_size) per season.
                 year_coefficient is 5 (current), 4 (prior), 3 (two-prior).
                 Missing seasons are excluded from the list.
        regression_constant: Regression weight in sample-size equivalents.
        league_avg: League-average rate (regression target).
        normalizer: 200 for batters (PA), 150 for pitchers (BF).

    Returns:
        Marcel-projected rate.
    """
    total_num = sum(yc * (pa / normalizer) * rate for yc, rate, pa in seasons)
    total_den = sum(yc * (pa / normalizer) for yc, rate, pa in seasons)
    w_lg = regression_constant / normalizer
    total_num += w_lg * league_avg
    total_den += w_lg
    if total_den <= 0:
        return league_avg
    return total_num / total_den



def _overall_league_avg_rates() -> dict[str, float]:
    """Simple average of all four matchup league averages (Marcel regression target for overall rates)."""
    stats = ("K", "BB", "HBP", "1B", "2B", "3B", "HR")
    matchups = list(LEAGUE_AVERAGES.values())
    result = {stat: sum(m.get(stat, 0.0) for m in matchups) / len(matchups) for stat in stats}
    result["OUT"] = 0.0
    return _normalize_rates(result)


def _marcel_source_tag(n_seasons: int) -> str:
    """Return the Marcel source tag based on how many seasons had usable data."""
    if n_seasons >= 3:
        return "marcel_3yr"
    if n_seasons == 2:
        return "marcel_2yr"
    if n_seasons == 1:
        return "marcel_1yr"
    return "league_avg"


def _apply_marcel(
    seasons: list[tuple[int, dict[str, float], float]],
    regression_constants: dict[str, float],
    league_avg_rates: dict[str, float],
    normalizer: float,
) -> tuple[dict[str, float], int]:
    """Apply Marcel blend to all rate stats for one player at one split level.

    Args:
        seasons: List of (year_coefficient, rates_dict, sample_size).
        regression_constants: Per-stat regression constants in sample-size equivalents.
        league_avg_rates: League-average rate dict (regression target).
        normalizer: 200 for batters (PA), 150 for pitchers (BF).

    Returns:
        (blended_rates, n_seasons_with_data) where rates are normalized to 1.0.
    """
    seasons_with_data = [(yc, rates, pa) for yc, rates, pa in seasons if pa > 0]
    n = len(seasons_with_data)
    if n == 0:
        return dict(league_avg_rates), 0

    blended: dict[str, float] = {}
    for stat in ("K", "BB", "HBP", "1B", "2B", "3B", "HR"):
        season_data = [(yc, rates.get(stat, 0.0), pa) for yc, rates, pa in seasons_with_data]
        blended[stat] = marcel_blend(
            season_data,
            regression_constants[stat],
            league_avg_rates.get(stat, 0.0),
            normalizer,
        )
    blended["OUT"] = 0.0  # derived by _normalize_rates
    return _normalize_rates(blended), n


def _apply_marcel_ratios(
    ratio_seasons: list[tuple[int, dict[str, float], float]],
    regression_constants: dict[str, float],
    normalizer: float,
) -> tuple[dict[str, float], int]:
    """Marcel-blend per-season split/overall ratios toward 1.0 (no platoon effect).

    Args:
        ratio_seasons: List of (year_coefficient, ratio_rates_dict, sample_size) where
                       each ratio is split_rate[stat] / overall_rate[stat] for that season.
        regression_constants: Halved per-stat regression constants (ratio regression).
        normalizer: 200 for batters (PA), 150 for pitchers (BF).

    Returns:
        (blended_ratios, n_seasons_with_data). Ratios are multipliers, NOT normalized.
    """
    seasons_with_data = [(yc, rates, pa) for yc, rates, pa in ratio_seasons if pa > 0]
    n = len(seasons_with_data)
    if n == 0:
        return {stat: 1.0 for stat in ("K", "BB", "HBP", "1B", "2B", "3B", "HR")}, 0

    blended: dict[str, float] = {}
    for stat in ("K", "BB", "HBP", "1B", "2B", "3B", "HR"):
        season_data = [(yc, rates.get(stat, 1.0), pa) for yc, rates, pa in seasons_with_data]
        blended[stat] = marcel_blend(
            season_data,
            regression_constants[stat],
            1.0,  # regression target: no platoon effect
            normalizer,
        )
    return blended, n


def _marcel_batter_player(
    s1_overall: dict[str, Any] | None,
    s2_overall: dict[str, Any] | None,
    s3_overall: dict[str, Any] | None,
    s1_vs_lhp: dict[str, Any] | None,
    s2_vs_lhp: dict[str, Any] | None,
    s3_vs_lhp: dict[str, Any] | None,
    s1_vs_rhp: dict[str, Any] | None,
    s2_vs_rhp: dict[str, Any] | None,
    s3_vs_rhp: dict[str, Any] | None,
    bats: Hand,
    overall_league_avg: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Apply Marcel projection to one batter across up to three seasons.

    Returns a merged player dict (same structure as the old merge functions) or None if no data.
    s1=current season (year_coeff=5), s2=prior (4), s3=two-prior (3).
    """
    overall_seasons = [
        (yc, dict(player.get("rates") or {}), _safe_int(player.get("pa")))
        for yc, player in ((5, s1_overall), (4, s2_overall), (3, s3_overall))
        if player and _safe_int(player.get("pa")) > 0
    ]
    if not overall_seasons:
        return None

    league_avg = overall_league_avg if overall_league_avg is not None else _overall_league_avg_rates()
    blended_rates, n_seasons = _apply_marcel(overall_seasons, _BATTER_REGRESSION_CONSTANTS, league_avg, _BATTER_NORMALIZER)
    source_tag = _marcel_source_tag(n_seasons)

    primary = s1_overall or s2_overall or s3_overall
    projected_pa = (
        _safe_int((s1_overall or {}).get("pa"))
        or _safe_int((s2_overall or {}).get("pa"))
        or _safe_int((s3_overall or {}).get("pa"))
    )

    result = dict(primary)
    result["pa"] = projected_pa
    result["rates"] = blended_rates
    result["source"] = source_tag
    result["overall"] = {
        "pa": projected_pa,
        "rates": blended_rates,
        "source": source_tag,
        "split_type": "overall",
    }

    # Two-step split projection: anchor to overall Marcel, apply ratio adjustment.
    # Step 1 (overall Marcel) is already done above.
    # Step 2: for each split, Marcel-blend per-season ratios (split/overall) toward 1.0,
    # then multiply overall rates by those ratios. Thin split history regresses ratio
    # toward 1.0 so the projection stays close to overall talent.
    splits: dict[str, dict[str, Any]] = {}
    for split_key, pitcher_hand, sp1, sp2, sp3 in (
        ("vs_lhp", Hand.LEFT, s1_vs_lhp, s2_vs_lhp, s3_vs_lhp),
        ("vs_rhp", Hand.RIGHT, s1_vs_rhp, s2_vs_rhp, s3_vs_rhp),
    ):
        ratio_seasons = []
        for yc, sp, ov in (
            (5, sp1, s1_overall),
            (4, sp2, s2_overall),
            (3, sp3, s3_overall),
        ):
            if not sp or _safe_int(sp.get("pa")) <= 0:
                continue
            sp_rates = dict(sp.get("rates") or {})
            ov_rates = dict((ov or {}).get("rates") or {})
            ratio_rates = {
                stat: (sp_rates.get(stat, 0.0) / ov_rates[stat] if ov_rates.get(stat, 0.0) > 0 else 1.0)
                for stat in ("K", "BB", "HBP", "1B", "2B", "3B", "HR")
            }
            ratio_seasons.append((yc, ratio_rates, _safe_int(sp.get("pa"))))

        if not ratio_seasons:
            continue

        blended_ratios, split_n = _apply_marcel_ratios(
            ratio_seasons, _BATTER_RATIO_REGRESSION_CONSTANTS, _BATTER_NORMALIZER
        )
        split_rates = {
            stat: blended_rates[stat] * blended_ratios.get(stat, 1.0)
            for stat in ("K", "BB", "HBP", "1B", "2B", "3B", "HR")
        }
        split_rates["OUT"] = 0.0
        split_rates = _normalize_rates(split_rates)

        split_pa = (
            _safe_int((sp1 or {}).get("pa"))
            or _safe_int((sp2 or {}).get("pa"))
            or _safe_int((sp3 or {}).get("pa"))
        )
        splits[split_key] = {
            "pa": split_pa,
            "rates": split_rates,
            "source": _marcel_source_tag(split_n),
            "split_type": split_key,
        }

    result["splits"] = splits
    return result


def _marcel_pitcher_player(
    s1_overall: dict[str, Any] | None,
    s2_overall: dict[str, Any] | None,
    s3_overall: dict[str, Any] | None,
    s1_vs_lhb: dict[str, Any] | None,
    s2_vs_lhb: dict[str, Any] | None,
    s3_vs_lhb: dict[str, Any] | None,
    s1_vs_rhb: dict[str, Any] | None,
    s2_vs_rhb: dict[str, Any] | None,
    s3_vs_rhb: dict[str, Any] | None,
    overall_league_avg: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Apply Marcel projection to one pitcher across up to three seasons.

    Returns a merged player dict or None if no data.
    s1=current season (year_coeff=5), s2=prior (4), s3=two-prior (3).
    """
    overall_seasons = [
        (yc, dict(player.get("rates") or {}), _safe_int(player.get("pa_against")))
        for yc, player in ((5, s1_overall), (4, s2_overall), (3, s3_overall))
        if player and _safe_int(player.get("pa_against")) > 0
    ]
    if not overall_seasons:
        return None

    regression_constants = {stat: _PITCHER_REGRESSION_CONSTANT for stat in ("K", "BB", "HBP", "1B", "2B", "3B", "HR")}
    league_avg = overall_league_avg if overall_league_avg is not None else _overall_league_avg_rates()
    blended_rates, n_seasons = _apply_marcel(overall_seasons, regression_constants, league_avg, _PITCHER_NORMALIZER)
    source_tag = _marcel_source_tag(n_seasons)

    primary = s1_overall or s2_overall or s3_overall
    throws_hand = _hand_from_value(primary.get("throws"))
    projected_pa = (
        _safe_int((s1_overall or {}).get("pa_against"))
        or _safe_int((s2_overall or {}).get("pa_against"))
        or _safe_int((s3_overall or {}).get("pa_against"))
    )
    projected_ip = (
        _safe_float((s1_overall or {}).get("ip"))
        or _safe_float((s2_overall or {}).get("ip"))
        or _safe_float((s3_overall or {}).get("ip"))
    )

    result = dict(primary)
    result["pa_against"] = projected_pa
    result["rates"] = blended_rates
    result["source"] = source_tag
    result["avg_pitch_count"] = max(60.0, min(110.0, projected_ip * _AVG_PITCHES_PER_INNING)) if projected_ip > 0 else 85.0
    result["overall"] = {
        "pa_against": projected_pa,
        "rates": blended_rates,
        "source": source_tag,
        "split_type": "overall",
    }

    # Two-step split projection: anchor to overall Marcel, apply ratio adjustment.
    ratio_regression_constants = {stat: _PITCHER_RATIO_REGRESSION_CONSTANT for stat in ("K", "BB", "HBP", "1B", "2B", "3B", "HR")}
    splits: dict[str, dict[str, Any]] = {}
    for split_key, batter_hand, sp1, sp2, sp3 in (
        ("vs_lhb", Hand.LEFT, s1_vs_lhb, s2_vs_lhb, s3_vs_lhb),
        ("vs_rhb", Hand.RIGHT, s1_vs_rhb, s2_vs_rhb, s3_vs_rhb),
    ):
        ratio_seasons = []
        for yc, sp, ov in (
            (5, sp1, s1_overall),
            (4, sp2, s2_overall),
            (3, sp3, s3_overall),
        ):
            if not sp or _safe_int(sp.get("pa_against")) <= 0:
                continue
            sp_rates = dict(sp.get("rates") or {})
            ov_rates = dict((ov or {}).get("rates") or {})
            ratio_rates = {
                stat: (sp_rates.get(stat, 0.0) / ov_rates[stat] if ov_rates.get(stat, 0.0) > 0 else 1.0)
                for stat in ("K", "BB", "HBP", "1B", "2B", "3B", "HR")
            }
            ratio_seasons.append((yc, ratio_rates, _safe_int(sp.get("pa_against"))))

        if not ratio_seasons:
            continue

        blended_ratios, split_n = _apply_marcel_ratios(
            ratio_seasons, ratio_regression_constants, _PITCHER_NORMALIZER
        )
        split_rates = {
            stat: blended_rates[stat] * blended_ratios.get(stat, 1.0)
            for stat in ("K", "BB", "HBP", "1B", "2B", "3B", "HR")
        }
        split_rates["OUT"] = 0.0
        split_rates = _normalize_rates(split_rates)

        split_pa = (
            _safe_int((sp1 or {}).get("pa_against"))
            or _safe_int((sp2 or {}).get("pa_against"))
            or _safe_int((sp3 or {}).get("pa_against"))
        )
        splits[split_key] = {
            "pa_against": split_pa,
            "rates": split_rates,
            "source": _marcel_source_tag(split_n),
            "split_type": split_key,
        }

    result["splits"] = splits
    return result


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


def _is_valid_hand_code(value: Any, *, allow_switch: bool = False) -> bool:
    text = str(value or "").strip().upper()
    valid = {Hand.LEFT.value, Hand.RIGHT.value}
    if allow_switch:
        valid.add(Hand.SWITCH.value)
    return text in valid


def _normalize_rates(rates: dict[str, float]) -> dict[str, float]:
    sanitized = {key: max(0.0, float(value)) for key, value in rates.items()}
    non_out_total = sum(value for key, value in sanitized.items() if key != "OUT")
    sanitized["OUT"] = max(0.0, 1.0 - non_out_total)
    total = sum(sanitized.values())
    if total <= 0:
        return {"K": 0.0, "BB": 0.0, "HBP": 0.0, "1B": 0.0, "2B": 0.0, "3B": 0.0, "HR": 0.0, "OUT": 1.0}
    return {key: value / total for key, value in sanitized.items()}


def _copy_league_averages(
    source: dict[tuple[Hand, Hand], dict[str, float]] | None = None,
) -> dict[tuple[Hand, Hand], dict[str, float]]:
    baseline = source or LEAGUE_AVERAGES
    return {key: dict(rates) for key, rates in baseline.items()}


def _league_average_rates_for_batter(hand: Hand, pitcher_hand: Hand) -> dict[str, float]:
    return dict(LEAGUE_AVERAGES[(hand, pitcher_hand)])


def _league_average_rates_for_pitcher(throws: Hand, batter_hand: Hand) -> dict[str, float]:
    effective_batter_hand = _effective_batter_hand_for_pitcher(batter_hand, throws)
    return dict(LEAGUE_AVERAGES[(effective_batter_hand, throws)])


def _serialize_matchup_rates(
    rates_by_matchup: dict[tuple[Hand, Hand], dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {f"{batter.value}_vs_{pitcher.value}": dict(rates) for (batter, pitcher), rates in rates_by_matchup.items()}


def _deserialize_matchup_rates(payload: dict[str, dict[str, float]]) -> dict[tuple[Hand, Hand], dict[str, float]]:
    matchup_rates: dict[tuple[Hand, Hand], dict[str, float]] = {}
    for key, rates in payload.items():
        batter_code, _, pitcher_code = key.partition("_vs_")
        if not pitcher_code:
            continue
        matchup_rates[(Hand(batter_code), Hand(pitcher_code))] = dict(rates)
    return matchup_rates


def _apply_runtime_league_averages(rates_by_matchup: dict[tuple[Hand, Hand], dict[str, float]]) -> None:
    for key, rates in rates_by_matchup.items():
        LEAGUE_AVERAGES[key].clear()
        LEAGUE_AVERAGES[key].update(dict(rates))


_HARDCODED_LEAGUE_AVERAGES = _copy_league_averages()


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


def _import_playerid_reverse_lookup():
    try:
        from pybaseball import playerid_reverse_lookup
    except ImportError as exc:
        raise ImportError(
            "pybaseball is required for MLB stats fetching. Install it with "
            "`venv/bin/pip install pybaseball`."
        ) from exc
    return playerid_reverse_lookup


def _chunked(values: list[int], size: int) -> list[list[int]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def _fetch_people_handedness(person_ids: list[int]) -> dict[str, dict[str, str]]:
    handedness: dict[str, dict[str, str]] = {}
    for batch in _chunked(person_ids, 50):
        response = requests.get(
            _MLB_PEOPLE_URL,
            params={"personIds": ",".join(str(person_id) for person_id in batch)},
            timeout=20,
        )
        response.raise_for_status()
        people = response.json().get("people") or []
        for person in people:
            mlbam_id = str(person.get("id") or "").strip()
            if not mlbam_id:
                continue
            handedness[mlbam_id] = {
                "bats": str(((person.get("batSide") or {}).get("code")) or "").strip().upper(),
                "throws": str(((person.get("pitchHand") or {}).get("code")) or "").strip().upper(),
            }
    return handedness


def _resolve_player_handedness(players: dict[str, dict], *, role: str) -> dict[str, dict[str, str]]:
    missing_fg_ids = sorted(
        {
            int(str(player.get("player_id") or "").strip())
            for player in players.values()
            if str(player.get("player_id") or "").strip().isdigit()
            and str(player.get("player_id")) not in _PLAYER_HANDEDNESS_CACHE
        }
    )
    if missing_fg_ids:
        playerid_reverse_lookup = _import_playerid_reverse_lookup()
        lookup = playerid_reverse_lookup(missing_fg_ids, key_type="fangraphs")
        mlbam_ids: list[int] = []
        fg_to_mlbam: dict[str, str] = {}
        for row in _frame_to_records(lookup):
            fg_id = str(_first_present(row, "key_fangraphs", default="")).strip()
            mlbam_id = str(_first_present(row, "key_mlbam", default="")).strip()
            if not fg_id or not mlbam_id:
                continue
            fg_to_mlbam[fg_id] = mlbam_id
            mlbam_ids.append(int(mlbam_id))

        mlbam_handedness = _fetch_people_handedness(sorted(set(mlbam_ids)))
        for fg_id, mlbam_id in fg_to_mlbam.items():
            info = mlbam_handedness.get(mlbam_id) or {}
            _PLAYER_HANDEDNESS_CACHE[fg_id] = {
                "bats": str(info.get("bats") or ""),
                "throws": str(info.get("throws") or ""),
            }

    resolved: dict[str, dict[str, str]] = {}
    for player in players.values():
        fg_id = str(player.get("player_id") or "").strip()
        if not fg_id:
            continue
        info = _PLAYER_HANDEDNESS_CACHE.get(fg_id) or {}
        hand_value = str(info.get("bats" if role == "batter" else "throws") or "").strip().upper()
        if hand_value:
            resolved[fg_id] = info
    return resolved


def _enrich_batter_handedness(players: dict[str, dict], season: int, kind: str) -> dict[str, dict]:
    needs_enrichment = any(not _is_valid_hand_code(player.get("bats"), allow_switch=True) for player in players.values())
    if not needs_enrichment:
        return players

    handedness = _resolve_player_handedness(players, role="batter")
    updated = False
    for player in players.values():
        fg_id = str(player.get("player_id") or "").strip()
        bats = str((handedness.get(fg_id) or {}).get("bats") or "").strip().upper()
        if _is_valid_hand_code(bats, allow_switch=True):
            player["bats"] = bats
            updated = True
    if updated:
        _save_raw_cache(kind, season, players)
    return players


def _enrich_pitcher_handedness(players: dict[str, dict], season: int, kind: str) -> dict[str, dict]:
    needs_enrichment = any(not _is_valid_hand_code(player.get("throws")) for player in players.values())
    if not needs_enrichment:
        return players

    handedness = _resolve_player_handedness(players, role="pitcher")
    updated = False
    for player in players.values():
        fg_id = str(player.get("player_id") or "").strip()
        throws = str((handedness.get(fg_id) or {}).get("throws") or "").strip().upper()
        if _is_valid_hand_code(throws):
            player["throws"] = throws
            updated = True
    if updated:
        _save_raw_cache(kind, season, players)
    return players


def _frame_to_records(frame: Any) -> list[dict[str, Any]]:
    if isinstance(frame, list):
        return frame
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
    k_rate = _safe_float(_first_present(row, "K%", "SO%", "K_pct", default=-1.0), default=-1.0)
    bb_rate = _safe_float(_first_present(row, "BB%", "BB_pct", default=-1.0), default=-1.0)
    if k_rate < 0:
        k_rate = _safe_float(_first_present(row, "SO", "K")) / pa
    if bb_rate < 0:
        bb_rate = _safe_float(_first_present(row, "BB")) / pa

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


def _fetch_batting_season_records_with_status(
    season: int,
    use_cache: bool,
    *,
    kind: str = "batting",
) -> tuple[list[dict[str, Any]], str]:
    if use_cache:
        cached_records = _load_raw_records_cache(kind, season)
        if cached_records is not None:
            return cached_records, "cache"
    return fetch_batting_season_rows(season), "fresh"


def _fetch_pitching_season_records_with_status(
    season: int,
    use_cache: bool,
    *,
    kind: str = "pitching",
) -> tuple[list[dict[str, Any]], str]:
    if use_cache:
        cached_records = _load_raw_records_cache(kind, season)
        if cached_records is not None:
            return cached_records, "cache"
    return fetch_pitching_season_rows(season), "fresh"


def _build_batting_players_from_records(
    records: list[dict[str, Any]],
    season: int,
    *,
    split_type: str,
) -> dict[str, dict]:
    players: dict[str, dict] = {}
    for row in records:
        source = f"{season}_{'split' if split_type != 'overall' else 'overall'}"
        player = _build_batter_player(row, season, source=source, split_type=split_type)
        if player is None:
            continue
        players[_normalize_name(player["name"])] = player
    return players


def _build_pitching_players_from_records(
    records: list[dict[str, Any]],
    season: int,
    *,
    split_type: str,
) -> dict[str, dict]:
    players: dict[str, dict] = {}
    for row in records:
        source = f"{season}_{'split' if split_type != 'overall' else 'overall'}"
        player = _build_pitcher_player(row, season, source=source, split_type=split_type)
        if player is None:
            continue
        players[_normalize_name(player["name"])] = player
    return players


def _fetch_batting_season_raw(
    season: int,
    use_cache: bool,
    *,
    kind: str = "batting",
    split_type: str = "overall",
) -> dict[str, dict]:
    players, _ = _fetch_batting_season_raw_with_status(
        season,
        use_cache,
        kind=kind,
        split_type=split_type,
    )
    return players


def _fetch_batting_season_raw_with_status(
    season: int,
    use_cache: bool,
    *,
    kind: str = "batting",
    split_type: str = "overall",
    records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, dict], str]:
    if use_cache:
        cached = _load_raw_cache(kind, season)
        if cached is not None:
            return _enrich_batter_handedness(cached, season, kind), "cache"

    resolved_records, source_tier = (
        (records, "fresh")
        if records is not None
        else _fetch_batting_season_records_with_status(season, use_cache, kind=kind)
    )
    players = _build_batting_players_from_records(resolved_records, season, split_type=split_type)

    players = _enrich_batter_handedness(players, season, kind)
    _save_raw_cache(kind, season, players, records=resolved_records)
    return players, source_tier


def _fetch_pitching_season_raw(
    season: int,
    use_cache: bool,
    *,
    kind: str = "pitching",
    split_type: str = "overall",
) -> dict[str, dict]:
    players, _ = _fetch_pitching_season_raw_with_status(
        season,
        use_cache,
        kind=kind,
        split_type=split_type,
    )
    return players


def _fetch_pitching_season_raw_with_status(
    season: int,
    use_cache: bool,
    *,
    kind: str = "pitching",
    split_type: str = "overall",
    records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, dict], str]:
    if use_cache:
        cached = _load_raw_cache(kind, season)
        if cached is not None:
            return _enrich_pitcher_handedness(cached, season, kind), "cache"

    resolved_records, source_tier = (
        (records, "fresh")
        if records is not None
        else _fetch_pitching_season_records_with_status(season, use_cache, kind=kind)
    )
    players = _build_pitching_players_from_records(resolved_records, season, split_type=split_type)

    players = _enrich_pitcher_handedness(players, season, kind)
    _save_raw_cache(kind, season, players, records=resolved_records)
    return players, source_tier


def _fetch_batting_split_raw(
    season: int,
    use_cache: bool,
    *,
    kind: str,
    split_type: str,
) -> dict[str, dict]:
    players, _ = _fetch_batting_split_raw_with_status(
        season,
        use_cache,
        kind=kind,
        split_type=split_type,
    )
    return players


def _fetch_batting_split_raw_with_status(
    season: int,
    use_cache: bool,
    *,
    kind: str,
    split_type: str,
) -> tuple[dict[str, dict], str]:
    if use_cache:
        cached = _load_raw_cache(kind, season)
        if cached is not None:
            return _enrich_batter_handedness(cached, season, kind), "cache"

    sit_code = _BATTING_SPLIT_SIT_CODES.get(split_type)
    if sit_code is None:
        raise ValueError(f"Unsupported batting split type: {split_type}")
    records = fetch_batting_split_rows(season, sit_code=sit_code)
    players = _build_batting_players_from_records(records, season, split_type=split_type)

    players = _enrich_batter_handedness(players, season, kind)
    _save_raw_cache(kind, season, players, records=records)
    return players, "fresh"


def _fetch_pitching_split_raw(
    season: int,
    use_cache: bool,
    *,
    kind: str,
    split_type: str,
) -> dict[str, dict]:
    players, _ = _fetch_pitching_split_raw_with_status(
        season,
        use_cache,
        kind=kind,
        split_type=split_type,
    )
    return players


def _fetch_pitching_split_raw_with_status(
    season: int,
    use_cache: bool,
    *,
    kind: str,
    split_type: str,
) -> tuple[dict[str, dict], str]:
    if use_cache:
        cached = _load_raw_cache(kind, season)
        if cached is not None:
            return _enrich_pitcher_handedness(cached, season, kind), "cache"

    sit_code = _PITCHING_SPLIT_SIT_CODES.get(split_type)
    if sit_code is None:
        raise ValueError(f"Unsupported pitching split type: {split_type}")
    records = fetch_pitching_split_rows(season, sit_code=sit_code)
    players = _build_pitching_players_from_records(records, season, split_type=split_type)

    players = _enrich_pitcher_handedness(players, season, kind)
    _save_raw_cache(kind, season, players, records=records)
    return players, "fresh"


def _status_entry(
    source_name: str,
    *,
    role: str,
    status: str,
    detail: str,
) -> DataSourceStatus:
    return DataSourceStatus(
        source_name=source_name,
        role=role,
        scope="run_wide",
        status=status,
        detail=detail,
    )


def _source_tier_detail(label: str, target_season: int, source_tier: str) -> str:
    source_text = "loaded from cache" if source_tier == "cache" else "fetched fresh"
    return f"{label} {target_season}: {source_text}"


def _load_overall_seasons(
    season: int,
    fetch_season_raw,
    use_cache: bool,
    *,
    label: str,
    source_prefix: str,
    source_statuses: list[DataSourceStatus] | None = None,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    """Fetch overall seasons independently so one failure does not abort Marcel."""
    results: dict[int, dict[str, dict]] = {}
    seasons = (season, season - 1, season - 2)
    for target_season in seasons:
        source_name = f"{source_prefix}_{target_season}"
        try:
            players, source_tier = fetch_season_raw(target_season, use_cache)
        except Exception as exc:
            logger.warning(
                "%s overall fetch failed for %d (%s); season excluded from Marcel",
                label,
                target_season,
                exc,
            )
            results[target_season] = {}
            if source_statuses is not None:
                source_statuses.append(
                    _status_entry(
                        source_name,
                        role="required",
                        status="degraded",
                        detail=f"{label} overall {target_season}: fetch failed; season excluded from Marcel",
                    )
                )
            continue

        results[target_season] = players
        if source_statuses is not None:
            if players:
                source_statuses.append(
                    _status_entry(
                        source_name,
                        role="required",
                        status=source_tier,
                        detail=_source_tier_detail(f"{label} overall", target_season, source_tier),
                    )
                )
            else:
                source_statuses.append(
                    _status_entry(
                        source_name,
                        role="required",
                        status="degraded",
                        detail=f"{label} overall {target_season}: no usable rows; season excluded from Marcel",
                    )
                )

    if not any(results[target_season] for target_season in seasons):
        joined = ", ".join(str(target_season) for target_season in seasons)
        raise RuntimeError(f"No usable {label.lower()} overall stats were available for seasons {joined}")

    return results[season], results[season - 1], results[season - 2]


def _load_split_seasons(
    split_fetches: list[tuple[int, str, str]],
    fetch_split_raw,
    use_cache: bool,
    *,
    label: str,
    source_prefix: str,
    source_statuses: list[DataSourceStatus] | None = None,
) -> dict[tuple[int, str], dict[str, dict]]:
    """Fetch split seasons independently so one 403 does not discard all splits."""
    split_results: dict[tuple[int, str], dict[str, dict]] = {}
    for target_season, kind, split_type in split_fetches:
        source_name = f"{source_prefix}_{split_type}_{target_season}"
        try:
            players, source_tier = fetch_split_raw(
                target_season,
                use_cache,
                kind=kind,
                split_type=split_type,
            )
        except Exception as exc:
            logger.warning("%s split fetch failed for %d %s (%s); split excluded from Marcel", label, target_season, split_type, exc)
            split_results[(target_season, split_type)] = {}
            if source_statuses is not None:
                readable_split = split_type.replace("_", " ").upper()
                source_statuses.append(
                    _status_entry(
                        source_name,
                        role="optional_enrichment",
                        status="degraded",
                        detail=f"{label} split {readable_split} {target_season}: fetch failed; split excluded from Marcel",
                    )
                )
            continue

        split_results[(target_season, split_type)] = players
        if source_statuses is not None:
            if players:
                readable_split = split_type.replace("_", " ").upper()
                source_statuses.append(
                    _status_entry(
                        source_name,
                        role="optional_enrichment",
                        status=source_tier,
                        detail=_source_tier_detail(f"{label} split {readable_split}", target_season, source_tier),
                    )
                )
            else:
                readable_split = split_type.replace("_", " ").upper()
                source_statuses.append(
                    _status_entry(
                        source_name,
                        role="optional_enrichment",
                        status="degraded",
                        detail=f"{label} split {readable_split} {target_season}: no usable rows; split excluded from Marcel",
                    )
                )
    return split_results


def fetch_batting_splits_with_statuses(
    season: int = SEASON,
    use_cache: bool = True,
) -> tuple[dict[str, dict], list[DataSourceStatus]]:
    """Fetch batting stats plus run-wide source statuses."""
    source_statuses: list[DataSourceStatus] = []
    s1_players, s2_players, s3_players = _load_overall_seasons(
        season,
        lambda target_season, cache_enabled: _fetch_batting_season_raw_with_status(target_season, cache_enabled),
        use_cache,
        label="Batting",
        source_prefix="batting_overall",
        source_statuses=source_statuses,
    )

    # Runtime helper already degrades through cache and hardcoded fallback.
    overall_lg = fetch_runtime_overall_league_averages(season=season, use_cache=use_cache)

    split_results = _load_split_seasons(
        [
            (season, "batting_vs_lhp", "vs_lhp"),
            (season - 1, "batting_vs_lhp", "vs_lhp"),
            (season - 2, "batting_vs_lhp", "vs_lhp"),
            (season, "batting_vs_rhp", "vs_rhp"),
            (season - 1, "batting_vs_rhp", "vs_rhp"),
            (season - 2, "batting_vs_rhp", "vs_rhp"),
        ],
        _fetch_batting_split_raw_with_status,
        use_cache,
        label="Batting",
        source_prefix="batting_split",
        source_statuses=source_statuses,
    )

    s1_vs_lhp = split_results[(season, "vs_lhp")]
    s2_vs_lhp = split_results[(season - 1, "vs_lhp")]
    s3_vs_lhp = split_results[(season - 2, "vs_lhp")]
    s1_vs_rhp = split_results[(season, "vs_rhp")]
    s2_vs_rhp = split_results[(season - 1, "vs_rhp")]
    s3_vs_rhp = split_results[(season - 2, "vs_rhp")]

    all_names = (
        set(s1_players) | set(s2_players) | set(s3_players)
        | set(s1_vs_lhp) | set(s2_vs_lhp) | set(s3_vs_lhp)
        | set(s1_vs_rhp) | set(s2_vs_rhp) | set(s3_vs_rhp)
    )
    players: dict[str, dict] = {}
    for name in all_names:
        bats_hand = _hand_from_value(
            (s1_players.get(name) or s2_players.get(name) or s3_players.get(name) or {}).get("bats"),
            allow_switch=True,
        )
        player = _marcel_batter_player(
            s1_players.get(name), s2_players.get(name), s3_players.get(name),
            s1_vs_lhp.get(name), s2_vs_lhp.get(name), s3_vs_lhp.get(name),
            s1_vs_rhp.get(name), s2_vs_rhp.get(name), s3_vs_rhp.get(name),
            bats_hand,
            overall_lg,
        )
        if player is not None:
            players[name] = player
    return players, source_statuses


def fetch_batting_splits(season: int = SEASON, use_cache: bool = True) -> dict[str, dict]:
    """Fetch batting stats for three seasons and merge via Marcel projection."""
    players, _ = fetch_batting_splits_with_statuses(season=season, use_cache=use_cache)
    return players


def fetch_pitching_splits_with_statuses(
    season: int = SEASON,
    use_cache: bool = True,
) -> tuple[dict[str, dict], list[DataSourceStatus]]:
    """Fetch pitching stats plus run-wide source statuses."""
    source_statuses: list[DataSourceStatus] = []
    s1_players, s2_players, s3_players = _load_overall_seasons(
        season,
        lambda target_season, cache_enabled: _fetch_pitching_season_raw_with_status(target_season, cache_enabled),
        use_cache,
        label="Pitching",
        source_prefix="pitching_overall",
        source_statuses=source_statuses,
    )

    # Runtime helper already degrades through cache and hardcoded fallback.
    overall_lg = fetch_runtime_overall_league_averages(season=season, use_cache=use_cache)

    split_results = _load_split_seasons(
        [
            (season, "pitching_vs_lhb", "vs_lhb"),
            (season - 1, "pitching_vs_lhb", "vs_lhb"),
            (season - 2, "pitching_vs_lhb", "vs_lhb"),
            (season, "pitching_vs_rhb", "vs_rhb"),
            (season - 1, "pitching_vs_rhb", "vs_rhb"),
            (season - 2, "pitching_vs_rhb", "vs_rhb"),
        ],
        _fetch_pitching_split_raw_with_status,
        use_cache,
        label="Pitching",
        source_prefix="pitching_split",
        source_statuses=source_statuses,
    )

    s1_vs_lhb = split_results[(season, "vs_lhb")]
    s2_vs_lhb = split_results[(season - 1, "vs_lhb")]
    s3_vs_lhb = split_results[(season - 2, "vs_lhb")]
    s1_vs_rhb = split_results[(season, "vs_rhb")]
    s2_vs_rhb = split_results[(season - 1, "vs_rhb")]
    s3_vs_rhb = split_results[(season - 2, "vs_rhb")]

    all_names = (
        set(s1_players) | set(s2_players) | set(s3_players)
        | set(s1_vs_lhb) | set(s2_vs_lhb) | set(s3_vs_lhb)
        | set(s1_vs_rhb) | set(s2_vs_rhb) | set(s3_vs_rhb)
    )
    players: dict[str, dict] = {}
    for name in all_names:
        player = _marcel_pitcher_player(
            s1_players.get(name), s2_players.get(name), s3_players.get(name),
            s1_vs_lhb.get(name), s2_vs_lhb.get(name), s3_vs_lhb.get(name),
            s1_vs_rhb.get(name), s2_vs_rhb.get(name), s3_vs_rhb.get(name),
            overall_lg,
        )
        if player is not None:
            players[name] = player
    return players, source_statuses


def fetch_pitching_splits(season: int = SEASON, use_cache: bool = True) -> dict[str, dict]:
    """Fetch pitching stats for three seasons and merge via Marcel projection."""
    players, _ = fetch_pitching_splits_with_statuses(season=season, use_cache=use_cache)
    return players


def compute_matchup_league_averages(batting_players: dict[str, dict]) -> dict[tuple[Hand, Hand], dict[str, float]]:
    """Aggregate handedness matchup baselines from batter split profiles."""
    totals: dict[tuple[Hand, Hand], dict[str, float]] = {
        (Hand.LEFT, Hand.RIGHT): {"PA": 0.0},
        (Hand.RIGHT, Hand.LEFT): {"PA": 0.0},
        (Hand.LEFT, Hand.LEFT): {"PA": 0.0},
        (Hand.RIGHT, Hand.RIGHT): {"PA": 0.0},
    }
    tracked_outcomes = ("K", "BB", "HBP", "1B", "2B", "3B", "HR")

    for player in batting_players.values():
        bats = _hand_from_value(player.get("bats"), allow_switch=True)
        splits = player.get("splits") or {}
        for pitcher_hand, split_key in ((Hand.LEFT, "vs_lhp"), (Hand.RIGHT, "vs_rhp")):
            split = splits.get(split_key)
            if split is None:
                continue
            pa = _safe_float(split.get("pa"))
            rates = split.get("rates") or {}
            if pa <= 0 or not rates:
                continue
            effective_bats = _effective_batter_hand_for_pitcher(bats, pitcher_hand)
            matchup_key = (effective_bats, pitcher_hand)
            matchup_totals = totals[matchup_key]
            matchup_totals["PA"] += pa
            for outcome in tracked_outcomes:
                matchup_totals[outcome] = matchup_totals.get(outcome, 0.0) + (pa * _safe_float(rates.get(outcome)))

    matchup_rates: dict[tuple[Hand, Hand], dict[str, float]] = {}
    for matchup_key, matchup_totals in totals.items():
        total_pa = matchup_totals.get("PA", 0.0)
        if total_pa <= 0:
            raise ValueError(f"Unable to compute matchup league average for {matchup_key}: no split PA")
        rates = {
            outcome: matchup_totals.get(outcome, 0.0) / total_pa
            for outcome in tracked_outcomes
        }
        matchup_rates[matchup_key] = _normalize_rates(dict(rates, OUT=0.0))
    return matchup_rates


def compute_matchup_league_averages_from_raw_splits(
    overall_batters: dict[str, dict],
    vs_lhp_batters: dict[str, dict],
    vs_rhp_batters: dict[str, dict],
) -> dict[tuple[Hand, Hand], dict[str, float]]:
    split_profiles: dict[str, dict] = {}
    for name, overall in overall_batters.items():
        profile = dict(overall)
        profile["splits"] = {}
        if name in vs_lhp_batters:
            profile["splits"]["vs_lhp"] = {
                "pa": _safe_int(vs_lhp_batters[name].get("pa")),
                "rates": dict(vs_lhp_batters[name].get("rates") or {}),
            }
        if name in vs_rhp_batters:
            profile["splits"]["vs_rhp"] = {
                "pa": _safe_int(vs_rhp_batters[name].get("pa")),
                "rates": dict(vs_rhp_batters[name].get("rates") or {}),
            }
        split_profiles[name] = profile
    return compute_matchup_league_averages(split_profiles)


def fetch_computed_league_averages(season: int = SEASON, use_cache: bool = True) -> dict[str, float]:
    """Strict helper: compute overall league averages from leaderboard data.

    This function is intentionally strict. Runtime simulation paths should use
    `fetch_runtime_overall_league_averages()` or `fetch_runtime_league_averages()`
    so fresh-fetch failures can degrade through cache or hardcoded fallbacks.
    """
    cache_key = (season, use_cache)
    cached = _COMPUTED_LEAGUE_AVERAGE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if use_cache:
        payload = _load_computed_cache("league_averages", season)
        if payload is not None and "rates" in payload:
            rates = dict(payload["rates"])
            matchup_rates = payload.get("matchup_rates")
            if isinstance(matchup_rates, dict):
                _MATCHUP_LEAGUE_AVERAGE_CACHE[cache_key] = _deserialize_matchup_rates(matchup_rates)
            _COMPUTED_LEAGUE_AVERAGE_CACHE[cache_key] = rates
            return rates

    batting_df, _ = _fetch_batting_season_records_with_status(season, use_cache, kind="batting")
    pitching_df, _ = _fetch_pitching_season_records_with_status(season, use_cache, kind="pitching")
    rates = compute_league_averages(batting_df, pitching_df)
    overall_batters, _ = _fetch_batting_season_raw_with_status(season, use_cache, records=batting_df)
    vs_lhp_batters = _fetch_batting_split_raw(
        season,
        use_cache,
        kind="batting_vs_lhp",
        split_type="vs_lhp",
    )
    vs_rhp_batters = _fetch_batting_split_raw(
        season,
        use_cache,
        kind="batting_vs_rhp",
        split_type="vs_rhp",
    )
    matchup_rates = compute_matchup_league_averages_from_raw_splits(
        overall_batters,
        vs_lhp_batters,
        vs_rhp_batters,
    )
    _save_computed_cache(
        "league_averages",
        season,
        {
            "season": season,
            "rates": rates,
            "matchup_rates": _serialize_matchup_rates(matchup_rates),
        },
    )
    _MATCHUP_LEAGUE_AVERAGE_CACHE[cache_key] = matchup_rates
    _COMPUTED_LEAGUE_AVERAGE_CACHE[cache_key] = rates
    return rates


def _compute_league_averages_and_matchups(
    season: int,
    use_cache: bool,
) -> tuple[dict[str, float], dict[tuple[Hand, Hand], dict[str, float]]]:
    """Strict helper that computes both overall and matchup league averages."""
    rates = fetch_computed_league_averages(season=season, use_cache=use_cache)
    cache_key = (season, use_cache)
    matchup_rates = _MATCHUP_LEAGUE_AVERAGE_CACHE.get(cache_key)
    if matchup_rates is None:
        raise KeyError(f"Missing matchup league averages for season={season} use_cache={use_cache}")
    return dict(rates), _copy_league_averages(matchup_rates)


def fetch_runtime_overall_league_averages(
    season: int = SEASON,
    use_cache: bool = True,
) -> dict[str, float]:
    """Fetch overall league averages with runtime fallbacks.

    Uses valid cache -> fresh strict fetch -> stale cache -> hardcoded overall
    proxy derived from the matchup fallback constants.
    """
    cache_key = (season, use_cache)
    cached = _COMPUTED_LEAGUE_AVERAGE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    hardcoded = _overall_league_avg_rates()

    if use_cache:
        payload = _load_computed_cache("league_averages", season)
        if payload is not None and "rates" in payload:
            rates = dict(payload["rates"])
            matchup_rates = payload.get("matchup_rates")
            if isinstance(matchup_rates, dict):
                _MATCHUP_LEAGUE_AVERAGE_CACHE[cache_key] = _deserialize_matchup_rates(matchup_rates)
            _COMPUTED_LEAGUE_AVERAGE_CACHE[cache_key] = rates
            return dict(rates)

    try:
        rates = fetch_computed_league_averages(season=season, use_cache=False)
        _COMPUTED_LEAGUE_AVERAGE_CACHE[cache_key] = dict(rates)
        return dict(rates)
    except Exception as exc:
        if use_cache:
            stale_path = _computed_cache_path("league_averages", season)
            if stale_path.exists():
                with open(stale_path) as f:
                    payload = json.load(f)
                if "rates" in payload:
                    rates = dict(payload["rates"])
                    matchup_rates = payload.get("matchup_rates")
                    if isinstance(matchup_rates, dict):
                        _MATCHUP_LEAGUE_AVERAGE_CACHE[cache_key] = _deserialize_matchup_rates(matchup_rates)
                    logger.warning(
                        "Overall league averages fetch failed for %d (%s); using cached computed league averages",
                        season,
                        exc,
                    )
                    _COMPUTED_LEAGUE_AVERAGE_CACHE[cache_key] = rates
                    return dict(rates)

        logger.warning(
            "Overall league averages fetch failed for %d (%s); using hardcoded fallback constants",
            season,
            exc,
        )
        return hardcoded


def fetch_runtime_league_averages(
    season: int = SEASON,
    use_cache: bool = True,
) -> dict[tuple[Hand, Hand], dict[str, float]]:
    """Fetch matchup league averages with cache->network->hardcoded fallback."""
    cache_key = (season, use_cache)
    cached = _MATCHUP_LEAGUE_AVERAGE_CACHE.get(cache_key)
    if cached is not None:
        return _copy_league_averages(cached)

    hardcoded = _copy_league_averages(_HARDCODED_LEAGUE_AVERAGES)

    if use_cache:
        payload = _load_computed_cache("league_averages", season)
        if payload is not None and "matchup_rates" in payload:
            matchup_rates = _deserialize_matchup_rates(dict(payload["matchup_rates"]))
            _MATCHUP_LEAGUE_AVERAGE_CACHE[cache_key] = matchup_rates
            _COMPUTED_LEAGUE_AVERAGE_CACHE[cache_key] = dict(payload.get("rates") or {})
            return _copy_league_averages(matchup_rates)

    try:
        fresh_rates, fresh_matchups = _compute_league_averages_and_matchups(season=season, use_cache=False)
        _MATCHUP_LEAGUE_AVERAGE_CACHE[cache_key] = _copy_league_averages(fresh_matchups)
        _COMPUTED_LEAGUE_AVERAGE_CACHE[cache_key] = dict(fresh_rates)
        return _copy_league_averages(fresh_matchups)
    except Exception as exc:
        if use_cache:
            stale_payload = _computed_cache_path("league_averages", season)
            if stale_payload.exists():
                with open(stale_payload) as f:
                    payload = json.load(f)
                matchup_rates = payload.get("matchup_rates")
                if isinstance(matchup_rates, dict):
                    logger.warning(
                        "League averages fetch failed for %d (%s); using cached computed league averages",
                        season,
                        exc,
                    )
                    parsed = _deserialize_matchup_rates(matchup_rates)
                    _MATCHUP_LEAGUE_AVERAGE_CACHE[cache_key] = parsed
                    _COMPUTED_LEAGUE_AVERAGE_CACHE[cache_key] = dict(payload.get("rates") or {})
                    return _copy_league_averages(parsed)

        logger.warning(
            "League averages fetch failed for %d (%s); using hardcoded fallback constants",
            season,
            exc,
        )
        return hardcoded


def ensure_runtime_league_averages(season: int = SEASON, use_cache: bool = True) -> dict[tuple[Hand, Hand], dict[str, float]]:
    """Refresh the global league-average mapping in place right before simulation context build."""
    matchup_rates = fetch_runtime_league_averages(season=season, use_cache=use_cache)
    _apply_runtime_league_averages(matchup_rates)
    return matchup_rates


def fetch_team_bullpen_stats(season: int = SEASON, use_cache: bool = True) -> dict[str, dict[str, Any]]:
    """Fetch team-level bullpen aggregate rates from reliever rows only."""
    cache_key = (season, use_cache)
    cached = _TEAM_BULLPEN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    records, _ = _fetch_pitching_season_records_with_status(season, use_cache, kind="pitching")
    # Warm the shared overall-pitching raw cache so bullpen aggregation and
    # other overall consumers reuse the same season payload instead of refetching.
    _fetch_pitching_season_raw_with_status(season, use_cache, records=records)
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
    pitcher_throws = _hand_from_value(pitcher_hand) if pitcher_hand is not None else Hand.RIGHT
    default_rates = dict(
        player_data.get("rates")
        or overall.get("rates")
        or _league_average_rates_for_batter(_effective_batter_hand_for_pitcher(bats, pitcher_throws), pitcher_throws)
    )
    default_pa = _safe_int(player_data.get("pa") or overall.get("pa"), 0)
    default_source = str(player_data.get("source") or overall.get("source") or "unknown")
    if use_overall or pitcher_hand is None:
        return default_rates, default_pa, default_source

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
    effective_batter_hand = _effective_batter_hand_for_pitcher(
        batter_hand if batter_hand is not None else Hand.RIGHT,
        throws,
    )
    default_rates = dict(
        player_data.get("rates")
        or overall.get("rates")
        or _league_average_rates_for_pitcher(throws, effective_batter_hand)
    )
    default_pa = _safe_int(player_data.get("pa_against") or overall.get("pa_against"), 0)
    default_source = str(player_data.get("source") or overall.get("source") or "unknown")
    if use_overall or batter_hand is None:
        return default_rates, default_pa, default_source

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
