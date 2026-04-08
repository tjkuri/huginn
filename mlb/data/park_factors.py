"""Park-factor lookup helpers for MLB simulation contexts."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from mlb.config import CACHE_DIR, SEASON
from mlb.data.models import DataSourceStatus, ParkFactors

logger = logging.getLogger(__name__)

# 2025 fallback park factors. Values are from the 2025 Savant cache snapshot
# where available; the temporary/unused venues called out below remain explicit
# placeholders until a full-season Savant value is available for them.
PARK_FACTORS = {
    "American Family Field": {"factors_vs_lhb": {"HR": 1.0, "2B": 0.78, "3B": 1.05, "1B": 0.95, "BB": 1.08, "K": 1.13}, "factors_vs_rhb": {"HR": 1.1, "2B": 0.93, "3B": 0.7, "1B": 0.93, "BB": 1.05, "K": 1.07}},
    "Angel Stadium": {"factors_vs_lhb": {"HR": 1.08, "2B": 0.86, "3B": 0.99, "1B": 0.95, "BB": 1.03, "K": 1.03}, "factors_vs_rhb": {"HR": 1.16, "2B": 0.92, "3B": 0.96, "1B": 0.99, "BB": 1.04, "K": 1.07}},
    "Busch Stadium": {"factors_vs_lhb": {"HR": 0.89, "2B": 1.0, "3B": 0.79, "1B": 1.07, "BB": 0.95, "K": 0.92}, "factors_vs_rhb": {"HR": 0.85, "2B": 1.08, "3B": 0.83, "1B": 1.07, "BB": 0.97, "K": 0.91}},
    "Chase Field": {"factors_vs_lhb": {"HR": 0.77, "2B": 1.1, "3B": 2.22, "1B": 1.03, "BB": 0.98, "K": 0.95}, "factors_vs_rhb": {"HR": 0.97, "2B": 1.18, "3B": 1.77, "1B": 1.03, "BB": 1.0, "K": 0.93}},
    "Citi Field": {"factors_vs_lhb": {"HR": 0.98, "2B": 0.93, "3B": 0.63, "1B": 0.93, "BB": 1.1, "K": 1.03}, "factors_vs_rhb": {"HR": 1.09, "2B": 0.87, "3B": 0.83, "1B": 0.93, "BB": 1.1, "K": 1.02}},
    "Citizens Bank Park": {"factors_vs_lhb": {"HR": 1.28, "2B": 0.93, "3B": 1.02, "1B": 0.97, "BB": 1.04, "K": 1.05}, "factors_vs_rhb": {"HR": 1.03, "2B": 0.99, "3B": 0.98, "1B": 1.0, "BB": 0.89, "K": 1.03}},
    "Comerica Park": {"factors_vs_lhb": {"HR": 0.99, "2B": 0.91, "3B": 2.0, "1B": 1.01, "BB": 0.94, "K": 1.02}, "factors_vs_rhb": {"HR": 1.0, "2B": 0.97, "3B": 0.82, "1B": 0.99, "BB": 1.06, "K": 0.96}},
    "Coors Field": {"factors_vs_lhb": {"HR": 1.05, "2B": 1.17, "3B": 1.73, "1B": 1.15, "BB": 1.1, "K": 0.92}, "factors_vs_rhb": {"HR": 1.06, "2B": 1.21, "3B": 2.34, "1B": 1.17, "BB": 0.93, "K": 0.9}},
    "Daikin Park": {"factors_vs_lhb": {"HR": 1.1, "2B": 0.96, "3B": 1.06, "1B": 0.98, "BB": 1.06, "K": 1.05}, "factors_vs_rhb": {"HR": 1.03, "2B": 0.96, "3B": 0.77, "1B": 1.01, "BB": 0.96, "K": 1.01}},
    "Dodger Stadium": {"factors_vs_lhb": {"HR": 1.19, "2B": 0.95, "3B": 0.62, "1B": 0.91, "BB": 0.96, "K": 1.02}, "factors_vs_rhb": {"HR": 1.35, "2B": 0.96, "3B": 0.67, "1B": 0.93, "BB": 1.06, "K": 0.98}},
    "Fenway Park": {"factors_vs_lhb": {"HR": 0.87, "2B": 1.44, "3B": 0.94, "1B": 1.06, "BB": 0.98, "K": 0.89}, "factors_vs_rhb": {"HR": 0.9, "2B": 1.03, "3B": 0.96, "1B": 1.07, "BB": 0.96, "K": 1.04}},
    # Placeholder fallback: 2025 Savant cache snapshot did not include this venue.
    # BB/K remain neutral because they are not data-derived here.
    "George M. Steinbrenner Field": {"factors_vs_lhb": {"HR": 1.03, "2B": 1.01, "3B": 0.95, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.01, "2B": 1.01, "3B": 0.95, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "Globe Life Field": {"factors_vs_lhb": {"HR": 1.02, "2B": 0.99, "3B": 0.65, "1B": 0.98, "BB": 1.0, "K": 0.99}, "factors_vs_rhb": {"HR": 1.05, "2B": 0.93, "3B": 0.83, "1B": 0.96, "BB": 1.0, "K": 1.02}},
    "Great American Ball Park": {"factors_vs_lhb": {"HR": 1.26, "2B": 0.99, "3B": 0.92, "1B": 0.95, "BB": 1.0, "K": 1.07}, "factors_vs_rhb": {"HR": 1.21, "2B": 0.99, "3B": 0.57, "1B": 0.96, "BB": 1.06, "K": 1.0}},
    "Kauffman Stadium": {"factors_vs_lhb": {"HR": 0.73, "2B": 1.14, "3B": 1.81, "1B": 1.02, "BB": 1.03, "K": 0.88}, "factors_vs_rhb": {"HR": 0.93, "2B": 1.12, "3B": 1.86, "1B": 1.04, "BB": 0.97, "K": 0.9}},
    "loanDepot Park": {"factors_vs_lhb": {"HR": 0.98, "2B": 1.04, "3B": 1.19, "1B": 1.09, "BB": 0.96, "K": 0.98}, "factors_vs_rhb": {"HR": 0.84, "2B": 1.09, "3B": 1.17, "1B": 1.02, "BB": 0.98, "K": 0.97}},
    "Nationals Park": {"factors_vs_lhb": {"HR": 0.97, "2B": 1.0, "3B": 0.89, "1B": 1.11, "BB": 0.91, "K": 0.86}, "factors_vs_rhb": {"HR": 0.91, "2B": 0.96, "3B": 1.15, "1B": 1.06, "BB": 0.97, "K": 0.93}},
    "Oracle Park": {"factors_vs_lhb": {"HR": 0.78, "2B": 1.02, "3B": 1.21, "1B": 1.04, "BB": 0.93, "K": 1.0}, "factors_vs_rhb": {"HR": 0.84, "2B": 1.02, "3B": 1.24, "1B": 1.04, "BB": 0.87, "K": 0.96}},
    "Oriole Park at Camden Yards": {"factors_vs_lhb": {"HR": 1.25, "2B": 0.91, "3B": 0.94, "1B": 1.03, "BB": 0.93, "K": 0.97}, "factors_vs_rhb": {"HR": 0.87, "2B": 1.03, "3B": 1.61, "1B": 1.04, "BB": 0.88, "K": 1.01}},
    "PNC Park": {"factors_vs_lhb": {"HR": 0.87, "2B": 1.21, "3B": 0.78, "1B": 0.98, "BB": 1.0, "K": 0.97}, "factors_vs_rhb": {"HR": 0.68, "2B": 1.1, "3B": 0.9, "1B": 1.05, "BB": 1.0, "K": 0.96}},
    "Petco Park": {"factors_vs_lhb": {"HR": 0.9, "2B": 0.99, "3B": 0.57, "1B": 0.97, "BB": 1.04, "K": 0.99}, "factors_vs_rhb": {"HR": 1.13, "2B": 0.87, "3B": 0.74, "1B": 0.96, "BB": 1.03, "K": 1.04}},
    "Progressive Field": {"factors_vs_lhb": {"HR": 0.93, "2B": 1.02, "3B": 0.51, "1B": 0.99, "BB": 0.95, "K": 0.97}, "factors_vs_rhb": {"HR": 0.75, "2B": 1.12, "3B": 1.08, "1B": 0.97, "BB": 1.06, "K": 1.06}},
    "Rate Field": {"factors_vs_lhb": {"HR": 0.96, "2B": 0.88, "3B": 0.78, "1B": 1.02, "BB": 1.09, "K": 0.99}, "factors_vs_rhb": {"HR": 0.96, "2B": 0.94, "3B": 0.62, "1B": 1.0, "BB": 0.99, "K": 1.01}},
    "Rogers Centre": {"factors_vs_lhb": {"HR": 1.02, "2B": 1.07, "3B": 0.78, "1B": 0.91, "BB": 0.99, "K": 1.07}, "factors_vs_rhb": {"HR": 1.05, "2B": 1.02, "3B": 0.62, "1B": 1.01, "BB": 1.01, "K": 0.91}},
    # Placeholder fallback: 2025 Savant cache snapshot did not include this venue.
    # BB/K remain neutral because they are not data-derived here.
    "Sutter Health Park": {"factors_vs_lhb": {"HR": 1.04, "2B": 1.02, "3B": 0.94, "1B": 1.01, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.04, "2B": 1.02, "3B": 0.94, "1B": 1.01, "BB": 1.00, "K": 1.00}},
    "Target Field": {"factors_vs_lhb": {"HR": 1.02, "2B": 1.03, "3B": 0.94, "1B": 1.01, "BB": 1.0, "K": 1.06}, "factors_vs_rhb": {"HR": 1.02, "2B": 1.17, "3B": 1.03, "1B": 0.99, "BB": 1.0, "K": 1.01}},
    "T-Mobile Park": {"factors_vs_lhb": {"HR": 0.95, "2B": 0.95, "3B": 0.48, "1B": 0.93, "BB": 0.93, "K": 1.15}, "factors_vs_rhb": {"HR": 0.91, "2B": 0.85, "3B": 0.57, "1B": 0.87, "BB": 1.0, "K": 1.18}},
    "Truist Park": {"factors_vs_lhb": {"HR": 1.09, "2B": 0.89, "3B": 0.82, "1B": 1.06, "BB": 0.93, "K": 1.05}, "factors_vs_rhb": {"HR": 1.01, "2B": 0.98, "3B": 1.01, "1B": 1.02, "BB": 1.04, "K": 1.06}},
    # Placeholder fallback: Tropicana Field was not a 2025 MLB home venue.
    # BB/K remain neutral because they are not data-derived here.
    "Tropicana Field": {"factors_vs_lhb": {"HR": 0.91, "2B": 0.98, "3B": 0.90, "1B": 0.99, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 0.92, "2B": 0.98, "3B": 0.90, "1B": 0.99, "BB": 1.00, "K": 1.00}},
    "Wrigley Field": {"factors_vs_lhb": {"HR": 0.95, "2B": 0.91, "3B": 1.37, "1B": 1.0, "BB": 0.99, "K": 1.07}, "factors_vs_rhb": {"HR": 1.02, "2B": 0.83, "3B": 0.95, "1B": 0.97, "BB": 1.0, "K": 1.0}},
    "Yankee Stadium": {"factors_vs_lhb": {"HR": 1.18, "2B": 0.91, "3B": 0.57, "1B": 0.91, "BB": 1.13, "K": 0.98}, "factors_vs_rhb": {"HR": 1.19, "2B": 0.89, "3B": 0.69, "1B": 0.91, "BB": 1.11, "K": 1.05}},
}

TEAM_TO_VENUE = {
    "Arizona Diamondbacks": "Chase Field",
    "Athletics": "Sutter Health Park",
    "Atlanta Braves": "Truist Park",
    "Baltimore Orioles": "Oriole Park at Camden Yards",
    "Boston Red Sox": "Fenway Park",
    "Chicago Cubs": "Wrigley Field",
    "Chicago White Sox": "Rate Field",
    "Cincinnati Reds": "Great American Ball Park",
    "Cleveland Guardians": "Progressive Field",
    "Colorado Rockies": "Coors Field",
    "Detroit Tigers": "Comerica Park",
    "Houston Astros": "Daikin Park",
    "Kansas City Royals": "Kauffman Stadium",
    "Los Angeles Angels": "Angel Stadium",
    "Los Angeles Dodgers": "Dodger Stadium",
    "Miami Marlins": "loanDepot Park",
    "Milwaukee Brewers": "American Family Field",
    "Minnesota Twins": "Target Field",
    "New York Mets": "Citi Field",
    "New York Yankees": "Yankee Stadium",
    "Philadelphia Phillies": "Citizens Bank Park",
    "Pittsburgh Pirates": "PNC Park",
    "San Diego Padres": "Petco Park",
    "San Francisco Giants": "Oracle Park",
    "Seattle Mariners": "T-Mobile Park",
    "St. Louis Cardinals": "Busch Stadium",
    "Tampa Bay Rays": "Tropicana Field",
    "Texas Rangers": "Globe Life Field",
    "Toronto Blue Jays": "Rogers Centre",
    "Washington Nationals": "Nationals Park",
}

_NEUTRAL_FACTORS = {"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0}
_VENUE_ALIASES = {
    "UNIQLO Field at Dodger Stadium": "Dodger Stadium",
    "loanDepot park": "loanDepot Park",
}
# Base URL returns combined ("All") factors; batSide=R/L returns handedness-specific factors.
_SAVANT_URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=year&year={season}"
_SAVANT_URL_SIDE = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=year&year={season}&batSide={bat_side}"
_SAVANT_DATA_RE = re.compile(r"var data = (\[.*?\]);", re.DOTALL)
_PARK_FACTOR_CACHE: dict[int, tuple[str, dict[str, dict[str, dict[str, float]]]]] = {}


def _park_factor_cache_path(season: int) -> Path:
    return CACHE_DIR / f"park_factors-{season}.json"


def _convert_savant_index_to_multiplier(value: Any) -> float:
    return float(value) / 100.0


def _fallback_park_factors() -> dict[str, dict[str, dict[str, float]]]:
    return {
        venue: {
            "factors_vs_lhb": dict(factors["factors_vs_lhb"]),
            "factors_vs_rhb": dict(factors["factors_vs_rhb"]),
        }
        for venue, factors in PARK_FACTORS.items()
    }


def _extract_data_rows(html: str) -> list[dict]:
    """Extract the embedded `var data = [...]` array from a Savant HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    match = _SAVANT_DATA_RE.search(soup.decode())
    if not match:
        raise ValueError("Could not locate embedded park factor data in Savant response")
    return json.loads(match.group(1))


def _rows_to_venue_factors(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Convert parsed data rows to a flat venue→factors mapping.

    Accepts all rows regardless of key_bat_side (caller already filtered by URL).
    When multiple rows share a venue, the last one wins.
    """
    factors: dict[str, dict[str, float]] = {}
    for row in rows:
        venue_name = _VENUE_ALIASES.get(str(row.get("venue_name") or "").strip(), str(row.get("venue_name") or "").strip())
        if not venue_name:
            continue
        factors[venue_name] = {
            "HR": _convert_savant_index_to_multiplier(row.get("index_hr", 100)),
            "2B": _convert_savant_index_to_multiplier(row.get("index_2b", 100)),
            "3B": _convert_savant_index_to_multiplier(row.get("index_3b", 100)),
            "1B": _convert_savant_index_to_multiplier(row.get("index_1b", 100)),
            "BB": _convert_savant_index_to_multiplier(row.get("index_bb", 100)),
            "K": _convert_savant_index_to_multiplier(row.get("index_so", 100)),
        }
    if not factors:
        raise ValueError("Parsed Savant response but found no park-factor rows")
    return factors


def _fetch_savant_side(season: int, bat_side: str) -> dict[str, dict[str, float]]:
    """Fetch per-venue factors for one bat side (R or L) from Savant.

    Raises on any network or parse failure.
    """
    url = _SAVANT_URL_SIDE.format(season=season, bat_side=bat_side)
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    rows = _extract_data_rows(response.text)
    # The side-specific page may include rows for all bat sides in the embedded data;
    # prefer rows that match the requested side, fall back to "All" rows.
    target_rows = [r for r in rows if str(r.get("key_bat_side") or "") == bat_side]
    if not target_rows:
        # Savant may label them differently; accept whatever rows are present.
        target_rows = rows
    return _rows_to_venue_factors(target_rows)


def _fetch_savant_combined(season: int) -> dict[str, dict[str, float]]:
    """Fetch combined ("All" bat side) factors from Savant as a fallback."""
    url = _SAVANT_URL.format(season=season)
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    rows = _extract_data_rows(response.text)
    all_rows = [r for r in rows if str(r.get("key_bat_side", "All")) == "All"]
    if not all_rows:
        all_rows = rows
    return _rows_to_venue_factors(all_rows)


def fetch_park_factors_with_source(season: int) -> tuple[str, dict[str, dict[str, dict[str, float]]]]:
    """Fetch season park factors from Baseball Savant with handedness splits.

    Makes separate requests for LHB (batSide=L) and RHB (batSide=R). If one
    side fails, falls back to the combined ("All") data for that side. If all
    Savant requests fail, falls back to the hardcoded PARK_FACTORS table.
    Results are cached per season in baseball_cache/park_factors-{season}.json.
    """
    cached = _PARK_FACTOR_CACHE.get(season)
    if cached is not None:
        return cached

    cache_path = _park_factor_cache_path(season)
    if cache_path.exists():
        with open(cache_path) as f:
            cached_payload = json.load(f)
        factors = cached_payload.get("factors") or {}
        if factors:
            source = str(cached_payload.get("source") or "cache")
            if source == "savant":
                logger.info("Park factors: loaded from cache (Baseball Savant %s)", season)
                cache_source = "cache"
            else:
                logger.info("Park factors: loaded from cache (hardcoded fallback)")
                cache_source = "hardcoded_fallback"
            _PARK_FACTOR_CACHE[season] = (cache_source, factors)
            return cache_source, factors

    try:
        # Attempt handedness-split requests.
        lhb_factors: dict[str, dict[str, float]] | None = None
        rhb_factors: dict[str, dict[str, float]] | None = None
        combined_factors: dict[str, dict[str, float]] | None = None

        try:
            lhb_factors = _fetch_savant_side(season, "L")
        except Exception as exc:
            logger.debug("Park factors: LHB request failed (%s); will use combined", exc)

        try:
            rhb_factors = _fetch_savant_side(season, "R")
        except Exception as exc:
            logger.debug("Park factors: RHB request failed (%s); will use combined", exc)

        if lhb_factors is None or rhb_factors is None:
            try:
                combined_factors = _fetch_savant_combined(season)
            except Exception as exc:
                logger.debug("Park factors: combined request also failed (%s)", exc)
            if lhb_factors is None:
                lhb_factors = combined_factors or rhb_factors
            if rhb_factors is None:
                rhb_factors = combined_factors or lhb_factors

        if lhb_factors is None or rhb_factors is None:
            raise ValueError("All Savant park factor requests failed")

        # Merge: union of all venue names seen across both sides.
        all_venues = set(lhb_factors) | set(rhb_factors)
        factors: dict[str, dict[str, dict[str, float]]] = {}
        for venue in all_venues:
            factors[venue] = {
                "factors_vs_lhb": dict(lhb_factors.get(venue, rhb_factors.get(venue, _NEUTRAL_FACTORS))),
                "factors_vs_rhb": dict(rhb_factors.get(venue, lhb_factors.get(venue, _NEUTRAL_FACTORS))),
            }

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"season": season, "source": "savant", "factors": factors}, f, indent=2)
        logger.info("Park factors: fetched from Baseball Savant (%s) with LHB/RHB splits", season)
        _PARK_FACTOR_CACHE[season] = ("fresh", factors)
        return "fresh", factors
    except Exception:
        logger.info("Park factors: all Savant requests failed; using hardcoded fallback")
        fallback = _fallback_park_factors()
        _PARK_FACTOR_CACHE[season] = ("hardcoded_fallback", fallback)
        return "hardcoded_fallback", fallback


def fetch_park_factors(season: int) -> dict[str, dict[str, dict[str, float]]]:
    """Fetch season park factors and return only the factor mapping."""
    _, factors = fetch_park_factors_with_source(season)
    return factors


def get_park_factors(venue_name: str) -> ParkFactors:
    """Return park-factor multipliers for the given venue."""
    park_factors, _ = get_park_factors_with_status(venue_name)
    return park_factors


def get_park_factors_with_status(venue_name: str) -> tuple[ParkFactors, DataSourceStatus]:
    """Return park factors plus provenance for one venue."""
    venue_name = _VENUE_ALIASES.get(venue_name, venue_name)
    source, all_factors = fetch_park_factors_with_source(SEASON)
    factors = all_factors.get(venue_name)
    status = source
    detail = f"Park factors resolved for {venue_name} from preloaded season park factors"
    if factors is None:
        # Savant data may be incomplete early in the season — fall back to hardcoded table.
        hardcoded = PARK_FACTORS.get(venue_name)
        if hardcoded is not None:
            logger.debug("Park factors: %r not in Savant data; using hardcoded fallback", venue_name)
            factors = {
                "factors_vs_lhb": dict(hardcoded["factors_vs_lhb"]),
                "factors_vs_rhb": dict(hardcoded["factors_vs_rhb"]),
            }
            status = "hardcoded_fallback"
            detail = f"Park factors for {venue_name} fell back to the hardcoded venue table"
        else:
            logger.warning("Park factors: unknown venue %r; using neutral factors", venue_name)
            factors = {
                "factors_vs_lhb": dict(_NEUTRAL_FACTORS),
                "factors_vs_rhb": dict(_NEUTRAL_FACTORS),
            }
            status = "degraded"
            detail = f"Park factors for {venue_name} used neutral multipliers"

    return ParkFactors(
        venue_id=venue_name or "unknown",
        venue_name=venue_name or "Unknown Venue",
        factors_vs_lhb=dict(factors["factors_vs_lhb"]),
        factors_vs_rhb=dict(factors["factors_vs_rhb"]),
    ), DataSourceStatus(
        source_name="park_factors",
        role="optional_enrichment",
        scope="run_wide",
        status=status,
        detail=detail,
    )


def get_venue_for_team(team_name: str) -> str:
    """Map a team name to its home venue name."""
    return TEAM_TO_VENUE.get(team_name, "")
