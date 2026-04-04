"""Park-factor lookup helpers for MLB simulation contexts."""
from __future__ import annotations

from mlb.data.models import ParkFactors

# Approximate 2025 park factors synthesized from FanGraphs Guts and Baseball
# Savant park-factor references. These are deliberately coarse v1 multipliers
# and should be refreshed annually as more stable season-level data arrives.
PARK_FACTORS = {
    "Angel Stadium": {"factors_vs_lhb": {"HR": 1.02, "2B": 1.00, "3B": 0.92, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.01, "2B": 1.01, "3B": 0.92, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "American Family Field": {"factors_vs_lhb": {"HR": 1.03, "2B": 1.00, "3B": 0.90, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.04, "2B": 1.00, "3B": 0.90, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "Busch Stadium": {"factors_vs_lhb": {"HR": 0.95, "2B": 0.98, "3B": 0.92, "1B": 0.99, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 0.94, "2B": 0.98, "3B": 0.92, "1B": 0.99, "BB": 1.00, "K": 1.00}},
    "Chase Field": {"factors_vs_lhb": {"HR": 1.03, "2B": 1.02, "3B": 1.02, "1B": 1.01, "BB": 1.00, "K": 0.99}, "factors_vs_rhb": {"HR": 1.05, "2B": 1.03, "3B": 1.02, "1B": 1.01, "BB": 1.00, "K": 0.99}},
    "Citi Field": {"factors_vs_lhb": {"HR": 0.96, "2B": 1.00, "3B": 0.95, "1B": 0.99, "BB": 1.00, "K": 1.01}, "factors_vs_rhb": {"HR": 0.97, "2B": 1.00, "3B": 0.95, "1B": 0.99, "BB": 1.00, "K": 1.01}},
    "Citizens Bank Park": {"factors_vs_lhb": {"HR": 1.12, "2B": 1.00, "3B": 0.86, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.09, "2B": 1.01, "3B": 0.86, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "Comerica Park": {"factors_vs_lhb": {"HR": 0.94, "2B": 1.06, "3B": 1.18, "1B": 1.01, "BB": 1.00, "K": 0.99}, "factors_vs_rhb": {"HR": 0.92, "2B": 1.05, "3B": 1.16, "1B": 1.01, "BB": 1.00, "K": 0.99}},
    "Coors Field": {"factors_vs_lhb": {"HR": 1.25, "2B": 1.20, "3B": 1.40, "1B": 1.10, "BB": 1.00, "K": 0.95}, "factors_vs_rhb": {"HR": 1.30, "2B": 1.18, "3B": 1.35, "1B": 1.08, "BB": 1.00, "K": 0.94}},
    "Daikin Park": {"factors_vs_lhb": {"HR": 1.00, "2B": 0.99, "3B": 0.94, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.08, "2B": 1.00, "3B": 0.94, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "Dodger Stadium": {"factors_vs_lhb": {"HR": 1.00, "2B": 0.99, "3B": 0.90, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.02, "2B": 1.00, "3B": 0.90, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "Fenway Park": {"factors_vs_lhb": {"HR": 0.98, "2B": 1.14, "3B": 0.85, "1B": 1.05, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.01, "2B": 1.18, "3B": 0.84, "1B": 1.05, "BB": 1.00, "K": 1.00}},
    "George M. Steinbrenner Field": {"factors_vs_lhb": {"HR": 1.03, "2B": 1.01, "3B": 0.95, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.01, "2B": 1.01, "3B": 0.95, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "Globe Life Field": {"factors_vs_lhb": {"HR": 0.98, "2B": 1.00, "3B": 0.91, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 0.99, "2B": 1.00, "3B": 0.91, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "Great American Ball Park": {"factors_vs_lhb": {"HR": 1.14, "2B": 0.98, "3B": 0.84, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.16, "2B": 0.99, "3B": 0.84, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "Guaranteed Rate Field": {"factors_vs_lhb": {"HR": 1.08, "2B": 0.99, "3B": 0.90, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.10, "2B": 1.00, "3B": 0.90, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "Kauffman Stadium": {"factors_vs_lhb": {"HR": 0.90, "2B": 1.07, "3B": 1.18, "1B": 1.02, "BB": 1.00, "K": 0.99}, "factors_vs_rhb": {"HR": 0.89, "2B": 1.06, "3B": 1.16, "1B": 1.02, "BB": 1.00, "K": 0.99}},
    "loanDepot Park": {"factors_vs_lhb": {"HR": 0.92, "2B": 0.98, "3B": 0.92, "1B": 0.99, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 0.94, "2B": 0.99, "3B": 0.92, "1B": 0.99, "BB": 1.00, "K": 1.00}},
    "Nationals Park": {"factors_vs_lhb": {"HR": 1.01, "2B": 1.00, "3B": 0.90, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.02, "2B": 1.00, "3B": 0.90, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "Oriole Park at Camden Yards": {"factors_vs_lhb": {"HR": 0.95, "2B": 1.02, "3B": 0.88, "1B": 1.01, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 0.99, "2B": 1.03, "3B": 0.88, "1B": 1.01, "BB": 1.00, "K": 1.00}},
    "Oracle Park": {"factors_vs_lhb": {"HR": 0.88, "2B": 1.02, "3B": 1.06, "1B": 1.01, "BB": 1.00, "K": 1.01}, "factors_vs_rhb": {"HR": 0.84, "2B": 1.02, "3B": 1.04, "1B": 1.01, "BB": 1.00, "K": 1.01}},
    "Petco Park": {"factors_vs_lhb": {"HR": 0.92, "2B": 1.00, "3B": 0.92, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 0.90, "2B": 1.00, "3B": 0.92, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "PNC Park": {"factors_vs_lhb": {"HR": 0.93, "2B": 1.01, "3B": 1.02, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 0.90, "2B": 1.01, "3B": 1.02, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "Progressive Field": {"factors_vs_lhb": {"HR": 0.99, "2B": 1.01, "3B": 0.90, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.00, "2B": 1.01, "3B": 0.90, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "Rate Field": {"factors_vs_lhb": {"HR": 1.08, "2B": 0.99, "3B": 0.90, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.10, "2B": 1.00, "3B": 0.90, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "Rogers Centre": {"factors_vs_lhb": {"HR": 1.07, "2B": 1.00, "3B": 0.88, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.08, "2B": 1.01, "3B": 0.88, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "Sutter Health Park": {"factors_vs_lhb": {"HR": 1.04, "2B": 1.02, "3B": 0.94, "1B": 1.01, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.04, "2B": 1.02, "3B": 0.94, "1B": 1.01, "BB": 1.00, "K": 1.00}},
    "Target Field": {"factors_vs_lhb": {"HR": 1.00, "2B": 1.00, "3B": 0.92, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.01, "2B": 1.00, "3B": 0.92, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "T-Mobile Park": {"factors_vs_lhb": {"HR": 0.94, "2B": 0.99, "3B": 0.90, "1B": 0.99, "BB": 1.00, "K": 1.01}, "factors_vs_rhb": {"HR": 0.92, "2B": 0.99, "3B": 0.90, "1B": 0.99, "BB": 1.00, "K": 1.01}},
    "Truist Park": {"factors_vs_lhb": {"HR": 1.02, "2B": 1.00, "3B": 0.88, "1B": 1.00, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.03, "2B": 1.00, "3B": 0.88, "1B": 1.00, "BB": 1.00, "K": 1.00}},
    "Tropicana Field": {"factors_vs_lhb": {"HR": 0.91, "2B": 0.98, "3B": 0.90, "1B": 0.99, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 0.92, "2B": 0.98, "3B": 0.90, "1B": 0.99, "BB": 1.00, "K": 1.00}},
    "Wrigley Field": {"factors_vs_lhb": {"HR": 1.02, "2B": 1.03, "3B": 1.06, "1B": 1.00, "BB": 1.00, "K": 0.99}, "factors_vs_rhb": {"HR": 1.03, "2B": 1.03, "3B": 1.05, "1B": 1.00, "BB": 1.00, "K": 0.99}},
    "Yankee Stadium": {"factors_vs_lhb": {"HR": 1.20, "2B": 0.95, "3B": 0.80, "1B": 0.98, "BB": 1.00, "K": 1.00}, "factors_vs_rhb": {"HR": 1.05, "2B": 0.98, "3B": 0.85, "1B": 1.00, "BB": 1.00, "K": 1.00}},
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


def get_park_factors(venue_name: str) -> ParkFactors:
    """Return park-factor multipliers for the given venue."""
    factors = PARK_FACTORS.get(venue_name)
    if factors is None:
        factors = {
            "factors_vs_lhb": dict(_NEUTRAL_FACTORS),
            "factors_vs_rhb": dict(_NEUTRAL_FACTORS),
        }

    return ParkFactors(
        venue_id=venue_name or "unknown",
        venue_name=venue_name or "Unknown Venue",
        factors_vs_lhb=dict(factors["factors_vs_lhb"]),
        factors_vs_rhb=dict(factors["factors_vs_rhb"]),
    )


def get_venue_for_team(team_name: str) -> str:
    """Map a team name to its home venue name."""
    return TEAM_TO_VENUE.get(team_name, "")
