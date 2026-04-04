"""Weather helpers for MLB game-context assembly."""
from __future__ import annotations

from mlb.config import WindDirection
from mlb.data.models import Weather

INDOOR_VENUES = {
    "American Family Field",
    "Chase Field",
    "Globe Life Field",
    "loanDepot Park",
    "Minute Maid Park",
    "Daikin Park",
    "Rogers Centre",
    "T-Mobile Park",
    "Tropicana Field",
}


def get_game_weather(venue_name: str, game_datetime: str) -> Weather:
    """Return neutral placeholder weather until a real API is integrated."""
    # TODO: Replace this stub with a real game-time weather feed.
    _ = game_datetime
    return Weather(
        temperature_f=72.0,
        wind_speed_mph=5.0,
        wind_direction=WindDirection.CALM,
        humidity_pct=50.0,
        is_indoor=venue_name in INDOOR_VENUES,
    )
