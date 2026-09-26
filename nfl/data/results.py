import csv
import io
import requests
from nfl.config import NFLVERSE_STATS_URL_TMPL


def fetch_season_rows(season):
    """
    Fetch nflverse's weekly player stats CSV for a season.
    Returns [] (not an error) on any request failure — an unreachable or not-yet-published
    season is "no results available yet," not a crash. Callers (grader.py) treat every
    lookup against an empty list as "not gradeable yet," same as a missing row.
    """
    url = NFLVERSE_STATS_URL_TMPL.format(season=season)
    try:
        resp = requests.get(url)
        resp.raise_for_status()
    except requests.RequestException:
        return []
    return list(csv.DictReader(io.StringIO(resp.text)))


def find_actual_yards(rows, player_id, week):
    """
    Find a QB's actual passing yards for a given week from already-fetched season rows.
    Returns None if the player/week isn't present (game not yet played, stat not yet
    published, or a bye week) — never raises for a missing row.
    """
    for row in rows:
        if (row.get("player_id") == player_id
                and str(row.get("week")) == str(week)
                and row.get("season_type") == "REG"):
            yards = row.get("passing_yards")
            if yards in (None, ""):
                return None
            return float(yards)
    return None
