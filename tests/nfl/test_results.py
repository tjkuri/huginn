import csv
import io
from unittest.mock import patch, Mock
import requests
from nfl.data.results import fetch_season_rows, find_actual_yards

CSV_TEXT = (
    "player_id,player_display_name,season,week,season_type,passing_yards\n"
    "00-0033873,Patrick Mahomes,2026,3,REG,301\n"
    "00-0033873,Patrick Mahomes,2026,4,REG,\n"
    "00-0033873,Patrick Mahomes,2026,3,POST,999\n"
)


class TestFetchSeasonRows:
    def test_returns_parsed_rows_on_success(self):
        mock_resp = Mock(status_code=200, text=CSV_TEXT)
        mock_resp.raise_for_status = Mock()
        with patch("nfl.data.results.requests.get", return_value=mock_resp):
            rows = fetch_season_rows(2026)
        assert len(rows) == 3
        assert rows[0]["player_id"] == "00-0033873"

    def test_returns_empty_list_on_request_failure(self):
        with patch("nfl.data.results.requests.get", side_effect=requests.RequestException("boom")):
            rows = fetch_season_rows(2026)
        assert rows == []


class TestFindActualYards:
    def setup_method(self):
        self.rows = list(csv.DictReader(io.StringIO(CSV_TEXT)))

    def test_finds_matching_player_and_week(self):
        assert find_actual_yards(self.rows, "00-0033873", 3) == 301.0

    def test_returns_none_for_missing_week(self):
        assert find_actual_yards(self.rows, "00-0033873", 99) is None

    def test_returns_none_for_empty_stat(self):
        assert find_actual_yards(self.rows, "00-0033873", 4) is None

    def test_returns_none_for_unknown_player(self):
        assert find_actual_yards(self.rows, "00-nobody", 3) is None

    def test_ignores_non_reg_rows_for_same_player_and_week(self):
        # Fixture has a POST row for the same player_id/week with a different (bogus) value —
        # proves the season_type == "REG" filter actually discriminates, not just passes
        # through because the fixture only ever had REG rows.
        assert find_actual_yards(self.rows, "00-0033873", 3) == 301.0
