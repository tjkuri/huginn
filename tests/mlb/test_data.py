import logging

import pytest

from mlb.config import Hand
from mlb.data.builder import build_game_context
from mlb.data.models import GameContext
from mlb.data.park_factors import get_park_factors, get_venue_for_team
from mlb.data.stats import (
    _batting_threshold_for_season,
    build_batter_stats,
    build_pitcher_stats,
    fetch_batting_splits,
    fetch_pitching_splits,
)
from mlb.data.lineups import fetch_todays_games
from mlb.data.weather import get_game_weather


def _raw_batter(name="Test Batter", bats="R"):
    return {
        "player_id": "b1",
        "name": name,
        "bats": bats,
        "pa": 150,
        "rates": {
            "K": 30 / 150,
            "BB": 12 / 150,
            "HBP": 2 / 150,
            "1B": 20 / 150,
            "2B": 5 / 150,
            "3B": 1 / 150,
            "HR": 3 / 150,
            "OUT": 77 / 150,
        },
    }


def _raw_pitcher(name="Test Pitcher", throws="R"):
    return {
        "player_id": "p1",
        "name": name,
        "throws": throws,
        "pa_against": 180,
        "ip": 60,
        "rates": {
            "K": 40 / 180,
            "BB": 18 / 180,
            "HBP": 2 / 180,
            "1B": 30 / 180,
            "2B": 8 / 180,
            "3B": 1 / 180,
            "HR": 6 / 180,
            "OUT": 75 / 180,
        },
    }


class TestBuildBatterStats:
    def test_produces_valid_batter_stats(self):
        batter = build_batter_stats(_raw_batter(), "R")
        assert batter.bats == Hand.RIGHT
        assert sum(batter.rates.values()) == pytest.approx(1.0)
        assert batter.rates["OUT"] > 0

    def test_rate_conversion_math(self):
        batter = build_batter_stats(_raw_batter(), "R")
        assert batter.rates["K"] == pytest.approx(30 / 150)
        assert batter.rates["BB"] == pytest.approx(12 / 150)
        assert batter.rates["HBP"] == pytest.approx(2 / 150)
        assert batter.rates["1B"] == pytest.approx(20 / 150)
        assert batter.rates["2B"] == pytest.approx(5 / 150)
        assert batter.rates["3B"] == pytest.approx(1 / 150)
        assert batter.rates["HR"] == pytest.approx(3 / 150)


class TestBuildPitcherStats:
    def test_produces_valid_pitcher_stats(self):
        pitcher = build_pitcher_stats(_raw_pitcher())
        assert pitcher.throws == Hand.RIGHT
        assert sum(pitcher.rates.values()) == pytest.approx(1.0)
        assert pitcher.avg_pitch_count > 0


class TestParkFactors:
    def test_known_park(self):
        park = get_park_factors("Coors Field")
        assert park.factors_vs_lhb["HR"] > 1.0
        assert park.factors_vs_rhb["HR"] > 1.0

    def test_unknown_park(self):
        park = get_park_factors("Unknown Field")
        assert all(value == 1.0 for value in park.factors_vs_lhb.values())
        assert all(value == 1.0 for value in park.factors_vs_rhb.values())

    def test_team_to_venue_lookup(self):
        assert get_venue_for_team("Houston Astros") == "Daikin Park"


class TestWeather:
    def test_indoor(self):
        weather = get_game_weather("Daikin Park", "2026-04-03T19:10:00Z")
        assert weather.is_indoor is True

    def test_outdoor(self):
        weather = get_game_weather("Wrigley Field", "2026-04-03T19:10:00Z")
        assert weather.is_indoor is False


class TestFetchBattingSplits:
    def test_fetch_batting_splits_converts_stats(self, monkeypatch):
        class FakeFrame:
            def to_dict(self, orient="records"):
                assert orient == "records"
                return [
                    {
                        "Name": "Sample Batter",
                        "PA": 150,
                        "K%": 0.20,
                        "BB%": 0.08,
                        "HBP": 2,
                        "H": 29,
                        "2B": 5,
                        "3B": 1,
                        "HR": 3,
                        "IDfg": 123,
                        "Team": "ABC",
                        "Bat": "L",
                    }
                ]

        monkeypatch.setattr("mlb.data.stats._load_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._import_pybaseball", lambda: (lambda season, **kwargs: FakeFrame(), None))

        data = fetch_batting_splits(use_cache=False)
        player = data["sample batter"]
        assert player["rates"]["K"] == pytest.approx(0.20)
        assert player["rates"]["BB"] == pytest.approx(0.08)
        assert player["rates"]["HBP"] == pytest.approx(2 / 150)
        assert player["rates"]["1B"] == pytest.approx(20 / 150)
        assert sum(player["rates"].values()) == pytest.approx(1.0)

    def test_uses_2025_fallback_for_sparse_2026(self, monkeypatch):
        class FakeFrame:
            def __init__(self, rows):
                self.rows = rows

            def to_dict(self, orient="records"):
                assert orient == "records"
                return self.rows

        rows_2026 = [
            {"Name": "Qualified Batter", "PA": 25, "K%": 0.20, "BB%": 0.08, "HBP": 1, "H": 10, "2B": 2, "3B": 0, "HR": 1, "IDfg": 1, "Team": "ABC", "Bat": "R"},
            {"Name": "Fallback Batter", "PA": 5, "K%": 0.10, "BB%": 0.05, "HBP": 0, "H": 3, "2B": 0, "3B": 0, "HR": 0, "IDfg": 2, "Team": "ABC", "Bat": "L"},
        ]
        rows_2025 = [
            {"Name": "Qualified Batter", "PA": 120, "K%": 0.21, "BB%": 0.09, "HBP": 2, "H": 30, "2B": 5, "3B": 1, "HR": 4, "IDfg": 1, "Team": "ABC", "Bat": "R"},
            {"Name": "Fallback Batter", "PA": 130, "K%": 0.22, "BB%": 0.11, "HBP": 1, "H": 32, "2B": 6, "3B": 1, "HR": 5, "IDfg": 2, "Team": "ABC", "Bat": "L"},
        ]

        monkeypatch.setattr("mlb.data.stats._load_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._season_date_today", lambda: __import__("datetime").date(2026, 4, 2))

        def fake_batting_stats(season, **kwargs):
            return FakeFrame(rows_2026 if season == 2026 else rows_2025)

        monkeypatch.setattr("mlb.data.stats._import_pybaseball", lambda: (fake_batting_stats, None))

        data = fetch_batting_splits(season=2026, use_cache=False)
        assert data["qualified batter"]["source"] == "2026"
        assert data["qualified batter"]["pa"] == 25
        assert data["fallback batter"]["source"] == "2025"
        assert data["fallback batter"]["pa"] == 130

    def test_early_season_threshold_is_20_pa(self, monkeypatch):
        monkeypatch.setattr("mlb.data.stats._season_date_today", lambda: __import__("datetime").date(2026, 4, 2))
        assert _batting_threshold_for_season(2026) == 20

    def test_2025_raw_cache_used_on_next_call(self, tmp_path, monkeypatch):
        """After the first scrape, raw_batting-2025.json is loaded instead of re-scraping."""
        import mlb.data.stats as stats_mod
        monkeypatch.setattr(stats_mod, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(stats_mod, "SEASON", 2026)

        class FakeFrame:
            def to_dict(self, orient="records"):
                return [{"Name": "Cached Batter", "PA": 120, "K%": 0.20, "BB%": 0.08,
                         "HBP": 2, "H": 30, "2B": 5, "3B": 1, "HR": 3, "IDfg": 1, "Team": "NY", "Bat": "R"}]

        scrape_count = {"n": 0}

        def fake_batting_stats(season, **kwargs):
            scrape_count["n"] += 1
            return FakeFrame()

        monkeypatch.setattr(stats_mod, "_import_pybaseball", lambda: (fake_batting_stats, None))
        monkeypatch.setattr(stats_mod, "_season_date_today", lambda: __import__("datetime").date(2026, 4, 2))

        # First call — scrapes and writes cache
        stats_mod._fetch_batting_season_raw(2025, fake_batting_stats, use_cache=True)
        assert scrape_count["n"] == 1

        # Second call — loads from cache, no new scrape
        result = stats_mod._fetch_batting_season_raw(2025, fake_batting_stats, use_cache=True)
        assert scrape_count["n"] == 1
        assert "cached batter" in result


class TestFetchPitchingSplits:
    def test_source_tagging_and_2025_fallback(self, monkeypatch):
        class FakeFrame:
            def __init__(self, rows):
                self.rows = rows

            def to_dict(self, orient="records"):
                assert orient == "records"
                return self.rows

        rows_2026 = [
            {"Name": "Qualified Pitcher", "IP": 12.0, "H": 8, "2B": 2, "3B": 0, "HR": 1, "BB": 3, "HBP": 0, "SO": 14, "IDfg": 1, "Team": "ABC", "Throws": "R"},
            {"Name": "Fallback Pitcher", "IP": 2.0, "H": 3, "2B": 0, "3B": 0, "HR": 0, "BB": 1, "HBP": 0, "SO": 1, "IDfg": 2, "Team": "ABC", "Throws": "L"},
        ]
        rows_2025 = [
            {"Name": "Fallback Pitcher", "IP": 90.0, "H": 70, "2B": 12, "3B": 1, "HR": 10, "BB": 25, "HBP": 2, "SO": 95, "IDfg": 2, "Team": "ABC", "Throws": "L"},
        ]

        monkeypatch.setattr("mlb.data.stats._load_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._season_date_today", lambda: __import__("datetime").date(2026, 4, 2))

        def fake_pitching_stats(season, **kwargs):
            return FakeFrame(rows_2026 if season == 2026 else rows_2025)

        monkeypatch.setattr("mlb.data.stats._import_pybaseball", lambda: (None, fake_pitching_stats))

        data = fetch_pitching_splits(season=2026, use_cache=False)
        assert data["qualified pitcher"]["source"] == "2026"
        assert data["fallback pitcher"]["source"] == "2025"


class TestFetchTodaysGames:
    def test_schedule_parses_statsapi_flat_keys(self, monkeypatch):
        class FakeStatsApi:
            @staticmethod
            def schedule(date=None):
                return [
                    {
                        "game_id": 123,
                        "away_name": "New York Yankees",
                        "away_id": 147,
                        "home_name": "Boston Red Sox",
                        "home_id": 111,
                        "game_datetime": "2026-04-02T23:05:00Z",
                        "venue_name": "Fenway Park",
                        "status": "Scheduled",
                    }
                ]

        monkeypatch.setattr("mlb.data.lineups._import_statsapi", lambda: FakeStatsApi)
        games = fetch_todays_games("2026-04-02")

        assert games == [
            {
                "game_id": "123",
                "away_team": "New York Yankees",
                "away_team_id": "147",
                "home_team": "Boston Red Sox",
                "home_team_id": "111",
                "game_datetime": "2026-04-02T23:05:00Z",
                "venue": "Fenway Park",
                "status": "Scheduled",
                "away_probable_pitcher": "",
                "home_probable_pitcher": "",
            }
        ]


class TestBuildGameContext:
    def test_builds_game_context_with_mock_data(self, monkeypatch):
        game_info = {
            "game_id": "1",
            "away_team": "Away Team",
            "away_team_id": "10",
            "home_team": "Houston Astros",
            "home_team_id": "20",
            "game_datetime": "2026-04-03T19:10:00Z",
            "venue": "Daikin Park",
            "status": "Scheduled",
        }

        away_batters = [
            {"name": f"Away Batter {i}", "id": f"a{i}", "batting_position": i, "bats": "R"}
            for i in range(1, 10)
        ]
        home_batters = [
            {"name": f"Home Batter {i}", "id": f"h{i}", "batting_position": i, "bats": "L"}
            for i in range(1, 10)
        ]
        lineup = {
            "away_batters": away_batters,
            "home_batters": home_batters,
            "away_pitcher": {"name": "Away Starter", "id": "ap", "throws": "R"},
            "home_pitcher": {"name": "Home Starter", "id": "hp", "throws": "L"},
        }
        batting_data = {
            f"away batter {i}": _raw_batter(name=f"Away Batter {i}", bats="R")
            for i in range(1, 10)
        }
        batting_data.update(
            {
                f"home batter {i}": _raw_batter(name=f"Home Batter {i}", bats="L")
                for i in range(1, 10)
            }
        )
        pitching_data = {
            "away starter": _raw_pitcher(name="Away Starter", throws="R"),
            "home starter": _raw_pitcher(name="Home Starter", throws="L"),
        }

        monkeypatch.setattr("mlb.data.builder.fetch_game_lineup", lambda game_id: lineup)

        context = build_game_context(game_info, batting_data, pitching_data)
        assert isinstance(context, GameContext)
        assert len(context.away_lineup.batting_order) == 9
        assert len(context.home_lineup.batting_order) == 9
        assert context.away_lineup.starting_pitcher.name == "Away Starter"
        assert context.home_lineup.starting_pitcher.name == "Home Starter"
        assert all(getattr(batter, "data_source", None) in {None, "unknown"} or getattr(batter, "data_source") == "2026" for batter in context.away_lineup.batting_order)
        assert len(context.away_lineup.bullpen) == 4
        assert len(context.home_lineup.bullpen) == 4
        assert context.weather is not None
        assert context.weather.is_indoor is True

    def test_missing_player_falls_back_to_league_average(self, monkeypatch, caplog):
        game_info = {
            "game_id": "1",
            "away_team": "Away Team",
            "away_team_id": "10",
            "home_team": "Home Team",
            "home_team_id": "20",
            "game_datetime": "2026-04-03T19:10:00Z",
            "venue": "Wrigley Field",
            "status": "Scheduled",
        }
        away_batters = [
            {"name": f"Away Batter {i}", "id": f"a{i}", "batting_position": i, "bats": "R"}
            for i in range(1, 10)
        ]
        home_batters = [
            {"name": f"Home Batter {i}", "id": f"h{i}", "batting_position": i, "bats": "L"}
            for i in range(1, 10)
        ]
        lineup = {
            "away_batters": away_batters,
            "home_batters": home_batters,
            "away_pitcher": {"name": "Away Starter", "id": "ap", "throws": "R"},
            "home_pitcher": {"name": "Home Starter", "id": "hp", "throws": "L"},
        }
        batting_data = {
            f"away batter {i}": _raw_batter(name=f"Away Batter {i}", bats="R")
            for i in range(1, 9)
        }
        batting_data.update(
            {
                f"home batter {i}": _raw_batter(name=f"Home Batter {i}", bats="L")
                for i in range(1, 10)
            }
        )
        pitching_data = {
            "away starter": _raw_pitcher(name="Away Starter", throws="R"),
            "home starter": _raw_pitcher(name="Home Starter", throws="L"),
        }

        monkeypatch.setattr("mlb.data.builder.fetch_game_lineup", lambda game_id: lineup)

        with caplog.at_level(logging.WARNING):
            context = build_game_context(game_info, batting_data, pitching_data)

        assert context.away_lineup.batting_order[-1].name == "Away Batter 9"
        assert getattr(context.away_lineup.batting_order[-1], "data_source") == "league_avg"
        assert sum(context.away_lineup.batting_order[-1].rates.values()) == pytest.approx(1.0)
        assert any("Missing batting data for Away Batter 9" in record.message for record in caplog.records)


class TestNormalizeName:
    def test_strips_accents(self):
        from mlb.data.stats import _normalize_name
        assert _normalize_name("Agustín Ramírez") == _normalize_name("Agustin Ramirez")

    def test_strips_accents_various(self):
        from mlb.data.stats import _normalize_name
        assert _normalize_name("José Caballero") == "jose caballero"
        assert _normalize_name("Néstor Cortés") == "nestor cortes"

    def test_strips_periods_from_initials(self):
        from mlb.data.stats import _normalize_name
        assert _normalize_name("J.C. Escarra") == _normalize_name("JC Escarra")

    def test_strips_trailing_jr_suffix(self):
        from mlb.data.stats import _normalize_name
        assert _normalize_name("Ronald Acuna Jr.") == _normalize_name("Ronald Acuna")

    def test_strips_trailing_sr_suffix(self):
        from mlb.data.stats import _normalize_name
        assert _normalize_name("Ken Griffey Sr") == _normalize_name("Ken Griffey")

    def test_strips_trailing_iii_suffix(self):
        from mlb.data.stats import _normalize_name
        assert _normalize_name("Cal Ripken III") == _normalize_name("Cal Ripken")

    def test_lowercases_and_collapses_whitespace(self):
        from mlb.data.stats import _normalize_name
        assert _normalize_name("  Aaron  Judge  ") == "aaron judge"

    def test_builder_normalize_name_matches_stats(self):
        from mlb.data.stats import _normalize_name as stats_norm
        from mlb.data.builder import _normalize_name as builder_norm
        assert stats_norm("Agustín Ramírez") == builder_norm("agustin ramirez")


