import logging

import pandas as pd
import pytest

from mlb.config import Hand
from mlb.data.builder import RunPreload, build_game_context, preload_run_context
from mlb.data.mlb_stats_api import fetch_batting_split_rows, fetch_pitching_season_rows, parse_baseball_innings
from mlb.data.models import DataSourceStatus, GameContext, ParkFactors
from mlb.data.park_factors import (
    _convert_savant_index_to_multiplier,
    get_park_factors,
    get_park_factors_with_status,
    get_venue_for_team,
)
from mlb.data.stats import (
    _apply_marcel,
    _marcel_source_tag,
    _overall_league_avg_rates,
    _BATTER_REGRESSION_CONSTANTS,
    _BATTER_NORMALIZER,
    _marcel_batter_player,
    _marcel_pitcher_player,
    build_batter_stats,
    build_pitcher_stats,
    compute_league_averages,
    fetch_batting_splits,
    fetch_batting_splits_with_statuses,
    fetch_team_bullpen_stats,
    fetch_runtime_overall_league_averages,
    fetch_pitching_splits,
    fetch_pitching_splits_with_statuses,
    marcel_blend,
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
        "overall": {
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
            "source": "2026_overall",
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
        "overall": {
            "pa_against": 180,
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
            "source": "2026_overall",
        },
    }


def _preload_fixture(
    *,
    bullpen_by_team: dict[str, dict] | None = None,
    park_factors_by_venue: dict[str, ParkFactors] | None = None,
    park_factor_status_by_venue: dict[str, DataSourceStatus] | None = None,
) -> RunPreload:
    return RunPreload(
        bullpen_by_team=bullpen_by_team or {},
        park_factors_by_venue=park_factors_by_venue or {},
        park_factor_status_by_venue=park_factor_status_by_venue or {},
    )


class TestBuildBatterStats:
    def test_produces_valid_batter_stats(self):
        batter = build_batter_stats(_raw_batter(), "R")
        assert batter.bats == Hand.RIGHT
        assert sum(batter.rates.values()) == pytest.approx(1.0)
        assert batter.rates["OUT"] > 0
        assert batter.data_source == "2026_overall"

    def test_rate_conversion_math(self):
        batter = build_batter_stats(_raw_batter(), "R")
        assert batter.rates["K"] == pytest.approx(30 / 150)
        assert batter.rates["BB"] == pytest.approx(12 / 150)
        assert batter.rates["HBP"] == pytest.approx(2 / 150)
        assert batter.rates["1B"] == pytest.approx(20 / 150)
        assert batter.rates["2B"] == pytest.approx(5 / 150)
        assert batter.rates["3B"] == pytest.approx(1 / 150)
        assert batter.rates["HR"] == pytest.approx(3 / 150)

    def test_uses_vs_lhp_split_when_facing_left_hander(self):
        raw = _raw_batter()
        raw["splits"] = {
            "vs_lhp": {
                "pa": 40,
                "rates": dict(raw["rates"], HR=0.08, OUT=0.7033333333333334),
                "source": "2026_split",
            }
        }
        batter = build_batter_stats(raw, "R", pitcher_hand="L")
        assert batter.rates["HR"] == pytest.approx(0.08)
        assert batter.data_source == "2026_split"

    def test_falls_back_to_overall_when_batter_split_missing(self):
        raw = _raw_batter()
        batter = build_batter_stats(raw, "R", pitcher_hand="L")
        assert batter.rates["HR"] == pytest.approx(3 / 150)
        assert batter.data_source == "2026_overall"


class TestBuildPitcherStats:
    def test_produces_valid_pitcher_stats(self):
        pitcher = build_pitcher_stats(_raw_pitcher())
        assert pitcher.throws == Hand.RIGHT
        assert sum(pitcher.rates.values()) == pytest.approx(1.0)
        assert pitcher.avg_pitch_count > 0
        assert pitcher.data_source == "2026_overall"

    def test_uses_vs_lhb_split_when_facing_left_handed_batter(self):
        raw = _raw_pitcher()
        raw["splits"] = {
            "vs_lhb": {
                "pa_against": 70,
                "rates": dict(raw["rates"], K=0.30, OUT=0.5222222222222223),
                "source": "2026_split",
            }
        }
        pitcher = build_pitcher_stats(raw, batter_hand="L")
        assert pitcher.rates["K"] == pytest.approx(0.30)
        assert pitcher.data_source == "2026_split"


class TestParkFactors:
    def test_known_park(self, monkeypatch):
        monkeypatch.setattr(
            "mlb.data.park_factors.fetch_park_factors_with_source",
            lambda season: ("fresh", {
                "Coors Field": {
                    "factors_vs_lhb": {"HR": 1.25, "2B": 1.20, "3B": 1.40, "1B": 1.10, "BB": 1.00, "K": 0.95},
                    "factors_vs_rhb": {"HR": 1.30, "2B": 1.18, "3B": 1.35, "1B": 1.08, "BB": 1.00, "K": 0.94},
                }
            }),
        )
        park = get_park_factors("Coors Field")
        assert park.factors_vs_lhb["HR"] > 1.0
        assert park.factors_vs_rhb["HR"] > 1.0

    def test_unknown_park(self, monkeypatch):
        monkeypatch.setattr("mlb.data.park_factors.fetch_park_factors_with_source", lambda season: ("fresh", {}))
        park = get_park_factors("Unknown Field")
        assert all(value == 1.0 for value in park.factors_vs_lhb.values())
        assert all(value == 1.0 for value in park.factors_vs_rhb.values())

    def test_park_factor_status_uses_hardcoded_fallback_for_missing_venue(self, monkeypatch):
        monkeypatch.setattr(
            "mlb.data.park_factors.fetch_park_factors_with_source",
            lambda season: ("fresh", {}),
        )
        park, status = get_park_factors_with_status("Yankee Stadium")
        assert park.venue_name == "Yankee Stadium"
        assert status.status == "hardcoded_fallback"

    def test_park_factor_status_uses_neutral_degraded_for_unknown_venue(self, monkeypatch):
        monkeypatch.setattr(
            "mlb.data.park_factors.fetch_park_factors_with_source",
            lambda season: ("fresh", {}),
        )
        park, status = get_park_factors_with_status("Unknown Field")
        assert park.venue_name == "Unknown Field"
        assert status.status == "degraded"

    def test_team_to_venue_lookup(self):
        assert get_venue_for_team("Houston Astros") == "Daikin Park"

    def test_savant_index_conversion(self):
        assert _convert_savant_index_to_multiplier(115) == pytest.approx(1.15)

    def test_fetch_park_factors_fallback(self, monkeypatch):
        monkeypatch.setattr("mlb.data.park_factors._PARK_FACTOR_CACHE", {})
        monkeypatch.setattr("mlb.data.park_factors._park_factor_cache_path", lambda season: __import__("pathlib").Path("/tmp/does-not-exist.json"))
        monkeypatch.setattr("mlb.data.park_factors.requests.get", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

        factors = __import__("mlb.data.park_factors", fromlist=["fetch_park_factors"]).fetch_park_factors(2025)
        assert "Yankee Stadium" in factors
        assert factors["Yankee Stadium"]["factors_vs_lhb"]["HR"] == pytest.approx(1.18)


class TestLeagueAverageComputation:
    def test_compute_league_averages_known_totals(self):
        batting_df = pd.DataFrame(
            [
                {"PA": 100, "H": 30, "2B": 5, "3B": 1, "HR": 4, "HBP": 2, "SO": 20, "BB": 10},
                {"PA": 50, "H": 10, "2B": 2, "3B": 0, "HR": 1, "HBP": 1, "SO": 15, "BB": 5},
            ]
        )
        pitching_df = pd.DataFrame([{"IP": 10.0}])

        rates = compute_league_averages(batting_df, pitching_df)

        assert rates["K"] == pytest.approx(35 / 150)
        assert rates["BB"] == pytest.approx(15 / 150)
        assert rates["HR"] == pytest.approx(5 / 150)
        assert rates["1B"] == pytest.approx((40 - 7 - 1 - 5) / 150)
        assert rates["AVG"] == pytest.approx(40 / (150 - 15 - 3))

    def test_compute_league_averages_sum_to_one(self):
        batting_df = pd.DataFrame(
            [
                {"PA": 100, "H": 25, "2B": 5, "3B": 1, "HR": 4, "HBP": 2, "SO": 22, "BB": 8},
            ]
        )
        pitching_df = pd.DataFrame([{"IP": 5.0}])

        rates = compute_league_averages(batting_df, pitching_df)
        total = sum(rates[key] for key in ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT"))
        assert total == pytest.approx(1.0)


class TestMlbStatsApiClient:
    def test_parse_baseball_innings_handles_outs_notation(self):
        assert parse_baseball_innings("115.1") == pytest.approx(115 + (1 / 3))
        assert parse_baseball_innings("40.2") == pytest.approx(40 + (2 / 3))

    def test_fetch_pitching_season_rows_maps_games_and_innings(self, monkeypatch):
        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, params=None, timeout=None):
            if "api/v1/stats" in url:
                return FakeResponse(
                    {
                        "stats": [
                            {
                                "totalSplits": 1,
                                "splits": [
                                    {
                                        "player": {"id": 123, "fullName": "Sample Pitcher"},
                                        "team": {"name": "Boston Red Sox"},
                                        "stat": {
                                            "inningsPitched": "115.1",
                                            "battersFaced": 500,
                                            "gamesPlayed": 30,
                                            "gamesStarted": 12,
                                            "hits": 100,
                                            "doubles": 20,
                                            "triples": 3,
                                            "homeRuns": 10,
                                            "baseOnBalls": 40,
                                            "hitBatsmen": 5,
                                            "strikeOuts": 120,
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                )
            return FakeResponse({"people": [{"id": 123, "pitchHand": {"code": "L"}, "batSide": {"code": "R"}}]})

        monkeypatch.setattr("mlb.data.mlb_stats_api.requests.get", fake_get)

        rows = fetch_pitching_season_rows(2026)

        assert len(rows) == 1
        row = rows[0]
        assert row["ID"] == "123"
        assert row["Team"] == "BOS"
        assert row["Throws"] == "L"
        assert row["IP"] == pytest.approx(115 + (1 / 3))
        assert row["G"] == 30
        assert row["GS"] == 12
        assert row["2B"] == 20
        assert row["3B"] == 3
        assert row["HBP"] == 5

    def test_fetch_batting_split_rows_maps_vs_right_split(self, monkeypatch):
        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        def fake_get(url, params=None, timeout=None):
            if "api/v1/stats" in url:
                assert params["sitCodes"] == "vr"
                return FakeResponse(
                    {
                        "stats": [
                            {
                                "totalSplits": 1,
                                "splits": [
                                    {
                                        "player": {"id": 456, "fullName": "Sample Batter"},
                                        "team": {"name": "Boston Red Sox"},
                                        "stat": {
                                            "plateAppearances": 120,
                                            "hits": 30,
                                            "doubles": 6,
                                            "triples": 1,
                                            "homeRuns": 4,
                                            "hitByPitch": 2,
                                            "strikeOuts": 20,
                                            "baseOnBalls": 10,
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                )
            return FakeResponse({"people": [{"id": 456, "pitchHand": {"code": "R"}, "batSide": {"code": "L"}}]})

        monkeypatch.setattr("mlb.data.mlb_stats_api.requests.get", fake_get)

        rows = fetch_batting_split_rows(2026, sit_code="vr")

        assert rows == [
            {
                "ID": "456",
                "Name": "Sample Batter",
                "Team": "BOS",
                "Bats": "L",
                "PA": 120,
                "H": 30,
                "2B": 6,
                "3B": 1,
                "HR": 4,
                "HBP": 2,
                "SO": 20,
                "BB": 10,
            }
        ]

    def test_bullpen_aggregation_uses_games_started_filter(self, monkeypatch):
        import mlb.data.stats as stats_mod

        monkeypatch.setattr(stats_mod, "_TEAM_BULLPEN_CACHE", {})
        monkeypatch.setattr(
            "mlb.data.stats.fetch_pitching_season_rows",
            lambda season: [
                {
                    "ID": "1",
                    "Name": "Starter",
                    "Team": "BOS",
                    "Throws": "R",
                    "IP": 100.0,
                    "BF": 400,
                    "G": 20,
                    "GS": 20,
                    "H": 90,
                    "2B": 20,
                    "3B": 2,
                    "HR": 12,
                    "BB": 30,
                    "HBP": 2,
                    "SO": 110,
                },
                {
                    "ID": "2",
                    "Name": "Reliever",
                    "Team": "BOS",
                    "Throws": "R",
                    "IP": 40.0,
                    "BF": 180,
                    "G": 35,
                    "GS": 0,
                    "H": 30,
                    "2B": 5,
                    "3B": 1,
                    "HR": 4,
                    "BB": 18,
                    "HBP": 1,
                    "SO": 45,
                },
            ],
        )

        bullpen = fetch_team_bullpen_stats(season=2026, use_cache=False)

        assert "BOS" in bullpen
        assert bullpen["BOS"]["pa_against"] == 180
        assert bullpen["BOS"]["ip"] == pytest.approx(40.0)


class TestWeather:
    def test_indoor(self):
        weather = get_game_weather("Daikin Park", "2026-04-03T19:10:00Z")
        assert weather.is_indoor is True

    def test_outdoor(self):
        weather = get_game_weather("Wrigley Field", "2026-04-03T19:10:00Z")
        assert weather.is_indoor is False


class TestMarcelBlend:
    def test_returns_league_avg_when_no_seasons(self):
        result = marcel_blend([], regression_constant=200.0, league_avg=0.10, normalizer=200.0)
        assert result == pytest.approx(0.10)

    def test_single_full_season_regresses_toward_league_avg(self):
        # w_player = 5 * (200/200) = 5.0
        # w_lg = 200/200 = 1.0  (regression_constant=200, normalizer=200)
        # projected = (5*0.15 + 1*0.10) / (5+1) = 0.8500/6 = 0.14167
        result = marcel_blend([(5, 0.15, 200)], regression_constant=200.0, league_avg=0.10, normalizer=200.0)
        assert result == pytest.approx((5 * 0.15 + 1.0 * 0.10) / (5.0 + 1.0))

    def test_three_seasons_weighted_blend(self):
        # w1=5*(100/200)=2.5, w2=4*(100/200)=2.0, w3=3*(100/200)=1.5, w_lg=320/200=1.6
        w1, w2, w3, w_lg = 2.5, 2.0, 1.5, 320 / 200
        lg = 0.033
        expected = (w1 * 0.040 + w2 * 0.035 + w3 * 0.030 + w_lg * lg) / (w1 + w2 + w3 + w_lg)
        result = marcel_blend(
            [(5, 0.040, 100), (4, 0.035, 100), (3, 0.030, 100)],
            regression_constant=320.0,
            league_avg=lg,
            normalizer=200.0,
        )
        assert result == pytest.approx(expected)

    def test_zero_pa_season_contributes_no_weight(self):
        # Passing season with pa=0: w = 4*(0/200)=0 — same result as omitting it
        result_with = marcel_blend([(5, 0.20, 200), (4, 0.30, 0)], regression_constant=150.0, league_avg=0.22, normalizer=200.0)
        result_without = marcel_blend([(5, 0.20, 200)], regression_constant=150.0, league_avg=0.22, normalizer=200.0)
        assert result_with == pytest.approx(result_without)

    def test_higher_regression_constant_pulls_more_toward_league_avg(self):
        # K% (150) vs HR% (320): same observed rate, same league avg, same PA
        # More regression → stronger pull toward 0.05
        k_result = marcel_blend([(5, 0.10, 200)], regression_constant=150.0, league_avg=0.05, normalizer=200.0)
        hr_result = marcel_blend([(5, 0.10, 200)], regression_constant=320.0, league_avg=0.05, normalizer=200.0)
        assert hr_result < k_result  # HR% regresses more

    def test_pitcher_normalizer_150(self):
        # BF normalizer for pitchers is 150 not 200
        # w_player = 5*(150/150)=5.0, w_lg=150/150=1.0
        result = marcel_blend([(5, 0.25, 150)], regression_constant=150.0, league_avg=0.20, normalizer=150.0)
        assert result == pytest.approx((5 * 0.25 + 1.0 * 0.20) / (5.0 + 1.0))


class TestMarcelHelpers:
    def test_source_tag_3yr(self):
        assert _marcel_source_tag(3) == "marcel_3yr"

    def test_source_tag_2yr(self):
        assert _marcel_source_tag(2) == "marcel_2yr"

    def test_source_tag_1yr(self):
        assert _marcel_source_tag(1) == "marcel_1yr"

    def test_source_tag_0yr(self):
        assert _marcel_source_tag(0) == "league_avg"

    def test_overall_league_avg_rates_sum_to_one(self):
        rates = _overall_league_avg_rates()
        total = sum(rates[k] for k in ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT"))
        assert total == pytest.approx(1.0)

    def test_overall_league_avg_rates_all_positive(self):
        rates = _overall_league_avg_rates()
        for k, v in rates.items():
            assert v >= 0.0, f"{k} is negative: {v}"

    def test_apply_marcel_returns_normalized_rates(self):
        seasons = [(5, {"K": 0.22, "BB": 0.08, "HBP": 0.01, "1B": 0.14, "2B": 0.04, "3B": 0.005, "HR": 0.03, "OUT": 0.465}, 200)]
        lg = _overall_league_avg_rates()
        blended, n = _apply_marcel(seasons, _BATTER_REGRESSION_CONSTANTS, lg, _BATTER_NORMALIZER)
        total = sum(blended[k] for k in ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT"))
        assert total == pytest.approx(1.0)
        assert n == 1

    def test_apply_marcel_counts_seasons_with_data(self):
        lg = _overall_league_avg_rates()
        rates = {"K": 0.22, "BB": 0.08, "HBP": 0.01, "1B": 0.14, "2B": 0.04, "3B": 0.005, "HR": 0.03, "OUT": 0.465}
        seasons = [(5, rates, 200), (4, rates, 100), (3, rates, 0)]  # 2024 has zero PA
        _, n = _apply_marcel(seasons, _BATTER_REGRESSION_CONSTANTS, lg, _BATTER_NORMALIZER)
        assert n == 2

    def test_apply_marcel_empty_seasons_returns_league_avg(self):
        lg = _overall_league_avg_rates()
        blended, n = _apply_marcel([], _BATTER_REGRESSION_CONSTANTS, lg, _BATTER_NORMALIZER)
        assert n == 0
        assert blended["K"] == pytest.approx(lg["K"])


def _batter_season_row(pa: int, k_rate: float = 0.22, hr_rate: float = 0.03, season: int = 2026) -> dict:
    """Build a minimal raw batter season record for Marcel tests."""
    bb = 0.08
    hbp = 0.01
    singles = 0.14
    doubles = 0.04
    triples = 0.005
    out = max(0.0, 1.0 - k_rate - bb - hbp - singles - doubles - triples - hr_rate)
    return {
        "player_id": "100",
        "name": "Test Player",
        "team": "NYY",
        "bats": "R",
        "pa": pa,
        "rates": {"K": k_rate, "BB": bb, "HBP": hbp, "1B": singles, "2B": doubles, "3B": triples, "HR": hr_rate, "OUT": out},
        "season": season,
        "source": f"{season}_overall",
        "split_type": "overall",
    }


def _pitcher_season_row(pa_against: int, ip: float = 30.0, k_rate: float = 0.25, season: int = 2026) -> dict:
    """Build a minimal raw pitcher season record for Marcel tests."""
    bb = 0.08
    hbp = 0.01
    singles = 0.14
    doubles = 0.04
    triples = 0.003
    hr = 0.03
    out = max(0.0, 1.0 - k_rate - bb - hbp - singles - doubles - triples - hr)
    return {
        "player_id": "200",
        "name": "Test Pitcher",
        "team": "BOS",
        "throws": "R",
        "pa_against": pa_against,
        "ip": ip,
        "rates": {"K": k_rate, "BB": bb, "HBP": hbp, "1B": singles, "2B": doubles, "3B": triples, "HR": hr, "OUT": out},
        "avg_pitch_count": 85.0,
        "season": season,
        "source": f"{season}_overall",
        "split_type": "overall",
    }


class TestMarcelBatterPlayer:
    def test_returns_none_with_no_data(self):
        from mlb.config import Hand
        result = _marcel_batter_player(
            None, None, None,
            None, None, None,
            None, None, None,
            Hand.RIGHT,
        )
        assert result is None

    def test_single_season_tagged_marcel_1yr(self):
        from mlb.config import Hand
        result = _marcel_batter_player(
            _batter_season_row(200), None, None,
            None, None, None,
            None, None, None,
            Hand.RIGHT,
        )
        assert result is not None
        assert result["source"] == "marcel_1yr"
        assert result["overall"]["source"] == "marcel_1yr"

    def test_two_seasons_tagged_marcel_2yr(self):
        from mlb.config import Hand
        result = _marcel_batter_player(
            _batter_season_row(200, season=2026), _batter_season_row(400, season=2025), None,
            None, None, None,
            None, None, None,
            Hand.RIGHT,
        )
        assert result["source"] == "marcel_2yr"

    def test_three_seasons_tagged_marcel_3yr(self):
        from mlb.config import Hand
        result = _marcel_batter_player(
            _batter_season_row(200, season=2026),
            _batter_season_row(400, season=2025),
            _batter_season_row(500, season=2024),
            None, None, None,
            None, None, None,
            Hand.RIGHT,
        )
        assert result["source"] == "marcel_3yr"

    def test_sparse_current_season_projects_closer_to_prior(self):
        """Player with 20 PA in 2026 (K=0.30) and 400 PA in 2025 (K=0.22) should project close to 0.22."""
        from mlb.config import Hand
        result = _marcel_batter_player(
            _batter_season_row(20, k_rate=0.30, season=2026),
            _batter_season_row(400, k_rate=0.22, season=2025),
            None,
            None, None, None,
            None, None, None,
            Hand.RIGHT,
        )
        assert result is not None
        k = result["rates"]["K"]
        assert abs(k - 0.22) < abs(k - 0.30), f"Marcel K%={k:.4f} should be closer to 0.22 than 0.30"

    def test_rates_sum_to_one(self):
        from mlb.config import Hand
        result = _marcel_batter_player(
            _batter_season_row(200), None, None,
            None, None, None,
            None, None, None,
            Hand.RIGHT,
        )
        total = sum(result["rates"].values())
        assert total == pytest.approx(1.0)

    def test_split_included_when_split_data_present(self):
        from mlb.config import Hand
        lhp_row = dict(_batter_season_row(80), split_type="vs_lhp", source="2026_split")
        lhp_row["pa"] = 80
        result = _marcel_batter_player(
            _batter_season_row(200), None, None,
            lhp_row, None, None,
            None, None, None,
            Hand.RIGHT,
        )
        assert "vs_lhp" in result["splits"]
        assert result["splits"]["vs_lhp"]["source"] == "marcel_1yr"

    def test_split_omitted_when_no_split_data(self):
        from mlb.config import Hand
        result = _marcel_batter_player(
            _batter_season_row(200), None, None,
            None, None, None,
            None, None, None,
            Hand.RIGHT,
        )
        assert result["splits"] == {}

    def test_overall_block_has_expected_keys(self):
        from mlb.config import Hand
        result = _marcel_batter_player(
            _batter_season_row(200), None, None,
            None, None, None,
            None, None, None,
            Hand.RIGHT,
        )
        assert set(result["overall"].keys()) >= {"pa", "rates", "source", "split_type"}

    def test_uses_primary_season_metadata(self):
        from mlb.config import Hand
        result = _marcel_batter_player(
            _batter_season_row(200), None, None,
            None, None, None,
            None, None, None,
            Hand.RIGHT,
        )
        assert result["player_id"] == "100"
        assert result["name"] == "Test Player"
        assert result["bats"] == "R"


class TestMarcelPitcherPlayer:
    def test_returns_none_with_no_data(self):
        result = _marcel_pitcher_player(
            None, None, None,
            None, None, None,
            None, None, None,
        )
        assert result is None

    def test_single_season_tagged_marcel_1yr(self):
        result = _marcel_pitcher_player(
            _pitcher_season_row(150, ip=30.0), None, None,
            None, None, None,
            None, None, None,
        )
        assert result["source"] == "marcel_1yr"

    def test_two_seasons_tagged_marcel_2yr(self):
        result = _marcel_pitcher_player(
            _pitcher_season_row(150, ip=30.0, season=2026),
            _pitcher_season_row(600, ip=120.0, season=2025),
            None,
            None, None, None,
            None, None, None,
        )
        assert result["source"] == "marcel_2yr"

    def test_rates_sum_to_one(self):
        result = _marcel_pitcher_player(
            _pitcher_season_row(150, ip=30.0), None, None,
            None, None, None,
            None, None, None,
        )
        total = sum(result["rates"].values())
        assert total == pytest.approx(1.0)

    def test_splits_included_when_split_data_present(self):
        lhb_row = dict(_pitcher_season_row(70, ip=20.0), split_type="vs_lhb")
        lhb_row["pa_against"] = 70
        result = _marcel_pitcher_player(
            _pitcher_season_row(150, ip=30.0), None, None,
            lhb_row, None, None,
            None, None, None,
        )
        assert "vs_lhb" in result["splits"]


class TestFetchBattingSplits:
    def test_fetch_batting_splits_converts_stats(self, monkeypatch):
        rows = [
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
                "ID": 123,
                "Team": "ABC",
                "Bats": "L",
            }
        ]

        monkeypatch.setattr("mlb.data.stats._load_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats.fetch_batting_season_rows", lambda season: rows)
        monkeypatch.setattr("mlb.data.stats._fetch_batting_split_raw_with_status", lambda *args, **kwargs: ({}, "fresh"))

        data = fetch_batting_splits(use_cache=False)
        player = data["sample batter"]
        # Marcel blends rates toward league average, so exact raw rates won't match.
        # Verify the rates are positive, in plausible range, and sum to 1.
        assert 0.15 < player["rates"]["K"] < 0.25
        assert 0.05 < player["rates"]["BB"] < 0.12
        assert player["rates"]["HBP"] > 0
        assert player["rates"]["1B"] > 0
        assert sum(player["rates"].values()) == pytest.approx(1.0)
        assert player["source"] == "marcel_3yr"
        assert player["overall"]["source"] == "marcel_3yr"

    def test_marcel_blends_sparse_current_season_toward_prior(self, monkeypatch):
        """A player with sparse 2026 data and good 2025 data should project closer to 2025 rates."""
        rows_2026 = [
            {"Name": "Qualified Batter", "PA": 25, "K%": 0.20, "BB%": 0.08, "HBP": 1, "H": 10, "2B": 2, "3B": 0, "HR": 1, "ID": 1, "Team": "ABC", "Bats": "R"},
            {"Name": "Sparse Batter", "PA": 5, "K%": 0.10, "BB%": 0.05, "HBP": 0, "H": 3, "2B": 0, "3B": 0, "HR": 0, "ID": 2, "Team": "ABC", "Bats": "L"},
        ]
        rows_2025 = [
            {"Name": "Qualified Batter", "PA": 120, "K%": 0.21, "BB%": 0.09, "HBP": 2, "H": 30, "2B": 5, "3B": 1, "HR": 4, "ID": 1, "Team": "ABC", "Bats": "R"},
            {"Name": "Sparse Batter", "PA": 130, "K%": 0.22, "BB%": 0.11, "HBP": 1, "H": 32, "2B": 6, "3B": 1, "HR": 5, "ID": 2, "Team": "ABC", "Bats": "L"},
        ]

        monkeypatch.setattr("mlb.data.stats._load_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._fetch_batting_split_raw_with_status", lambda *args, **kwargs: ({}, "fresh"))

        monkeypatch.setattr(
            "mlb.data.stats.fetch_batting_season_rows",
            lambda season: rows_2026 if season == 2026 else rows_2025 if season == 2025 else [],
        )

        data = fetch_batting_splits(season=2026, use_cache=False)

        assert data["qualified batter"]["source"] == "marcel_2yr"
        assert data["sparse batter"]["source"] == "marcel_2yr"

        # Sparse batter: 5 PA in 2026 (K%=0.10), 130 PA in 2025 (K%=0.22)
        # Marcel heavily weights 2025 → projected K% should be close to 0.22, not 0.10
        sparse_k = data["sparse batter"]["rates"]["K"]
        assert abs(sparse_k - 0.22) < abs(sparse_k - 0.10), (
            f"Marcel K%={sparse_k:.4f} should be closer to 0.22 (2025) than 0.10 (2026)"
        )

    def test_2025_raw_cache_used_on_next_call(self, tmp_path, monkeypatch):
        """After the first scrape, raw_batting-2025.json is loaded instead of re-scraping."""
        import mlb.data.stats as stats_mod
        monkeypatch.setattr(stats_mod, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(stats_mod, "SEASON", 2026)

        scrape_count = {"n": 0}

        def fake_batting_rows(season):
            scrape_count["n"] += 1
            return [{"Name": "Cached Batter", "PA": 120, "K%": 0.20, "BB%": 0.08,
                     "HBP": 2, "H": 30, "2B": 5, "3B": 1, "HR": 3, "ID": 1, "Team": "NY", "Bats": "R"}]

        monkeypatch.setattr(stats_mod, "fetch_batting_season_rows", fake_batting_rows)

        # First call — scrapes and writes cache
        stats_mod._fetch_batting_season_raw(2025, use_cache=True)
        assert scrape_count["n"] == 1

        # Second call — loads from cache, no new scrape
        result = stats_mod._fetch_batting_season_raw(2025, use_cache=True)
        assert scrape_count["n"] == 1
        assert "cached batter" in result

    def test_batting_split_raw_uses_statsapi_hand_mapping(self, monkeypatch):
        import mlb.data.stats as stats_mod

        seen_codes: list[str] = []
        monkeypatch.setattr(stats_mod, "_load_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr(stats_mod, "_save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            stats_mod,
            "fetch_batting_split_rows",
            lambda season, sit_code: seen_codes.append(sit_code) or [
                {
                    "ID": "1",
                    "Name": "Split Batter",
                    "Team": "ABC",
                    "Bats": "L",
                    "PA": 80,
                    "H": 20,
                    "2B": 5,
                    "3B": 1,
                    "HR": 3,
                    "HBP": 1,
                    "SO": 12,
                    "BB": 7,
                }
            ],
        )

        players_lhp, tier_lhp = stats_mod._fetch_batting_split_raw_with_status(
            2026,
            stats_mod._SPLIT_MONTH_VS_LEFT,
            False,
            kind="batting_vs_lhp",
            split_type="vs_lhp",
        )
        players_rhp, tier_rhp = stats_mod._fetch_batting_split_raw_with_status(
            2026,
            stats_mod._SPLIT_MONTH_VS_RIGHT,
            False,
            kind="batting_vs_rhp",
            split_type="vs_rhp",
        )

        assert seen_codes == ["vl", "vr"]
        assert tier_lhp == "fresh"
        assert tier_rhp == "fresh"
        assert "split batter" in players_lhp
        assert "split batter" in players_rhp

    def test_overall_fetch_failure_degrades_to_prior_season(self, monkeypatch):
        rows_2025 = [
            {"Name": "Fallback Batter", "PA": 120, "K%": 0.21, "BB%": 0.09, "HBP": 2, "H": 30, "2B": 5, "3B": 1, "HR": 4, "ID": 1, "Team": "ABC", "Bats": "R"},
        ]

        monkeypatch.setattr("mlb.data.stats._load_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._fetch_batting_split_raw_with_status", lambda *args, **kwargs: ({}, "fresh"))

        def fake_batting_rows(season):
            if season == 2026:
                raise RuntimeError("2026 blocked")
            if season == 2025:
                return rows_2025
            raise RuntimeError("2024 blocked")

        monkeypatch.setattr("mlb.data.stats.fetch_batting_season_rows", fake_batting_rows)

        data = fetch_batting_splits(season=2026, use_cache=False)

        assert data["fallback batter"]["source"] == "marcel_1yr"

    def test_source_statuses_capture_batting_overall_and_split_seasons(self, monkeypatch):
        rows_2026 = [
            {"Name": "Status Batter", "PA": 120, "K%": 0.21, "BB%": 0.09, "HBP": 2, "H": 30, "2B": 5, "3B": 1, "HR": 4, "ID": 1, "Team": "ABC", "Bats": "R"},
        ]
        monkeypatch.setattr("mlb.data.stats._load_raw_cache", lambda kind, season, max_age_hours=None: {"status batter": _raw_batter("Status Batter")} if season == 2025 else None)
        monkeypatch.setattr("mlb.data.stats._save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            "mlb.data.stats._fetch_batting_split_raw_with_status",
            lambda season, split_month, use_cache, **kwargs: (
                ({"status batter": _raw_batter("Status Batter")} if season == 2025 else {}),
                ("cache" if season == 2025 else "fresh"),
            ),
        )
        monkeypatch.setattr(
            "mlb.data.stats.fetch_batting_season_rows",
            lambda season: rows_2026 if season == 2026 else (_ for _ in ()).throw(RuntimeError(f"{season} blocked")),
        )

        _, source_statuses = fetch_batting_splits_with_statuses(season=2026, use_cache=True)

        status_map = {status.source_name: status for status in source_statuses}
        assert status_map["batting_overall_2026"].status == "fresh"
        assert status_map["batting_overall_2025"].status == "cache"
        assert status_map["batting_overall_2024"].status == "degraded"
        assert status_map["batting_split_vs_lhp_2025"].status == "cache"
        assert status_map["batting_split_vs_lhp_2026"].status == "degraded"

    def test_legacy_batting_wrapper_matches_status_return_data(self, monkeypatch):
        monkeypatch.setattr(
            "mlb.data.stats.fetch_batting_splits_with_statuses",
            lambda season=2026, use_cache=True: ({"wrapped batter": {"source": "marcel_1yr"}}, [DataSourceStatus("x", "required", "run_wide", "fresh", "ok")]),
        )

        data = fetch_batting_splits(season=2026, use_cache=True)

        assert data == {"wrapped batter": {"source": "marcel_1yr"}}

    def test_raises_when_no_usable_overall_batting_stats_exist(self, monkeypatch):
        monkeypatch.setattr("mlb.data.stats._load_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._fetch_batting_split_raw_with_status", lambda *args, **kwargs: ({}, "fresh"))
        monkeypatch.setattr("mlb.data.stats.fetch_batting_season_rows", lambda season: (_ for _ in ()).throw(RuntimeError(f"{season} blocked")))

        with pytest.raises(RuntimeError, match="No usable batting overall stats"):
            fetch_batting_splits(season=2026, use_cache=False)


class TestFetchPitchingSplits:
    def test_source_tagging_and_2025_fallback(self, monkeypatch):
        rows_2026 = [
            {"Name": "Qualified Pitcher", "IP": 12.0, "H": 8, "2B": 2, "3B": 0, "HR": 1, "BB": 3, "HBP": 0, "SO": 14, "ID": 1, "Team": "ABC", "Throws": "R"},
            {"Name": "Sparse Pitcher", "IP": 2.0, "H": 3, "2B": 0, "3B": 0, "HR": 0, "BB": 1, "HBP": 0, "SO": 1, "ID": 2, "Team": "ABC", "Throws": "L"},
        ]
        rows_2025 = [
            {"Name": "Sparse Pitcher", "IP": 90.0, "H": 70, "2B": 12, "3B": 1, "HR": 10, "BB": 25, "HBP": 2, "SO": 95, "ID": 2, "Team": "ABC", "Throws": "L"},
        ]

        monkeypatch.setattr("mlb.data.stats._load_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._fetch_pitching_split_raw_with_status", lambda *args, **kwargs: ({}, "fresh"))

        monkeypatch.setattr(
            "mlb.data.stats.fetch_pitching_season_rows",
            lambda season: rows_2026 if season == 2026 else rows_2025 if season == 2025 else [],
        )

        data = fetch_pitching_splits(season=2026, use_cache=False)
        assert data["qualified pitcher"]["source"] == "marcel_1yr"
        assert data["qualified pitcher"]["splits"] == {}
        assert data["sparse pitcher"]["source"] == "marcel_2yr"

    def test_overall_only_mode_keeps_split_keys_empty(self, monkeypatch):
        rows = [
            {"Name": "Sample Pitcher", "IP": 25.0, "H": 20, "2B": 3, "3B": 0, "HR": 2, "BB": 5, "HBP": 1, "SO": 28, "ID": 1, "Team": "ABC", "Throws": "R"},
        ]

        monkeypatch.setattr("mlb.data.stats._load_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats.fetch_pitching_season_rows", lambda season: rows)
        monkeypatch.setattr("mlb.data.stats._fetch_pitching_split_raw_with_status", lambda *args, **kwargs: ({}, "fresh"))

        data = fetch_pitching_splits(season=2026, use_cache=False)
        assert data["sample pitcher"]["splits"] == {}

    def test_overall_fetch_failure_degrades_to_prior_season(self, monkeypatch):
        rows_2025 = [
            {"Name": "Fallback Pitcher", "IP": 90.0, "H": 70, "2B": 12, "3B": 1, "HR": 10, "BB": 25, "HBP": 2, "SO": 95, "ID": 2, "Team": "ABC", "Throws": "L"},
        ]

        monkeypatch.setattr("mlb.data.stats._load_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._fetch_pitching_split_raw_with_status", lambda *args, **kwargs: ({}, "fresh"))

        def fake_pitching_rows(season):
            if season == 2026:
                raise RuntimeError("2026 blocked")
            if season == 2025:
                return rows_2025
            raise RuntimeError("2024 blocked")

        monkeypatch.setattr("mlb.data.stats.fetch_pitching_season_rows", fake_pitching_rows)

        data = fetch_pitching_splits(season=2026, use_cache=False)

        assert data["fallback pitcher"]["source"] == "marcel_1yr"

    def test_source_statuses_capture_pitching_overall_and_split_seasons(self, monkeypatch):
        rows_2026 = [
            {"Name": "Status Pitcher", "IP": 90.0, "H": 70, "2B": 12, "3B": 1, "HR": 10, "BB": 25, "HBP": 2, "SO": 95, "ID": 2, "Team": "ABC", "Throws": "L"},
        ]
        monkeypatch.setattr("mlb.data.stats._load_raw_cache_payload", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._load_raw_cache", lambda kind, season, max_age_hours=None: {"status pitcher": _raw_pitcher("Status Pitcher")} if season == 2025 else None)
        monkeypatch.setattr("mlb.data.stats._save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            "mlb.data.stats._fetch_pitching_split_raw_with_status",
            lambda season, split_month, use_cache, **kwargs: (
                ({"status pitcher": _raw_pitcher("Status Pitcher")} if season == 2025 else {}),
                ("cache" if season == 2025 else "fresh"),
            ),
        )
        monkeypatch.setattr(
            "mlb.data.stats.fetch_pitching_season_rows",
            lambda season: rows_2026 if season == 2026 else (_ for _ in ()).throw(RuntimeError(f"{season} blocked")),
        )

        _, source_statuses = fetch_pitching_splits_with_statuses(season=2026, use_cache=True)

        status_map = {status.source_name: status for status in source_statuses}
        assert status_map["pitching_overall_2026"].status == "fresh"
        assert status_map["pitching_overall_2025"].status == "cache"
        assert status_map["pitching_overall_2024"].status == "degraded"
        assert status_map["pitching_split_vs_lhb_2025"].status == "cache"
        assert status_map["pitching_split_vs_lhb_2026"].status == "degraded"

    def test_legacy_pitching_wrapper_matches_status_return_data(self, monkeypatch):
        monkeypatch.setattr(
            "mlb.data.stats.fetch_pitching_splits_with_statuses",
            lambda season=2026, use_cache=True: ({"wrapped pitcher": {"source": "marcel_1yr"}}, [DataSourceStatus("x", "required", "run_wide", "fresh", "ok")]),
        )

        data = fetch_pitching_splits(season=2026, use_cache=True)

        assert data == {"wrapped pitcher": {"source": "marcel_1yr"}}

    def test_raises_when_no_usable_overall_pitching_stats_exist(self, monkeypatch):
        monkeypatch.setattr("mlb.data.stats._load_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._fetch_pitching_split_raw_with_status", lambda *args, **kwargs: ({}, "fresh"))
        monkeypatch.setattr("mlb.data.stats.fetch_pitching_season_rows", lambda season: (_ for _ in ()).throw(RuntimeError(f"{season} blocked")))

        with pytest.raises(RuntimeError, match="No usable pitching overall stats"):
            fetch_pitching_splits(season=2026, use_cache=False)


class TestRuntimeLeagueAverages:
    def test_computed_league_averages_fetch_overall_rows_once_on_cold_cache(self, monkeypatch):
        import mlb.data.stats as stats_mod

        batting_calls = {"count": 0}
        pitching_calls = {"count": 0}
        batting_rows = [
            {"Name": "Sample Batter", "PA": 100, "K%": 0.2, "BB%": 0.1, "HBP": 1, "H": 25, "2B": 5, "3B": 1, "HR": 4, "ID": 1, "Team": "ABC", "Bats": "R"},
        ]
        pitching_rows = [
            {"Name": "Sample Pitcher", "IP": 30.0, "BF": 120, "G": 10, "GS": 5, "H": 20, "2B": 3, "3B": 0, "HR": 2, "BB": 5, "HBP": 1, "SO": 28, "ID": 2, "Team": "ABC", "Throws": "R"},
        ]

        monkeypatch.setattr(stats_mod, "_COMPUTED_LEAGUE_AVERAGE_CACHE", {})
        monkeypatch.setattr(stats_mod, "_MATCHUP_LEAGUE_AVERAGE_CACHE", {})
        monkeypatch.setattr(stats_mod, "_load_computed_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr(stats_mod, "_load_raw_cache_payload", lambda *args, **kwargs: None)
        monkeypatch.setattr(stats_mod, "_save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr(stats_mod, "_save_computed_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            stats_mod,
            "fetch_batting_season_rows",
            lambda season: batting_calls.__setitem__("count", batting_calls["count"] + 1) or batting_rows,
        )
        monkeypatch.setattr(
            stats_mod,
            "fetch_pitching_season_rows",
            lambda season: pitching_calls.__setitem__("count", pitching_calls["count"] + 1) or pitching_rows,
        )
        monkeypatch.setattr(stats_mod, "_fetch_batting_split_raw", lambda *args, **kwargs: {})
        monkeypatch.setattr(
            stats_mod,
            "compute_matchup_league_averages_from_raw_splits",
            lambda overall_batters, vs_lhp_batters, vs_rhp_batters: {
                (Hand.LEFT, Hand.LEFT): _overall_league_avg_rates(),
                (Hand.LEFT, Hand.RIGHT): _overall_league_avg_rates(),
                (Hand.RIGHT, Hand.LEFT): _overall_league_avg_rates(),
                (Hand.RIGHT, Hand.RIGHT): _overall_league_avg_rates(),
            },
        )

        stats_mod.fetch_computed_league_averages(season=2026, use_cache=False)

        assert batting_calls["count"] == 1
        assert pitching_calls["count"] == 1

    def test_runtime_overall_league_averages_use_stale_cache_when_fresh_fetch_fails(self, monkeypatch, tmp_path):
        import json
        import mlb.data.stats as stats_mod

        payload = {
            "season": 2026,
            "rates": {"K": 0.21, "BB": 0.08, "HBP": 0.01, "1B": 0.14, "2B": 0.04, "3B": 0.005, "HR": 0.03, "OUT": 0.485},
            "matchup_rates": {"L_vs_R": {"K": 0.2, "BB": 0.08, "HBP": 0.01, "1B": 0.15, "2B": 0.04, "3B": 0.005, "HR": 0.03, "OUT": 0.485}},
        }
        stale_path = tmp_path / "computed_league_averages-2026.json"
        stale_path.write_text(json.dumps(payload))

        monkeypatch.setattr(stats_mod, "_COMPUTED_LEAGUE_AVERAGE_CACHE", {})
        monkeypatch.setattr(stats_mod, "_MATCHUP_LEAGUE_AVERAGE_CACHE", {})
        monkeypatch.setattr(stats_mod, "_load_computed_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr(stats_mod, "_computed_cache_path", lambda *args, **kwargs: stale_path)
        monkeypatch.setattr(
            stats_mod,
            "fetch_computed_league_averages",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        rates = fetch_runtime_overall_league_averages(season=2026, use_cache=True)

        assert rates["K"] == pytest.approx(0.21)
        assert rates["HR"] == pytest.approx(0.03)

    def test_runtime_overall_league_averages_fall_back_to_hardcoded_proxy(self, monkeypatch):
        import mlb.data.stats as stats_mod
        from pathlib import Path

        monkeypatch.setattr(stats_mod, "_COMPUTED_LEAGUE_AVERAGE_CACHE", {})
        monkeypatch.setattr(stats_mod, "_MATCHUP_LEAGUE_AVERAGE_CACHE", {})
        monkeypatch.setattr(stats_mod, "_load_computed_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr(stats_mod, "_computed_cache_path", lambda *args, **kwargs: Path("/tmp/does-not-exist-runtime-overall.json"))
        monkeypatch.setattr(
            stats_mod,
            "fetch_computed_league_averages",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        rates = fetch_runtime_overall_league_averages(season=2026, use_cache=True)

        expected = _overall_league_avg_rates()
        assert rates["K"] == pytest.approx(expected["K"])
        assert rates["BB"] == pytest.approx(expected["BB"])
        assert rates["HR"] == pytest.approx(expected["HR"])
        assert sum(rates[k] for k in ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT")) == pytest.approx(1.0)

    def test_batting_splits_use_runtime_overall_league_average_helper(self, monkeypatch):
        rows = [
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
                "ID": 123,
                "Team": "ABC",
                "Bats": "L",
            }
        ]

        monkeypatch.setattr("mlb.data.stats._load_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats.fetch_batting_season_rows", lambda season: rows)
        monkeypatch.setattr("mlb.data.stats._fetch_batting_split_raw", lambda *args, **kwargs: {})
        monkeypatch.setattr(
            "mlb.data.stats.fetch_computed_league_averages",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("strict helper should not be called")),
        )
        monkeypatch.setattr("mlb.data.stats.fetch_runtime_overall_league_averages", lambda *args, **kwargs: _overall_league_avg_rates())

        data = fetch_batting_splits(season=2026, use_cache=False)

        assert "sample batter" in data

    def test_pitching_splits_use_runtime_overall_league_average_helper(self, monkeypatch):
        rows = [
            {
                "Name": "Sample Pitcher",
                "IP": 25.0,
                "H": 20,
                "2B": 3,
                "3B": 0,
                "HR": 2,
                "BB": 5,
                "HBP": 1,
                "SO": 28,
                "ID": 1,
                "Team": "ABC",
                "Throws": "R",
            }
        ]

        monkeypatch.setattr("mlb.data.stats._load_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats._save_raw_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlb.data.stats.fetch_pitching_season_rows", lambda season: rows)
        monkeypatch.setattr("mlb.data.stats._fetch_pitching_split_raw", lambda *args, **kwargs: {})
        monkeypatch.setattr(
            "mlb.data.stats.fetch_computed_league_averages",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("strict helper should not be called")),
        )
        monkeypatch.setattr("mlb.data.stats.fetch_runtime_overall_league_averages", lambda *args, **kwargs: _overall_league_avg_rates())

        data = fetch_pitching_splits(season=2026, use_cache=False)

        assert "sample pitcher" in data

    def test_runtime_matchup_league_averages_fall_back_to_hardcoded_constants(self, monkeypatch):
        import mlb.data.stats as stats_mod
        from pathlib import Path
        from mlb.config import LEAGUE_AVERAGES

        monkeypatch.setattr(stats_mod, "_COMPUTED_LEAGUE_AVERAGE_CACHE", {})
        monkeypatch.setattr(stats_mod, "_MATCHUP_LEAGUE_AVERAGE_CACHE", {})
        monkeypatch.setattr(stats_mod, "_load_computed_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr(stats_mod, "_computed_cache_path", lambda *args, **kwargs: Path("/tmp/does-not-exist-runtime-matchup.json"))
        monkeypatch.setattr(
            stats_mod,
            "_compute_league_averages_and_matchups",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        matchups = stats_mod.fetch_runtime_league_averages(season=2026, use_cache=True)

        assert matchups[(Hand.LEFT, Hand.RIGHT)]["K"] == pytest.approx(LEAGUE_AVERAGES[(Hand.LEFT, Hand.RIGHT)]["K"])
        assert matchups[(Hand.RIGHT, Hand.LEFT)]["HR"] == pytest.approx(LEAGUE_AVERAGES[(Hand.RIGHT, Hand.LEFT)]["HR"])


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
    def test_preload_run_context_warms_shared_resources_once(self, monkeypatch):
        calls = {"league": 0, "bullpen": 0, "parks": []}

        monkeypatch.setattr("mlb.data.builder.ensure_runtime_league_averages", lambda season=2026: calls.__setitem__("league", calls["league"] + 1))
        monkeypatch.setattr("mlb.data.builder.fetch_team_bullpen_stats", lambda season=2026: calls.__setitem__("bullpen", calls["bullpen"] + 1) or {})
        monkeypatch.setattr(
            "mlb.data.builder.get_park_factors_with_status",
            lambda venue_name: (
                calls["parks"].append(venue_name) or ParkFactors(
                    venue_id=venue_name,
                    venue_name=venue_name,
                    factors_vs_lhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
                    factors_vs_rhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
                ),
                DataSourceStatus(
                    source_name="park_factors",
                    role="optional_enrichment",
                    scope="run_wide",
                    status="fresh",
                    detail=f"{venue_name} test park factors",
                ),
            ),
        )

        preload_run_context(
            [
                {"home_team": "Boston Red Sox", "venue": "Fenway Park"},
                {"home_team": "Boston Red Sox", "venue": "Fenway Park"},
                {"home_team": "Houston Astros", "venue": ""},
            ]
        )

        assert calls["league"] == 1
        assert calls["bullpen"] == 1
        assert sorted(calls["parks"]) == ["Daikin Park", "Fenway Park"]

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
            f"away batter {i}": dict(
                _raw_batter(name=f"Away Batter {i}", bats="R"),
                splits={"vs_lhp": {"pa": 40, "rates": dict(_raw_batter()["rates"], HR=0.08, OUT=0.7033333333333334), "source": "2026_split"}},
            )
            for i in range(1, 10)
        }
        batting_data.update(
            {
                f"home batter {i}": dict(
                    _raw_batter(name=f"Home Batter {i}", bats="L"),
                    splits={"vs_rhp": {"pa": 50, "rates": dict(_raw_batter()["rates"], BB=0.12, OUT=0.6733333333333333), "source": "2026_split"}},
                )
                for i in range(1, 10)
            }
        )
        pitching_data = {
            "away starter": _raw_pitcher(name="Away Starter", throws="R"),
            "home starter": _raw_pitcher(name="Home Starter", throws="L"),
        }

        monkeypatch.setattr("mlb.data.builder.fetch_game_lineup", lambda game_id: lineup)
        monkeypatch.setattr("mlb.data.builder.fangraphs_team_code", lambda team_name: "HOU" if team_name == "Houston Astros" else "")
        preload = _preload_fixture(
            bullpen_by_team={
                "": {
                    "player_id": "away-bullpen",
                    "name": "Away Team Bullpen",
                    "throws": "R",
                    "pa_against": 300,
                    "rates": _raw_pitcher()["rates"],
                    "avg_pitch_count": 120.0,
                    "source": "2026",
                },
                "HOU": {
                    "player_id": "hou-bullpen",
                    "name": "Houston Astros Bullpen",
                    "throws": "R",
                    "pa_against": 300,
                    "rates": _raw_pitcher()["rates"],
                    "avg_pitch_count": 120.0,
                    "source": "2026",
                },
            },
            park_factors_by_venue={
                "Daikin Park": ParkFactors(
                    venue_id="Daikin Park",
                    venue_name="Daikin Park",
                    factors_vs_lhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
                    factors_vs_rhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
                )
            },
            park_factor_status_by_venue={
                "Daikin Park": DataSourceStatus(
                    source_name="park_factors",
                    role="optional_enrichment",
                    scope="run_wide",
                    status="cache",
                    detail="park factors loaded from cache",
                )
            },
        )

        context = build_game_context(game_info, batting_data, pitching_data, preload)
        assert isinstance(context, GameContext)
        assert len(context.away_lineup.batting_order) == 9
        assert len(context.home_lineup.batting_order) == 9
        assert context.away_lineup.starting_pitcher.name == "Away Starter"
        assert context.home_lineup.starting_pitcher.name == "Home Starter"
        assert all(getattr(batter, "data_source", None) == "2026_split" for batter in context.away_lineup.batting_order)
        assert len(context.away_lineup.bullpen) == 1
        assert len(context.home_lineup.bullpen) == 1
        assert context.away_lineup.bullpen[0].name == "Away Team Bullpen"
        assert context.home_lineup.bullpen[0].name == "Houston Astros Bullpen"
        assert context.weather is not None
        assert context.weather.is_indoor is True
        status_names = {status.source_name for status in context.source_statuses}
        assert {"away_lineup", "home_lineup", "away_starter", "home_starter", "away_bullpen", "home_bullpen", "park_factors", "weather"} <= status_names
        weather_status = next(status for status in context.source_statuses if status.source_name == "weather")
        assert weather_status.status == "placeholder"
        park_status = next(status for status in context.source_statuses if status.source_name == "park_factors")
        assert park_status.status == "cache"
        away_starter_status = next(status for status in context.source_statuses if status.source_name == "away_starter")
        home_starter_status = next(status for status in context.source_statuses if status.source_name == "home_starter")
        assert away_starter_status.status == "degraded"
        assert home_starter_status.status == "degraded"

    def test_missing_preloaded_park_factors_raise_error(self, monkeypatch):
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
        lineup = {
            "away_batters": [
                {"name": f"Away Batter {i}", "id": f"a{i}", "batting_position": i, "bats": "R"}
                for i in range(1, 10)
            ],
            "home_batters": [
                {"name": f"Home Batter {i}", "id": f"h{i}", "batting_position": i, "bats": "L"}
                for i in range(1, 10)
            ],
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

        with pytest.raises(KeyError, match="Missing preloaded park factors"):
            build_game_context(
                game_info,
                batting_data,
                pitching_data,
                _preload_fixture(
                    bullpen_by_team={},
                    park_factors_by_venue={},
                    park_factor_status_by_venue={},
                ),
            )

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
        monkeypatch.setattr("mlb.data.builder.fangraphs_team_code", lambda team_name: "")
        preload = _preload_fixture(
            bullpen_by_team={},
            park_factors_by_venue={
                "Wrigley Field": ParkFactors(
                    venue_id="Wrigley Field",
                    venue_name="Wrigley Field",
                    factors_vs_lhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
                    factors_vs_rhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
                )
            },
            park_factor_status_by_venue={
                "Wrigley Field": DataSourceStatus(
                    source_name="park_factors",
                    role="optional_enrichment",
                    scope="run_wide",
                    status="hardcoded_fallback",
                    detail="park factors fell back to hardcoded table",
                )
            },
        )

        with caplog.at_level(logging.WARNING):
            context = build_game_context(game_info, batting_data, pitching_data, preload)

        assert context.away_lineup.batting_order[-1].name == "Away Batter 9"
        assert getattr(context.away_lineup.batting_order[-1], "data_source") == "league_avg"
        assert sum(context.away_lineup.batting_order[-1].rates.values()) == pytest.approx(1.0)
        assert any("Away Batter 9" in record.message and "league average" in record.message for record in caplog.records)
        assert context.away_lineup.bullpen[0].name == "Away Team Bullpen"
        away_bullpen_status = next(status for status in context.source_statuses if status.source_name == "away_bullpen")
        assert away_bullpen_status.status == "degraded"

    def test_roster_fallback_tags_lineup_and_starter_sources(self, monkeypatch):
        game_info = {
            "game_id": "1",
            "away_team": "Away Team",
            "away_team_id": "10",
            "home_team": "Home Team",
            "home_team_id": "20",
            "game_datetime": "2026-04-03T19:10:00Z",
            "venue": "Wrigley Field",
            "status": "Scheduled",
            "away_probable_pitcher": "",
            "home_probable_pitcher": "Home Probable",
        }
        roster = [
            {"name": f"Batter {i}", "id": f"b{i}", "position": "1B", "bats": "R", "throws": "R"}
            for i in range(1, 10)
        ] + [{"name": "Roster Arm", "id": "p1", "position": "P", "bats": "R", "throws": "L"}]
        batting_data = {f"batter {i}": _raw_batter(name=f"Batter {i}", bats="R") for i in range(1, 10)}
        pitching_data = {
            "roster arm": _raw_pitcher(name="Roster Arm", throws="L"),
            "home probable": _raw_pitcher(name="Home Probable", throws="R"),
        }

        monkeypatch.setattr("mlb.data.builder.fetch_game_lineup", lambda game_id: None)
        monkeypatch.setattr("mlb.data.lineups.fetch_team_roster", lambda team_id, season=2026: roster)
        monkeypatch.setattr("mlb.data.builder.fangraphs_team_code", lambda team_name: "")
        preload = _preload_fixture(
            bullpen_by_team={},
            park_factors_by_venue={
                "Wrigley Field": ParkFactors(
                    venue_id="Wrigley Field",
                    venue_name="Wrigley Field",
                    factors_vs_lhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
                    factors_vs_rhb={"HR": 1.0, "2B": 1.0, "3B": 1.0, "1B": 1.0, "BB": 1.0, "K": 1.0},
                )
            },
            park_factor_status_by_venue={
                "Wrigley Field": DataSourceStatus(
                    source_name="park_factors",
                    role="optional_enrichment",
                    scope="run_wide",
                    status="fresh",
                    detail="park factors from savant",
                )
            },
        )

        context = build_game_context(game_info, batting_data, pitching_data, preload)

        away_lineup_status = next(status for status in context.source_statuses if status.source_name == "away_lineup")
        home_lineup_status = next(status for status in context.source_statuses if status.source_name == "home_lineup")
        away_starter_status = next(status for status in context.source_statuses if status.source_name == "away_starter")
        home_starter_status = next(status for status in context.source_statuses if status.source_name == "home_starter")
        assert away_lineup_status.status == "degraded"
        assert home_lineup_status.status == "degraded"
        assert away_starter_status.status == "degraded"
        assert home_starter_status.status == "fresh"


class TestNormalizeName:
    def test_strips_accents(self):
        from mlb.utils.normalize import normalize_name
        assert normalize_name("Agustín Ramírez") == "agustin ramirez"

    def test_strips_accents_various(self):
        from mlb.utils.normalize import normalize_name
        assert normalize_name("José Caballero") == "jose caballero"
        assert normalize_name("Néstor Cortés") == "nestor cortes"

    def test_strips_accents_angel(self):
        from mlb.utils.normalize import normalize_name
        assert normalize_name("Ángel Zerpa") == "angel zerpa"

    def test_strips_periods_from_initials(self):
        from mlb.utils.normalize import normalize_name
        assert normalize_name("J.C. Escarra") == "jc escarra"

    def test_strips_trailing_jr_suffix(self):
        from mlb.utils.normalize import normalize_name
        assert normalize_name("Ronald Acuña Jr.") == "ronald acuna"

    def test_strips_trailing_jr_suffix_no_period(self):
        from mlb.utils.normalize import normalize_name
        assert normalize_name("Ronald Acuna Jr") == "ronald acuna"

    def test_strips_trailing_sr_suffix(self):
        from mlb.utils.normalize import normalize_name
        assert normalize_name("Ken Griffey Sr") == "ken griffey"

    def test_strips_trailing_ii_suffix(self):
        from mlb.utils.normalize import normalize_name
        assert normalize_name("C.J. Abrams II") == "cj abrams"

    def test_strips_trailing_iii_suffix(self):
        from mlb.utils.normalize import normalize_name
        assert normalize_name("Cal Ripken III") == "cal ripken"

    def test_strips_trailing_iv_suffix(self):
        from mlb.utils.normalize import normalize_name
        assert normalize_name("Player Name IV") == "player name"

    def test_iv_not_stripped_from_middle_of_name(self):
        from mlb.utils.normalize import normalize_name
        assert normalize_name("Ivan Rodriguez") == "ivan rodriguez"

    def test_lowercases_and_collapses_whitespace(self):
        from mlb.utils.normalize import normalize_name
        assert normalize_name("  Aaron  Judge  ") == "aaron judge"

    def test_multi_word_last_name(self):
        from mlb.utils.normalize import normalize_name
        assert normalize_name("José De La Cruz") == "jose de la cruz"

    def test_empty_string(self):
        from mlb.utils.normalize import normalize_name
        assert normalize_name("") == ""

    def test_canonical_matches_stats_and_builder(self):
        from mlb.utils.normalize import normalize_name
        from mlb.data.stats import _normalize_name as stats_norm
        from mlb.data.builder import _normalize_name as builder_norm
        names = ["Agustín Ramírez", "Ronald Acuña Jr.", "J.C. Escarra", "Ángel Zerpa"]
        for name in names:
            assert normalize_name(name) == stats_norm(name) == builder_norm(name)


class TestLeagueAverageFallbacks:
    def test_league_average_batter_uses_pitcher_hand(self):
        from mlb.data.builder import _league_average_batter
        from mlb.config import LEAGUE_AVERAGES

        batter = _league_average_batter("Switch Sample", "S", "L")
        assert batter["rates"] == LEAGUE_AVERAGES[(Hand.RIGHT, Hand.LEFT)]

    def test_extract_batter_rates_uses_count_columns_when_percentages_missing(self):
        from mlb.data.stats import _extract_batter_rates

        rates = _extract_batter_rates(
            {
                "PA": 100,
                "H": 25,
                "2B": 5,
                "3B": 1,
                "HR": 4,
                "HBP": 2,
                "SO": 20,
                "BB": 10,
            }
        )

        assert rates["K"] == pytest.approx(0.20)
        assert rates["BB"] == pytest.approx(0.10)
