import math
from models.v2_current import weighted_mean, weighted_variance, normal_cdf
from models.v2_current import compute_my_line, compute_confidence_and_ev, compute_recommendation


class TestWeightedMean:
    def test_single_item(self):
        items = [{"date": "2026-01-10", "value": 100}]
        assert weighted_mean(items, "2026-01-10", 0.96) == 100.0

    def test_same_day_is_simple_mean(self):
        items = [
            {"date": "2026-01-10", "value": 100},
            {"date": "2026-01-10", "value": 120},
        ]
        assert weighted_mean(items, "2026-01-10", 0.96) == 110.0

    def test_decay_favors_recent(self):
        items = [
            {"date": "2026-01-10", "value": 100},  # w = 1.0
            {"date": "2026-01-08", "value": 120},  # w = 0.96^2
        ]
        result = weighted_mean(items, "2026-01-10", 0.96)
        w1, w2 = 1.0, 0.96 ** 2
        expected = (100 * w1 + 120 * w2) / (w1 + w2)
        assert abs(result - expected) < 1e-10
        assert result < 110.0  # recent item (100) pulls mean down

    def test_empty_returns_none(self):
        assert weighted_mean([], "2026-01-10", 0.96) is None


class TestWeightedVariance:
    def test_equal_weights(self):
        items = [
            {"date": "2026-01-10", "value": 100},
            {"date": "2026-01-10", "value": 120},
        ]
        result = weighted_variance(items, "2026-01-10", 0.96, 110.0)
        # Both w=1: ((100-110)^2 + (120-110)^2) / 2 = 100
        assert result == 100.0

    def test_single_item_returns_none(self):
        items = [{"date": "2026-01-10", "value": 100}]
        assert weighted_variance(items, "2026-01-10", 0.96, 100.0) is None

    def test_empty_returns_none(self):
        assert weighted_variance([], "2026-01-10", 0.96, 100.0) is None

    def test_none_mean_returns_none(self):
        items = [
            {"date": "2026-01-10", "value": 100},
            {"date": "2026-01-10", "value": 120},
        ]
        assert weighted_variance(items, "2026-01-10", 0.96, None) is None


class TestNormalCdf:
    def test_zero(self):
        assert abs(normal_cdf(0) - 0.5) < 1e-7

    def test_positive_1(self):
        assert abs(normal_cdf(1.0) - 0.8413) < 0.001

    def test_negative_1(self):
        assert abs(normal_cdf(-1.0) - 0.1587) < 0.001

    def test_196(self):
        assert abs(normal_cdf(1.96) - 0.975) < 0.001

    def test_large(self):
        assert abs(normal_cdf(3.0) - 0.9987) < 0.001

    def test_symmetry(self):
        assert abs(normal_cdf(1.5) + normal_cdf(-1.5) - 1.0) < 1e-7


def _make_cfg(**overrides):
    cfg = {
        "sample_size": 10,
        "decay_factor": 0.96,
        "min_z_threshold": 0.5,
        "z_medium": 0.8,
        "z_high": 1.5,
        "home_boost": 1.5,
        "min_home_away_games": 4,
        "vig_win": 0.9091,
        "vig_risk": 1.0,
    }
    cfg.update(overrides)
    return cfg


class TestComputeMyLine:
    def test_basic_projection(self):
        home_games = [
            {"date": "2026-01-09", "pointsScored": 110, "pointsAllowed": 100,
             "isHome": True, "wentToOT": False},
            {"date": "2026-01-09", "pointsScored": 120, "pointsAllowed": 105,
             "isHome": False, "wentToOT": False},
        ]
        away_games = [
            {"date": "2026-01-09", "pointsScored": 105, "pointsAllowed": 115,
             "isHome": False, "wentToOT": False},
            {"date": "2026-01-09", "pointsScored": 100, "pointsAllowed": 110,
             "isHome": True, "wentToOT": False},
        ]
        result = compute_my_line(home_games, away_games, "2026-01-10", _make_cfg())
        # All same-day so equal weights. homeOff=115, homeDef=102.5,
        # awayOff=102.5, awayDef=112.5. < 4 home/away splits -> default boost 1.5
        # projHome = (115+112.5)/2 + 1.5/2 = 114.5
        # projAway = (102.5+102.5)/2 - 1.5/2 = 101.75
        # myLine = 216.25
        assert abs(result["my_line"] - 216.25) < 0.01
        assert abs(result["proj_home"] - 114.5) < 0.01
        assert abs(result["proj_away"] - 101.75) < 0.01
        assert result["sd_total"] is not None
        assert result["components"]["home_boost"] == 1.5

    def test_empty_returns_none(self):
        result = compute_my_line([], [], "2026-01-10", _make_cfg())
        assert result["my_line"] is None
        assert result["sd_total"] is None

    def test_dynamic_home_boost(self):
        # 4 isHome=True (scoring 110) + 4 isHome=False (scoring 100)
        # -> dynamic boost = 110 - 100 = 10
        home_games = (
            [{"date": "2026-01-09", "pointsScored": 110, "pointsAllowed": 100,
              "isHome": True, "wentToOT": False}] * 4
            + [{"date": "2026-01-09", "pointsScored": 100, "pointsAllowed": 110,
                "isHome": False, "wentToOT": False}] * 4
        )
        away_games = [
            {"date": "2026-01-09", "pointsScored": 105, "pointsAllowed": 108,
             "isHome": False, "wentToOT": False},
            {"date": "2026-01-09", "pointsScored": 105, "pointsAllowed": 108,
             "isHome": False, "wentToOT": False},
        ]
        result = compute_my_line(home_games, away_games, "2026-01-10", _make_cfg())
        assert result["components"]["home_boost"] == 10.0

    def test_sd_total_computed(self):
        home_games = [
            {"date": "2026-01-09", "pointsScored": 100, "pointsAllowed": 110,
             "isHome": True, "wentToOT": False},
            {"date": "2026-01-09", "pointsScored": 120, "pointsAllowed": 90,
             "isHome": False, "wentToOT": False},
        ]
        away_games = [
            {"date": "2026-01-09", "pointsScored": 105, "pointsAllowed": 115,
             "isHome": False, "wentToOT": False},
            {"date": "2026-01-09", "pointsScored": 95, "pointsAllowed": 105,
             "isHome": True, "wentToOT": False},
        ]
        result = compute_my_line(home_games, away_games, "2026-01-10", _make_cfg())
        assert result["sd_total"] is not None
        assert result["sd_total"] > 0


class TestComputeConfidenceAndEV:
    def test_high_confidence(self):
        result = compute_confidence_and_ev(15.0, 8.0, _make_cfg())
        assert result["confidence"] == "HIGH"
        assert result["z_score"] == round(15.0 / 8.0, 3)

    def test_medium_confidence(self):
        result = compute_confidence_and_ev(8.0, 8.0, _make_cfg())
        assert result["confidence"] == "MEDIUM"  # |z|=1.0, >= 0.8

    def test_low_confidence(self):
        result = compute_confidence_and_ev(4.0, 8.0, _make_cfg())
        assert result["confidence"] == "LOW"  # |z|=0.5, < 0.8

    def test_negative_discrepancy(self):
        result = compute_confidence_and_ev(-12.0, 8.0, _make_cfg())
        assert result["z_score"] < 0
        assert result["confidence"] == "HIGH"  # |z|=1.5

    def test_none_sd(self):
        result = compute_confidence_and_ev(10.0, None, _make_cfg())
        assert result["z_score"] is None
        assert result["confidence"] is None

    def test_zero_sd(self):
        result = compute_confidence_and_ev(10.0, 0, _make_cfg())
        assert result["z_score"] is None

    def test_ev_positive_for_high_z(self):
        result = compute_confidence_and_ev(15.0, 8.0, _make_cfg())
        assert result["expected_value"] > 0


class TestComputeRecommendation:
    def test_over(self):
        assert compute_recommendation(230, 220, 1.0, 5.0, _make_cfg()) == "O"

    def test_under(self):
        assert compute_recommendation(210, 220, -1.0, 5.0, _make_cfg()) == "U"

    def test_no_bet_low_z(self):
        assert compute_recommendation(225, 220, 0.3, 5.0, _make_cfg()) == "NO_BET"

    def test_no_bet_negative_ev(self):
        assert compute_recommendation(225, 220, 0.6, -1.0, _make_cfg()) == "NO_BET"

    def test_none_line(self):
        assert compute_recommendation(None, 220, 1.0, 5.0, _make_cfg()) is None

    def test_push(self):
        assert compute_recommendation(220, 220, 1.0, 5.0, _make_cfg()) == "P"
