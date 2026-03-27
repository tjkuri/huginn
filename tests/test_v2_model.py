import math
from models.v2_current import weighted_mean, weighted_variance, normal_cdf


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
