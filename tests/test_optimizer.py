import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_optimizer import build_config, validate_config, TARGETS


class _MockTrial:
    """Minimal mock that returns midpoint values for suggest_* calls."""

    def suggest_int(self, name, low, high):
        return (low + high) // 2

    def suggest_float(self, name, low, high):
        return (low + high) / 2

    def set_user_attr(self, key, value):
        pass


class TestBuildConfig:
    def test_contains_search_params(self):
        config = build_config(_MockTrial())
        assert "sample_size" in config
        assert "decay_factor" in config
        assert "min_z_threshold" in config
        assert "z_medium" in config
        assert "z_high" in config
        assert "home_boost" in config

    def test_contains_fixed_params(self):
        config = build_config(_MockTrial())
        assert config["min_home_away_games"] == 4
        assert config["vig_win"] == 0.9091
        assert config["vig_risk"] == 1.0

    def test_sample_size_is_int(self):
        config = build_config(_MockTrial())
        assert isinstance(config["sample_size"], int)


class TestValidateConfig:
    def test_valid_config(self):
        config = {"min_z_threshold": 0.5, "z_medium": 0.8, "z_high": 1.5}
        assert validate_config(config) is True

    def test_z_medium_ge_z_high_invalid(self):
        config = {"min_z_threshold": 0.5, "z_medium": 2.0, "z_high": 1.5}
        assert validate_config(config) is False

    def test_z_medium_eq_z_high_invalid(self):
        config = {"min_z_threshold": 0.5, "z_medium": 1.5, "z_high": 1.5}
        assert validate_config(config) is False

    def test_min_z_gt_z_medium_invalid(self):
        config = {"min_z_threshold": 1.5, "z_medium": 0.8, "z_high": 2.0}
        assert validate_config(config) is False

    def test_min_z_eq_z_medium_valid(self):
        config = {"min_z_threshold": 0.8, "z_medium": 0.8, "z_high": 1.5}
        assert validate_config(config) is True


class TestTargets:
    def test_known_targets(self):
        assert "beat_rate" in TARGETS
        assert "avg_miss" in TARGETS
        assert "roi" in TARGETS
        assert "advantage" in TARGETS

    def test_beat_rate_maximizes(self):
        assert TARGETS["beat_rate"]["direction"] == "maximize"

    def test_avg_miss_minimizes(self):
        assert TARGETS["avg_miss"]["direction"] == "minimize"

    def test_roi_maximizes(self):
        assert TARGETS["roi"]["direction"] == "maximize"

    def test_advantage_maximizes(self):
        assert TARGETS["advantage"]["direction"] == "maximize"
