import pytest

from mlb.engine.probabilities import odds_ratio


class TestOddsRatio:
    """Tom Tango Odds Ratio method: combine batter + pitcher rates."""

    def test_hand_calculated_example(self):
        """Batter HR=0.04, pitcher HR=0.025, league=0.03 → ~0.0334."""
        result = odds_ratio(0.04, 0.025, 0.03)
        assert result == pytest.approx(0.0334, abs=0.001)

    def test_zero_batter_returns_zero(self):
        assert odds_ratio(0.0, 0.025, 0.03) == 0.0

    def test_zero_pitcher_returns_zero(self):
        assert odds_ratio(0.04, 0.0, 0.03) == 0.0

    def test_zero_league_returns_zero(self):
        assert odds_ratio(0.04, 0.025, 0.0) == 0.0

    def test_near_one_does_not_crash(self):
        result = odds_ratio(0.999, 0.025, 0.03)
        assert 0.0 <= result <= 1.0

    def test_one_input_clamped(self):
        """Input of exactly 1.0 is clamped to 0.999 — no crash."""
        result = odds_ratio(1.0, 0.5, 0.5)
        assert 0.0 <= result <= 1.0

    def test_symmetry_all_equal_league(self):
        """If batter == pitcher == league, result == league."""
        for p in [0.03, 0.15, 0.45, 0.005]:
            result = odds_ratio(p, p, p)
            assert result == pytest.approx(p, abs=1e-9)

    def test_result_always_in_valid_range(self):
        """Output clamped to [0.0, 1.0]."""
        result = odds_ratio(0.04, 0.025, 0.03)
        assert 0.0 <= result <= 1.0
