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


from mlb.config import Hand, Outcome, LEAGUE_AVERAGES
from mlb.data.models import BatterStats, PitcherStats
from mlb.engine.probabilities import compute_matchup_rates, apply_park_factors


def _sample_batter():
    """LHB with rates summing to 1.0."""
    return BatterStats('b1', 'Sample Batter', Hand.LEFT, 500, {
        'K': 0.200, 'BB': 0.090, 'HBP': 0.010,
        '1B': 0.155, '2B': 0.050, '3B': 0.005, 'HR': 0.040,
        'OUT': 0.450,
    })


def _sample_pitcher():
    """RHP with rates summing to 1.0."""
    return PitcherStats('p1', 'Sample Pitcher', Hand.RIGHT, 600, {
        'K': 0.230, 'BB': 0.075, 'HBP': 0.010,
        '1B': 0.140, '2B': 0.045, '3B': 0.003, 'HR': 0.025,
        'OUT': 0.472,
    })


class TestComputeMatchupRates:
    def test_all_outcomes_present(self):
        league = LEAGUE_AVERAGES[(Hand.LEFT, Hand.RIGHT)]
        rates = compute_matchup_rates(_sample_batter(), _sample_pitcher(), league)
        assert set(rates.keys()) == {o.value for o in Outcome}

    def test_all_rates_positive(self):
        league = LEAGUE_AVERAGES[(Hand.LEFT, Hand.RIGHT)]
        rates = compute_matchup_rates(_sample_batter(), _sample_pitcher(), league)
        for outcome, rate in rates.items():
            assert rate > 0, f"{outcome} should be positive"

    def test_symmetry_equals_league(self):
        """Batter and pitcher both at league average → output equals league avg."""
        league = LEAGUE_AVERAGES[(Hand.LEFT, Hand.RIGHT)]
        batter = BatterStats('b1', 'Avg', Hand.LEFT, 500, dict(league))
        pitcher = PitcherStats('p1', 'Avg', Hand.RIGHT, 600, dict(league))
        rates = compute_matchup_rates(batter, pitcher, league)
        for outcome in Outcome:
            assert rates[outcome.value] == pytest.approx(
                league[outcome.value], abs=1e-9
            ), f"{outcome.value} should match league average"


class TestApplyParkFactors:
    def test_hr_boosted(self):
        """HR rate 0.03 with park factor 1.15 → 0.0345."""
        rates = {'HR': 0.03, '1B': 0.15, 'OUT': 0.45, 'K': 0.20,
                 'BB': 0.08, 'HBP': 0.01, '2B': 0.05, '3B': 0.004}
        pf = {'HR': 1.15, '1B': 1.05}
        result = apply_park_factors(rates, pf)
        assert result['HR'] == pytest.approx(0.0345, abs=0.0001)
        assert result['1B'] == pytest.approx(0.1575, abs=0.0001)

    def test_missing_factor_uses_default(self):
        """Outcomes without a park factor use 1.0 (no change)."""
        rates = {'BB': 0.08, 'HBP': 0.01, 'OUT': 0.45, 'K': 0.20,
                 '1B': 0.15, '2B': 0.05, '3B': 0.004, 'HR': 0.03}
        pf = {'HR': 1.15}
        result = apply_park_factors(rates, pf)
        assert result['BB'] == 0.08

    def test_hbp_never_adjusted(self):
        """HBP is not park-dependent — ignored even if factor provided."""
        rates = {'HBP': 0.012, 'HR': 0.03, '1B': 0.15, 'OUT': 0.45,
                 'K': 0.20, 'BB': 0.08, '2B': 0.05, '3B': 0.004}
        pf = {'HBP': 1.50}
        result = apply_park_factors(rates, pf)
        assert result['HBP'] == 0.012

    def test_out_never_adjusted(self):
        """OUT is derived from normalization — not directly park-adjusted."""
        rates = {'OUT': 0.45, 'HR': 0.03, '1B': 0.15, 'HBP': 0.01,
                 'K': 0.20, 'BB': 0.08, '2B': 0.05, '3B': 0.004}
        pf = {'OUT': 0.80}
        result = apply_park_factors(rates, pf)
        assert result['OUT'] == 0.45
