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


from mlb.config import WindDirection
from mlb.data.models import Weather
from mlb.engine.probabilities import apply_weather_adjustments, normalize


def _base_rates():
    """Baseline rates for weather tests."""
    return {
        'K': 0.220, 'BB': 0.080, 'HBP': 0.012,
        '1B': 0.150, '2B': 0.048, '3B': 0.005, 'HR': 0.035,
        'OUT': 0.450,
    }


class TestApplyWeatherAdjustments:
    def test_hot_day_increases_hr(self):
        """90°F → HR should increase vs 70°F baseline."""
        rates = _base_rates()
        weather = Weather(90.0, 0.0, WindDirection.CALM, 50.0)
        result = apply_weather_adjustments(rates, weather)
        assert result['HR'] > rates['HR']

    def test_cold_day_decreases_hr(self):
        """50°F → HR should decrease vs 70°F baseline."""
        rates = _base_rates()
        weather = Weather(50.0, 0.0, WindDirection.CALM, 50.0)
        result = apply_weather_adjustments(rates, weather)
        assert result['HR'] < rates['HR']

    def test_baseline_temp_no_change(self):
        """70°F (baseline) → no temperature adjustment."""
        rates = _base_rates()
        weather = Weather(70.0, 0.0, WindDirection.CALM, 50.0)
        result = apply_weather_adjustments(rates, weather)
        assert result['HR'] == pytest.approx(rates['HR'], abs=1e-9)
        assert result['2B'] == pytest.approx(rates['2B'], abs=1e-9)

    def test_wind_out_increases_hr(self):
        """15 mph wind out to CF → HR increases by known amount."""
        rates = _base_rates()
        weather = Weather(70.0, 15.0, WindDirection.OUT_TO_CF, 50.0)
        result = apply_weather_adjustments(rates, weather)
        assert result['HR'] == pytest.approx(0.035 * 1.12, abs=0.0001)
        assert result['2B'] == pytest.approx(0.048 * 1.045, abs=0.0001)

    def test_wind_in_decreases_hr(self):
        """15 mph wind in from CF → HR decreases."""
        rates = _base_rates()
        weather = Weather(70.0, 15.0, WindDirection.IN_FROM_CF, 50.0)
        result = apply_weather_adjustments(rates, weather)
        assert result['HR'] < rates['HR']

    def test_cross_wind_no_change(self):
        """Cross wind → no wind adjustment."""
        rates = _base_rates()
        weather = Weather(70.0, 15.0, WindDirection.CROSS, 50.0)
        result = apply_weather_adjustments(rates, weather)
        assert result['HR'] == pytest.approx(rates['HR'], abs=1e-9)

    def test_indoor_skips_all_adjustments(self):
        """is_indoor=True → no changes, even with extreme conditions."""
        rates = _base_rates()
        weather = Weather(110.0, 30.0, WindDirection.OUT_TO_CF, 90.0, is_indoor=True)
        result = apply_weather_adjustments(rates, weather)
        for outcome in rates:
            assert result[outcome] == rates[outcome]

    def test_none_weather_returns_copy(self):
        """None weather → return unchanged copy."""
        rates = _base_rates()
        result = apply_weather_adjustments(rates, None)
        assert result == rates
        assert result is not rates  # must be a copy

    def test_extreme_cold_clamps_above_minimum(self):
        """Very low rates + extreme cold → clamped to 0.001 minimum."""
        rates = {'HR': 0.002, 'K': 0.20, 'BB': 0.08, 'HBP': 0.01,
                 '1B': 0.15, '2B': 0.002, '3B': 0.001, 'OUT': 0.45}
        weather = Weather(-20.0, 0.0, WindDirection.CALM, 50.0)
        result = apply_weather_adjustments(rates, weather)
        for outcome in result:
            assert result[outcome] >= 0.001

    def test_does_not_mutate_input(self):
        """Input dict should not be modified."""
        rates = _base_rates()
        original = dict(rates)
        weather = Weather(90.0, 15.0, WindDirection.OUT_TO_CF, 50.0)
        apply_weather_adjustments(rates, weather)
        assert rates == original


class TestNormalize:
    def test_sums_to_one(self):
        """Rates summing to 1.3 get normalized to 1.0."""
        rates = {'K': 0.26, 'BB': 0.10, 'HBP': 0.013,
                 '1B': 0.19, '2B': 0.065, '3B': 0.006, 'HR': 0.046,
                 'OUT': 0.620}
        result = normalize(rates)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_preserves_ratios(self):
        """Relative proportions are maintained."""
        rates = {'A': 0.2, 'B': 0.4, 'C': 0.7}
        result = normalize(rates)
        assert result['B'] / result['A'] == pytest.approx(2.0)
        assert result['C'] / result['A'] == pytest.approx(3.5)

    def test_already_normalized(self):
        """Rates that already sum to 1.0 are unchanged."""
        rates = {'K': 0.25, 'OUT': 0.75}
        result = normalize(rates)
        assert result['K'] == pytest.approx(0.25)
        assert result['OUT'] == pytest.approx(0.75)

    def test_zero_sum_uniform_fallback(self):
        """All-zero rates → uniform distribution fallback."""
        rates = {'A': 0.0, 'B': 0.0, 'C': 0.0}
        result = normalize(rates)
        assert result == pytest.approx({'A': 1 / 3, 'B': 1 / 3, 'C': 1 / 3})
