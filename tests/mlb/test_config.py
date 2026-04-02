from mlb.config import LEAGUE_AVERAGES, OUTCOMES, Hand, Outcome


class TestLeagueAverages:
    """Verify league-average rate tables are well-formed."""

    def test_all_matchups_present(self):
        expected = {
            (Hand.LEFT, Hand.RIGHT),
            (Hand.RIGHT, Hand.LEFT),
            (Hand.LEFT, Hand.LEFT),
            (Hand.RIGHT, Hand.RIGHT),
        }
        assert set(LEAGUE_AVERAGES.keys()) == expected

    def test_each_matchup_sums_to_one(self):
        for key, rates in LEAGUE_AVERAGES.items():
            total = sum(rates.values())
            assert abs(total - 1.0) < 0.001, (
                f"{key}: rates sum to {total:.4f}"
            )

    def test_each_matchup_has_all_outcomes(self):
        for key, rates in LEAGUE_AVERAGES.items():
            assert set(rates.keys()) == set(OUTCOMES), (
                f"{key}: missing or extra outcomes"
            )

    def test_all_rates_non_negative(self):
        for key, rates in LEAGUE_AVERAGES.items():
            for outcome, rate in rates.items():
                assert rate >= 0, f"{key}[{outcome}] = {rate}"


class TestOutcomes:
    """Verify OUTCOMES list matches Outcome enum."""

    def test_outcomes_match_enum(self):
        assert set(OUTCOMES) == {o.value for o in Outcome}

    def test_outcomes_length(self):
        assert len(OUTCOMES) == 8


class TestHandEnum:
    """Verify Hand enum values and str interoperability."""

    def test_values(self):
        assert Hand.LEFT == 'L'
        assert Hand.RIGHT == 'R'
        assert Hand.SWITCH == 'S'

    def test_str_lookup(self):
        assert Hand('L') == Hand.LEFT
        assert Hand('R') == Hand.RIGHT
