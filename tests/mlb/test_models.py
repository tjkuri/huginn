import logging

import pytest

from mlb.config import Hand, Outcome, WindDirection
from mlb.data.models import (
    BaseState,
    BatterStats,
    GameState,
    PAResult,
    ParkFactors,
    PitcherStats,
    PlayerSimStats,
    SimulatedGame,
    SimulationResult,
    Weather,
    resolve_batter_hand,
)


def _valid_rates():
    """Return a valid set of PA outcome rates summing to 1.0."""
    return {
        'K': 0.207, 'BB': 0.088, 'HBP': 0.012,
        '1B': 0.153, '2B': 0.048, '3B': 0.005, 'HR': 0.035,
        'OUT': 0.452,
    }


# ── BatterStats ──────────────────────────────────────────────────────────────

class TestBatterStats:
    def test_valid_rates_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            b = BatterStats('p1', 'Good Hitter', Hand.RIGHT, 500, _valid_rates())
        assert b.name == 'Good Hitter'
        assert b.bats == Hand.RIGHT
        assert len(caplog.records) == 0

    def test_rates_bad_sum_warns(self, caplog):
        rates = _valid_rates()
        rates['HR'] = 0.200  # pushes sum well above 1.0
        with caplog.at_level(logging.WARNING):
            BatterStats('p1', 'Bad Sum', Hand.RIGHT, 500, rates)
        assert any('rates sum to' in r.message for r in caplog.records)

    def test_missing_outcome_key_warns(self, caplog):
        rates = _valid_rates()
        del rates['HR']
        with caplog.at_level(logging.WARNING):
            BatterStats('p1', 'Missing Key', Hand.RIGHT, 500, rates)
        assert any('missing' in r.message for r in caplog.records)

    def test_extra_outcome_key_warns(self, caplog):
        rates = _valid_rates()
        rates['BUNT'] = 0.001
        with caplog.at_level(logging.WARNING):
            BatterStats('p1', 'Extra Key', Hand.LEFT, 300, rates)
        assert any('unexpected' in r.message for r in caplog.records)

    def test_tolerance_within_threshold(self, caplog):
        """Rates summing to 0.99 (within 0.02 tolerance) should not warn."""
        rates = _valid_rates()
        rates['OUT'] = 0.442  # total = 0.990, within 0.02 of 1.0
        with caplog.at_level(logging.WARNING):
            BatterStats('p1', 'Close Enough', Hand.RIGHT, 500, rates)
        sum_warnings = [r for r in caplog.records if 'rates sum to' in r.message]
        assert len(sum_warnings) == 0


# ── PitcherStats ─────────────────────────────────────────────────────────────

class TestPitcherStats:
    def test_valid_rates_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            p = PitcherStats('p1', 'Ace', Hand.LEFT, 600, _valid_rates(), 95.0)
        assert p.throws == Hand.LEFT
        assert p.avg_pitch_count == 95.0
        assert len(caplog.records) == 0

    def test_rates_bad_sum_warns(self, caplog):
        rates = _valid_rates()
        rates['K'] = 0.500
        with caplog.at_level(logging.WARNING):
            PitcherStats('p1', 'Wild', Hand.RIGHT, 400, rates)
        assert any('rates sum to' in r.message for r in caplog.records)

    def test_default_pitch_count(self):
        p = PitcherStats('p1', 'Default', Hand.RIGHT, 400, _valid_rates())
        assert p.avg_pitch_count == 0.0


# ── resolve_batter_hand ──────────────────────────────────────────────────────

class TestResolveBatterHand:
    def test_switch_vs_righty_bats_left(self):
        assert resolve_batter_hand(Hand.SWITCH, Hand.RIGHT) == Hand.LEFT

    def test_switch_vs_lefty_bats_right(self):
        assert resolve_batter_hand(Hand.SWITCH, Hand.LEFT) == Hand.RIGHT

    def test_lefty_unchanged(self):
        assert resolve_batter_hand(Hand.LEFT, Hand.RIGHT) == Hand.LEFT
        assert resolve_batter_hand(Hand.LEFT, Hand.LEFT) == Hand.LEFT

    def test_righty_unchanged(self):
        assert resolve_batter_hand(Hand.RIGHT, Hand.LEFT) == Hand.RIGHT
        assert resolve_batter_hand(Hand.RIGHT, Hand.RIGHT) == Hand.RIGHT


# ── BaseState ────────────────────────────────────────────────────────────────

class TestBaseState:
    def test_defaults_empty(self):
        bs = BaseState()
        assert not bs.first and not bs.second and not bs.third

    def test_frozen(self):
        bs = BaseState(first=True)
        with pytest.raises(AttributeError):
            bs.first = False  # type: ignore[misc]

    def test_equality(self):
        assert BaseState(first=True) == BaseState(first=True)
        assert BaseState() != BaseState(third=True)


# ── ParkFactors ──────────────────────────────────────────────────────────────

class TestParkFactors:
    def _make_park(self):
        return ParkFactors(
            'v1', 'Coors Field',
            factors_vs_lhb={'HR': 1.20, '1B': 1.05, 'OUT': 0.92},
            factors_vs_rhb={'HR': 1.10, '1B': 1.03, 'OUT': 0.95},
        )

    def test_get_factors_lhb(self):
        pf = self._make_park()
        assert pf.get_factors(Hand.LEFT) == {'HR': 1.20, '1B': 1.05, 'OUT': 0.92}

    def test_get_factors_rhb(self):
        pf = self._make_park()
        assert pf.get_factors(Hand.RIGHT) == {'HR': 1.10, '1B': 1.03, 'OUT': 0.95}


# ── Weather ──────────────────────────────────────────────────────────────────

class TestWeather:
    def test_outdoor_default(self):
        w = Weather(72.0, 5.0, WindDirection.OUT_TO_CF, 50.0)
        assert w.is_indoor is False

    def test_indoor_explicit(self):
        w = Weather(72.0, 0.0, WindDirection.CALM, 50.0, is_indoor=True)
        assert w.is_indoor is True

    def test_wind_direction_enum(self):
        w = Weather(80.0, 12.0, WindDirection.CROSS, 65.0)
        assert w.wind_direction == 'cross'


# ── GameState ────────────────────────────────────────────────────────────────

class TestGameState:
    def test_defaults(self):
        gs = GameState()
        assert gs.inning == 1
        assert gs.is_top is True
        assert gs.outs == 0
        assert gs.bases == BaseState()
        assert gs.away_score == 0
        assert gs.home_score == 0
        assert gs.away_batting_index == 0
        assert gs.home_batting_index == 0
        assert gs.away_pitch_count == 0
        assert gs.home_pitch_count == 0

    def test_mutable(self):
        gs = GameState()
        gs.outs = 2
        gs.away_score = 3
        assert gs.outs == 2
        assert gs.away_score == 3


# ── PAResult ─────────────────────────────────────────────────────────────────

class TestPAResult:
    def test_construction(self):
        pa = PAResult(
            outcome=Outcome.HR,
            batter_id='b1',
            pitcher_id='p1',
            inning=3,
            runners_before=BaseState(first=True, second=True),
            runs_scored=3,
        )
        assert pa.outcome == Outcome.HR
        assert pa.runs_scored == 3
        assert pa.runners_before.first is True


# ── PlayerSimStats ───────────────────────────────────────────────────────────

class TestPlayerSimStats:
    def test_construction(self):
        pss = PlayerSimStats(
            player_id='b1', name='Slugger',
            pa_per_game=4.2, hits_per_game=1.1, hr_per_game=0.3,
            bb_per_game=0.5, k_per_game=1.2, runs_per_game=0.8,
        )
        assert pss.hr_per_game == 0.3


# ── SimulationResult ─────────────────────────────────────────────────────────

class TestSimulationResult:
    def test_construction(self):
        sr = SimulationResult(
            game_id='g1', n_simulations=10000,
            away_team='NYY', home_team='BOS',
            away_runs_mean=4.5, away_runs_std=2.1,
            home_runs_mean=4.8, home_runs_std=2.3,
            total_runs_mean=9.3, total_runs_std=3.0,
            home_win_pct=0.54, away_win_pct=0.46,
            player_stats={},
        )
        assert sr.home_win_pct + sr.away_win_pct == pytest.approx(1.0)
