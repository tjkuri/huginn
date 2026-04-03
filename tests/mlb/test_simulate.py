"""Tests for mlb.engine.simulate."""
import numpy as np
import pytest

from mlb.config import Hand, Outcome
from mlb.data.models import BatterStats
from mlb.engine.simulate import resolve_pa_outcome


class TestResolvePAOutcome:
    """resolve_pa_outcome: sample one outcome from a probability table."""

    def test_deterministic_hr(self):
        """HR=1.0 and everything else 0 always returns HR."""
        table = {o.value: 0.0 for o in Outcome}
        table['HR'] = 1.0
        rng = np.random.default_rng(42)
        for _ in range(100):
            assert resolve_pa_outcome(table, rng) == Outcome.HR

    def test_deterministic_k(self):
        """K=1.0 always returns K."""
        table = {o.value: 0.0 for o in Outcome}
        table['K'] = 1.0
        rng = np.random.default_rng(42)
        for _ in range(100):
            assert resolve_pa_outcome(table, rng) == Outcome.K

    def test_uniform_distribution_covers_all_outcomes(self):
        """Uniform distribution produces all outcomes over many trials."""
        n = len(Outcome)
        table = {o.value: 1.0 / n for o in Outcome}
        rng = np.random.default_rng(42)
        seen = set()
        for _ in range(5000):
            seen.add(resolve_pa_outcome(table, rng))
        assert seen == set(Outcome)

    def test_returns_outcome_enum(self):
        """Return type is Outcome enum, not a string."""
        table = {o.value: 0.0 for o in Outcome}
        table['BB'] = 1.0
        rng = np.random.default_rng(42)
        result = resolve_pa_outcome(table, rng)
        assert isinstance(result, Outcome)


from mlb.data.models import BaseState
from mlb.engine.simulate import advance_runners


class TestAdvanceRunnersStrikeout:
    """advance_runners: strikeout behavior."""

    def test_k_no_runners(self):
        """Strikeout with empty bases: no runs, outs +1."""
        bases = BaseState()
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.K, 0, rng)
        assert new_bases == BaseState()
        assert runs == 0
        assert outs == 1

    def test_k_bases_loaded(self):
        """Strikeout with bases loaded: runners hold, outs +1."""
        bases = BaseState(first=True, second=True, third=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.K, 1, rng)
        assert new_bases == BaseState(first=True, second=True, third=True)
        assert runs == 0
        assert outs == 2


class TestAdvanceRunnersWalkHBP:
    """advance_runners: walk and HBP behavior (force advances only)."""

    def test_walk_empty_bases(self):
        """Walk with nobody on: batter to 1st."""
        bases = BaseState()
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.BB, 0, rng)
        assert new_bases == BaseState(first=True)
        assert runs == 0
        assert outs == 0

    def test_walk_runner_on_first(self):
        """Walk with runner on 1st: runners on 1st and 2nd."""
        bases = BaseState(first=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.BB, 0, rng)
        assert new_bases == BaseState(first=True, second=True)
        assert runs == 0
        assert outs == 0

    def test_walk_runners_on_first_and_second(self):
        """Walk with 1st and 2nd occupied: bases loaded."""
        bases = BaseState(first=True, second=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.BB, 0, rng)
        assert new_bases == BaseState(first=True, second=True, third=True)
        assert runs == 0
        assert outs == 0

    def test_walk_bases_loaded(self):
        """Walk with bases loaded: 1 run scores, bases stay loaded."""
        bases = BaseState(first=True, second=True, third=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.BB, 0, rng)
        assert new_bases == BaseState(first=True, second=True, third=True)
        assert runs == 1
        assert outs == 0

    def test_walk_runner_on_second_only(self):
        """Walk with runner on 2nd only: batter to 1st, runner holds at 2nd (no force)."""
        bases = BaseState(second=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.BB, 0, rng)
        assert new_bases == BaseState(first=True, second=True)
        assert runs == 0
        assert outs == 0

    def test_walk_runner_on_third_only(self):
        """Walk with runner on 3rd only: batter to 1st, runner holds at 3rd (no force)."""
        bases = BaseState(third=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.BB, 0, rng)
        assert new_bases == BaseState(first=True, third=True)
        assert runs == 0
        assert outs == 0

    def test_walk_first_and_third(self):
        """Walk with 1st and 3rd: runner on 1st forced to 2nd, 3rd holds."""
        bases = BaseState(first=True, third=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.BB, 0, rng)
        assert new_bases == BaseState(first=True, second=True, third=True)
        assert runs == 0
        assert outs == 0

    def test_hbp_same_as_walk(self):
        """HBP forces same advancement as walk."""
        bases = BaseState(first=True, second=True, third=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.HBP, 0, rng)
        assert new_bases == BaseState(first=True, second=True, third=True)
        assert runs == 1
        assert outs == 0


class TestAdvanceRunnersOut:
    """advance_runners: generic out behavior."""

    def test_out_empty_bases(self):
        """Out with nobody on: outs +1."""
        bases = BaseState()
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.OUT, 0, rng)
        assert new_bases == BaseState()
        assert runs == 0
        assert outs == 1

    def test_out_runner_on_third_less_than_two_outs_sac_fly(self):
        """Out with runner on 3rd, < 2 outs: sac fly ~50% of time over many trials."""
        rng = np.random.default_rng(42)
        scored_count = 0
        n_trials = 2000
        for _ in range(n_trials):
            bases = BaseState(third=True)
            _, runs, _ = advance_runners(bases, Outcome.OUT, 0, rng)
            scored_count += runs
        ratio = scored_count / n_trials
        assert 0.40 < ratio < 0.60, f"Sac fly ratio {ratio} outside expected range"

    def test_out_runner_on_third_two_outs_no_sac(self):
        """Out with runner on 3rd, 2 outs: no sac fly, runner holds."""
        bases = BaseState(third=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.OUT, 2, rng)
        assert new_bases == BaseState(third=True)
        assert runs == 0
        assert outs == 3

    def test_out_runner_on_first_holds(self):
        """Out with runner on 1st: runner holds (no GIDP in v1)."""
        bases = BaseState(first=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.OUT, 0, rng)
        assert new_bases == BaseState(first=True)
        assert runs == 0
        assert outs == 1


class TestAdvanceRunnersSingle:
    """advance_runners: single behavior."""

    def test_single_empty_bases(self):
        """Single with nobody on: batter to 1st."""
        bases = BaseState()
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.SINGLE, 0, rng)
        assert new_bases.first is True
        assert runs == 0
        assert outs == 0

    def test_single_runner_on_second_scores_or_third(self):
        """Single with runner on 2nd: scores (90%) or holds 3rd (10%)."""
        rng = np.random.default_rng(42)
        scored = 0
        n_trials = 2000
        for _ in range(n_trials):
            bases = BaseState(second=True)
            new_bases, runs, outs = advance_runners(bases, Outcome.SINGLE, 0, rng)
            assert new_bases.first is True
            assert outs == 0
            scored += runs
        ratio = scored / n_trials
        assert 0.85 < ratio < 0.95, f"Runner from 2nd scoring ratio {ratio}"

    def test_single_runner_on_third_scores(self):
        """Single with runner on 3rd: always scores."""
        bases = BaseState(third=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.SINGLE, 0, rng)
        assert runs == 1
        assert new_bases.third is False
        assert outs == 0

    def test_single_runner_on_first_advances(self):
        """Single with runner on 1st: advances to 2nd (70%) or 3rd (30%)."""
        rng = np.random.default_rng(42)
        to_third_count = 0
        n_trials = 2000
        for _ in range(n_trials):
            bases = BaseState(first=True)
            new_bases, runs, outs = advance_runners(bases, Outcome.SINGLE, 0, rng)
            assert new_bases.first is True  # batter on 1st
            if new_bases.third:
                to_third_count += 1
        ratio = to_third_count / n_trials
        assert 0.25 < ratio < 0.35, f"Runner 1st-to-3rd ratio {ratio}"

    def test_single_bases_loaded(self):
        """Single with bases loaded: runner on 3rd scores, batter at 1st."""
        bases = BaseState(first=True, second=True, third=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.SINGLE, 0, rng)
        assert runs >= 1  # at least runner on 3rd scores
        assert new_bases.first is True
        assert outs == 0


class TestAdvanceRunnersDouble:
    """advance_runners: double behavior."""

    def test_double_empty_bases(self):
        """Double with nobody on: batter to 2nd."""
        bases = BaseState()
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.DOUBLE, 0, rng)
        assert new_bases == BaseState(second=True)
        assert runs == 0
        assert outs == 0

    def test_double_runner_on_second_scores(self):
        """Double with runner on 2nd: runner scores."""
        bases = BaseState(second=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.DOUBLE, 0, rng)
        assert new_bases.second is True  # batter at 2nd
        assert runs == 1
        assert outs == 0

    def test_double_runner_on_third_scores(self):
        """Double with runner on 3rd: runner scores."""
        bases = BaseState(third=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.DOUBLE, 0, rng)
        assert runs == 1
        assert outs == 0

    def test_double_runner_on_first_to_third_or_scores(self):
        """Double with runner on 1st: to 3rd (60%) or scores (40%)."""
        rng = np.random.default_rng(42)
        scored = 0
        n_trials = 2000
        for _ in range(n_trials):
            bases = BaseState(first=True)
            new_bases, runs, outs = advance_runners(bases, Outcome.DOUBLE, 0, rng)
            assert new_bases.second is True  # batter at 2nd
            scored += runs
        ratio = scored / n_trials
        assert 0.35 < ratio < 0.45, f"Runner 1st scoring on double ratio {ratio}"

    def test_double_bases_loaded(self):
        """Double with bases loaded: runners on 2nd and 3rd score, runner on 1st to 3rd or scores."""
        bases = BaseState(first=True, second=True, third=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.DOUBLE, 0, rng)
        assert runs >= 2  # at least 2nd and 3rd score
        assert new_bases.second is True  # batter
        assert outs == 0


class TestAdvanceRunnersTriple:
    """advance_runners: triple behavior."""

    def test_triple_empty_bases(self):
        """Triple with nobody on: batter to 3rd."""
        bases = BaseState()
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.TRIPLE, 0, rng)
        assert new_bases == BaseState(third=True)
        assert runs == 0
        assert outs == 0

    def test_triple_bases_loaded(self):
        """Triple with bases loaded: 3 runs score, batter on 3rd."""
        bases = BaseState(first=True, second=True, third=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.TRIPLE, 0, rng)
        assert new_bases == BaseState(third=True)
        assert runs == 3
        assert outs == 0


class TestAdvanceRunnersHR:
    """advance_runners: home run behavior."""

    def test_hr_empty_bases(self):
        """Solo HR: 1 run, bases empty."""
        bases = BaseState()
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.HR, 0, rng)
        assert new_bases == BaseState()
        assert runs == 1
        assert outs == 0

    def test_hr_bases_loaded(self):
        """Grand slam: 4 runs, bases empty."""
        bases = BaseState(first=True, second=True, third=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.HR, 0, rng)
        assert new_bases == BaseState()
        assert runs == 4
        assert outs == 0

    def test_hr_runner_on_second(self):
        """HR with runner on 2nd: 2 runs, bases empty."""
        bases = BaseState(second=True)
        rng = np.random.default_rng(42)
        new_bases, runs, outs = advance_runners(bases, Outcome.HR, 0, rng)
        assert new_bases == BaseState()
        assert runs == 2
        assert outs == 0


from mlb.config import LEAGUE_AVERAGES
from mlb.data.models import GameState, Lineup, PitcherStats
from mlb.engine.simulate import should_pull_starter, get_current_pitcher


def _make_pitcher(name: str, throws: str = 'R') -> PitcherStats:
    """Helper to create a PitcherStats with neutral rates."""
    rates = {o.value: 1.0 / len(Outcome) for o in Outcome}
    return PitcherStats(
        player_id=name, name=name, throws=Hand(throws),
        pa_against=500, rates=rates,
    )


def _make_lineup(starter_name: str = 'Starter', bullpen_names: list | None = None) -> Lineup:
    """Helper to create a Lineup with a starter and bullpen."""
    if bullpen_names is None:
        bullpen_names = ['Reliever1', 'Reliever2', 'Closer']
    batter_rates = {o.value: 1.0 / len(Outcome) for o in Outcome}
    batters = [
        BatterStats(player_id=f'b{i}', name=f'Batter{i}', bats=Hand.RIGHT, pa=500, rates=batter_rates)
        for i in range(9)
    ]
    return Lineup(
        team_id='TST', team_name='Test Team',
        batting_order=batters,
        starting_pitcher=_make_pitcher(starter_name),
        bullpen=[_make_pitcher(name) for name in bullpen_names],
    )


class TestShouldPullStarter:
    """should_pull_starter: pitcher substitution logic."""

    def test_under_limit_keep(self):
        """99 pitches, 5 innings, 3 runs: keep starter."""
        assert should_pull_starter(99, 5.0, 3) is False

    def test_at_limit_pull(self):
        """100 pitches: pull starter."""
        assert should_pull_starter(100, 5.0, 3) is True

    def test_over_limit_pull(self):
        """120 pitches: pull starter."""
        assert should_pull_starter(120, 5.0, 3) is True

    def test_complete_game_pull(self):
        """9.0 innings pitched: pull (complete game cap)."""
        assert should_pull_starter(80, 9.0, 2) is True

    def test_blowup_pull(self):
        """8 runs allowed: pull regardless of pitch count."""
        assert should_pull_starter(40, 3.0, 8) is True

    def test_custom_config_pitch_limit(self):
        """Custom pitch count limit via config."""
        assert should_pull_starter(80, 5.0, 3, config={'pitch_count_limit': 80}) is True
        assert should_pull_starter(79, 5.0, 3, config={'pitch_count_limit': 80}) is False

    def test_custom_config_innings_limit(self):
        """Custom innings limit via config."""
        assert should_pull_starter(50, 7.0, 2, config={'innings_limit': 7.0}) is True

    def test_custom_config_runs_limit(self):
        """Custom runs allowed limit via config."""
        assert should_pull_starter(50, 5.0, 5, config={'runs_limit': 5}) is True


class TestGetCurrentPitcher:
    """get_current_pitcher: pitcher selection and bullpen transitions."""

    def test_returns_starter_initially(self):
        """Before any substitution, returns the starting pitcher."""
        lineup = _make_lineup()
        state = GameState()
        pitcher = get_current_pitcher(lineup, state, is_home=True)
        assert pitcher.name == 'Starter'

    def test_returns_bullpen_after_pull(self):
        """After starter pulled (home_bullpen_index=0), returns first bullpen arm."""
        lineup = _make_lineup()
        state = GameState(home_bullpen_index=0, home_pitch_count=0)
        pitcher = get_current_pitcher(lineup, state, is_home=True)
        assert pitcher.name == 'Reliever1'

    def test_returns_second_bullpen_arm(self):
        """home_bullpen_index=1 returns second reliever."""
        lineup = _make_lineup()
        state = GameState(home_bullpen_index=1, home_pitch_count=0)
        pitcher = get_current_pitcher(lineup, state, is_home=True)
        assert pitcher.name == 'Reliever2'

    def test_away_team_uses_away_fields(self):
        """is_home=False reads from away_bullpen_index."""
        lineup = _make_lineup()
        state = GameState(away_bullpen_index=0)
        pitcher = get_current_pitcher(lineup, state, is_home=False)
        assert pitcher.name == 'Reliever1'

    def test_exhausted_bullpen_uses_last_arm(self):
        """If bullpen index exceeds length, use last available arm."""
        lineup = _make_lineup(bullpen_names=['OnlyRelief'])
        state = GameState(home_bullpen_index=5)
        pitcher = get_current_pitcher(lineup, state, is_home=True)
        assert pitcher.name == 'OnlyRelief'
