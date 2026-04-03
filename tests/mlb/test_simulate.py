"""Tests for mlb.engine.simulate."""
import numpy as np
import pytest

from mlb.config import Outcome
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
