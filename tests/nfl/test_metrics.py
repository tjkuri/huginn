from nfl.backtest.metrics import compute_metrics


def _graded(**overrides):
    g = {
        "verdict": {"confidence": "HIGH"}, "scope_used": "last_season",
        "actual_yards": 250.0, "hit": True, "hypothetical_hit": None, "abs_error": 5.0,
    }
    g.update(overrides)
    return g


class TestComputeMetrics:
    def test_overall_hit_rate(self):
        graded = [_graded(hit=True), _graded(hit=False), _graded(hit=True)]
        metrics = compute_metrics(graded)
        assert metrics["overall_hit_rate"]["n"] == 3
        assert metrics["overall_hit_rate"]["hit_rate"] == round(2 / 3, 4)

    def test_by_confidence_breakdown(self):
        graded = [
            _graded(hit=True, verdict={"confidence": "HIGH"}),
            _graded(hit=True, verdict={"confidence": "HIGH"}),
            _graded(hit=False, verdict={"confidence": "LOW"}),
        ]
        metrics = compute_metrics(graded)
        assert metrics["by_confidence"]["HIGH"]["hit_rate"] == 1.0
        assert metrics["by_confidence"]["LOW"]["hit_rate"] == 0.0

    def test_excludes_ungradeable_predictions(self):
        graded = [_graded(actual_yards=None, hit=None), _graded(hit=True)]
        metrics = compute_metrics(graded)
        assert metrics["gradeable"] == 1
        assert metrics["overall_hit_rate"]["n"] == 1

    def test_no_bet_hypothetical_hit_rate(self):
        graded = [
            _graded(hit=None, hypothetical_hit=True),
            _graded(hit=None, hypothetical_hit=False),
        ]
        metrics = compute_metrics(graded)
        assert metrics["no_bet_hypothetical"]["n"] == 2
        assert metrics["no_bet_hypothetical"]["hit_rate"] == 0.5

    def test_mean_abs_error(self):
        graded = [_graded(abs_error=10.0), _graded(abs_error=20.0)]
        metrics = compute_metrics(graded)
        assert metrics["mean_abs_error"] == 15.0

    def test_mean_abs_error_by_scope(self):
        graded = [
            _graded(scope_used="last_season", abs_error=10.0),
            _graded(scope_used="career", abs_error=30.0),
        ]
        metrics = compute_metrics(graded)
        assert metrics["mean_abs_error_by_scope"]["last_season"] == 10.0
        assert metrics["mean_abs_error_by_scope"]["career"] == 30.0

    def test_empty_input(self):
        metrics = compute_metrics([])
        assert metrics["total_predictions"] == 0
        assert metrics["gradeable"] == 0
