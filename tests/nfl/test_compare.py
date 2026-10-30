import json
from nfl.backtest.compare import compare_strategies


PRED = {
    "player_id": "00-0033873", "player_name": "Patrick Mahomes", "team_abbr": "KC",
    "opponent_abbr": "MIA", "season": 2026, "week": 3,
    "line": 235.5, "scope_used": "last_season",
    "verdict": {"recommendation": "OVER", "confidence": "HIGH", "win_probability": 78.6,
                "market_fair_probability": 50.0, "edge": 28.6, "z_score": -0.68},
    "adjusted_mean": 273.4, "logged_at": "2026-09-26T15:00:00.000Z",
}


def _write(dir_path, filename, data):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / filename).write_text(json.dumps(data))


class TestCompareStrategies:
    def test_discovers_every_strategy_subdirectory(self, tmp_path, monkeypatch):
        _write(tmp_path / "baseline", "2026-wk03-00-0033873.json", PRED)
        _write(tmp_path / "recency_blend_v1", "2026-wk03-00-0033873.json", dict(PRED, adjusted_mean=280.0))

        # No actual game has been played for this synthetic player/week, so nflverse will
        # correctly return no actual yards -- grade_all handles that as "not yet gradeable."
        # That's fine for this test: we're checking STRUCTURE (one metrics dict per strategy
        # subdirectory found), not grading correctness (already covered by existing grader
        # tests).
        import nfl.backtest.compare as compare_module
        monkeypatch.setattr(compare_module, "grade_all", lambda preds: [
            {**p, "actual_yards": None, "hit": None, "hypothetical_hit": None, "abs_error": None} for p in preds
        ])

        results = compare_strategies(tmp_path)
        assert set(results.keys()) == {"baseline", "recency_blend_v1"}
        assert results["baseline"]["total_predictions"] == 1
        assert results["recency_blend_v1"]["total_predictions"] == 1

    def test_empty_backtest_dir_returns_empty_dict(self, tmp_path):
        results = compare_strategies(tmp_path)
        assert results == {}

    def test_ignores_non_directory_entries(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "README.md").write_text("not a strategy")
        _write(tmp_path / "baseline", "2026-wk03-00-0033873.json", PRED)

        import nfl.backtest.compare as compare_module

        def fake_grade_all(preds):
            return [{**p, "actual_yards": None, "hit": None, "hypothetical_hit": None, "abs_error": None} for p in preds]

        import nfl.backtest.compare as _cm
        _cm.grade_all = fake_grade_all

        results = compare_strategies(tmp_path)
        assert set(results.keys()) == {"baseline"}
