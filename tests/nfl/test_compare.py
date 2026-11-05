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


def _graded_stub(hit_by_week):
    """grade_all stand-in: marks each prediction's hit from its week, no network."""
    def fake(preds):
        return [{**p, "actual_yards": 250.0, "hit": hit_by_week[p["week"]],
                 "hypothetical_hit": None, "abs_error": 5.0} for p in preds]
    return fake


class TestOutOfSample:
    def test_counts_only_weeks_from_the_strategys_start(self, tmp_path, monkeypatch):
        oos = {"season": 2026, "week": 8}
        _write(tmp_path / "fade", "2026-wk03-a.json", dict(PRED, week=3, player_id="a", out_of_sample_from=oos))
        _write(tmp_path / "fade", "2026-wk08-b.json", dict(PRED, week=8, player_id="b", out_of_sample_from=oos))
        _write(tmp_path / "fade", "2026-wk09-c.json", dict(PRED, week=9, player_id="c", out_of_sample_from=oos))
        import nfl.backtest.compare as compare_module
        monkeypatch.setattr(compare_module, "grade_all", _graded_stub({3: True, 8: False, 9: True}))

        results = compare_strategies(tmp_path)

        assert results["fade"]["overall_hit_rate"]["n"] == 3          # full record still includes week 3
        oos_result = results["fade"]["out_of_sample"]
        assert oos_result["from"] == oos
        assert oos_result["overall_hit_rate"] == {"n": 2, "hit_rate": 0.5}  # weeks 8 and 9 only

    def test_missing_stamp_means_no_out_of_sample_record(self, tmp_path, monkeypatch):
        _write(tmp_path / "old", "2026-wk03-a.json", PRED)  # predates the out_of_sample_from field
        import nfl.backtest.compare as compare_module
        monkeypatch.setattr(compare_module, "grade_all", _graded_stub({3: True}))

        results = compare_strategies(tmp_path)

        assert results["old"]["out_of_sample"] is None

    def test_later_stamp_wins_when_files_disagree(self, tmp_path, monkeypatch):
        # Moving a strategy's start later must never count earlier weeks; take the latest stamp.
        _write(tmp_path / "s", "2026-wk08-a.json", dict(PRED, week=8, player_id="a", out_of_sample_from={"season": 2026, "week": 8}))
        _write(tmp_path / "s", "2026-wk09-b.json", dict(PRED, week=9, player_id="b", out_of_sample_from={"season": 2026, "week": 9}))
        import nfl.backtest.compare as compare_module
        monkeypatch.setattr(compare_module, "grade_all", _graded_stub({8: True, 9: True}))

        results = compare_strategies(tmp_path)

        assert results["s"]["out_of_sample"]["from"] == {"season": 2026, "week": 9}
        assert results["s"]["out_of_sample"]["overall_hit_rate"]["n"] == 1
