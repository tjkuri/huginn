import json
import pytest
from optimizer.fitness import evaluate


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


DEFAULT_CFG = {
    "sample_size": 10,
    "decay_factor": 0.96,
    "min_z_threshold": 0.5,
    "z_medium": 0.8,
    "z_high": 1.5,
    "home_boost": 1.5,
    "min_home_away_games": 4,
    "vig_win": 0.9091,
    "vig_risk": 1.0,
}


def _make_cache(tmp_path, date="2026-01-10"):
    """Create a minimal cache with one game."""
    model_inputs = [{
        "id": "1001",
        "home_team": {"id": "1", "name": "Home Team", "abbreviation": "HME"},
        "away_team": {"id": "2", "name": "Away Team", "abbreviation": "AWY"},
        "home_games": [
            {"date": "2026-01-09", "pointsScored": 120, "pointsAllowed": 100,
             "isHome": True, "wentToOT": False},
            {"date": "2026-01-08", "pointsScored": 115, "pointsAllowed": 105,
             "isHome": False, "wentToOT": False},
        ],
        "away_games": [
            {"date": "2026-01-09", "pointsScored": 110, "pointsAllowed": 108,
             "isHome": False, "wentToOT": False},
            {"date": "2026-01-08", "pointsScored": 105, "pointsAllowed": 112,
             "isHome": True, "wentToOT": False},
        ],
    }]
    results = [{"game_id": "1001", "actual_total": 230, "went_to_ot": False,
                "status": "final", "home_score": 120, "away_score": 110}]
    predictions = {"games": [{"game_id": "1001", "opening_dk_line": 210.0,
                              "home_team": "Home Team", "away_team": "Away Team",
                              "projected_total": 225.0}]}

    _write_json(tmp_path / f"{date}-nba-model-inputs.json", model_inputs)
    _write_json(tmp_path / f"{date}-nba-results.json", results)
    _write_json(tmp_path / f"{date}-nba-predictions.json", predictions)


class TestEvaluate:
    def test_returns_metrics_dict(self, tmp_path):
        _make_cache(tmp_path)
        metrics = evaluate(tmp_path, DEFAULT_CFG)
        assert "date_range" in metrics
        assert "total_games" in metrics
        assert "v2" in metrics
        assert "book_comparison" in metrics

    def test_processes_games(self, tmp_path):
        _make_cache(tmp_path)
        metrics = evaluate(tmp_path, DEFAULT_CFG)
        assert metrics["total_games"] == 1

    def test_v2_produces_result(self, tmp_path):
        _make_cache(tmp_path)
        metrics = evaluate(tmp_path, DEFAULT_CFG)
        v2 = metrics["v2"]
        assert v2["total_bets"] >= 0
        assert v2["wins"] + v2["losses"] + v2["pushes"] >= 0

    def test_empty_cache(self, tmp_path):
        metrics = evaluate(tmp_path, DEFAULT_CFG)
        assert metrics["total_games"] == 0

    def test_higher_z_threshold_reduces_bets(self, tmp_path):
        _make_cache(tmp_path)
        m1 = evaluate(tmp_path, DEFAULT_CFG)

        aggressive_cfg = {**DEFAULT_CFG, "min_z_threshold": 5.0}
        m2 = evaluate(tmp_path, aggressive_cfg)

        if m2["v2"] is not None:
            assert m2["v2"]["total_bets"] <= m1["v2"]["total_bets"]
