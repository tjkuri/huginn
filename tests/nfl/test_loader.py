import json
from nfl.data.loader import load_predictions

PRED_1 = {
    "player_id": "00-0033873", "player_name": "Patrick Mahomes", "team_abbr": "KC",
    "opponent_abbr": "MIA", "season": 2026, "week": 3,
    "line": 235.5, "scope_used": "last_season",
    "verdict": {"recommendation": "OVER", "confidence": "HIGH", "win_probability": 78.6,
                "market_fair_probability": 50.0, "edge": 28.6, "z_score": -0.68},
    "adjusted_mean": 273.4, "logged_at": "2026-09-26T15:00:00.000Z",
}

PRED_2 = {
    "player_id": "00-0034857", "player_name": "Joe Burrow", "team_abbr": "CIN",
    "opponent_abbr": "BUF", "season": 2026, "week": 4,
    "line": 240.5, "scope_used": "career",
    "verdict": {"recommendation": "NO_BET", "confidence": "LOW", "win_probability": 55.0,
                "market_fair_probability": 51.0, "edge": 4.0, "z_score": -0.2},
    "adjusted_mean": 250.0, "logged_at": "2026-09-27T12:00:00.000Z",
}


def _write(tmp_path, filename, data):
    (tmp_path / filename).write_text(json.dumps(data))


class TestLoadPredictions:
    def test_loads_all_snapshots(self, tmp_path):
        _write(tmp_path, "2026-wk03-00-0033873.json", PRED_1)
        _write(tmp_path, "2026-wk04-00-0034857.json", PRED_2)

        predictions = load_predictions(tmp_path)
        assert len(predictions) == 2

    def test_returns_empty_for_missing_dir(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        predictions = load_predictions(missing)
        assert predictions == []

    def test_player_filter_case_insensitive(self, tmp_path):
        _write(tmp_path, "2026-wk03-00-0033873.json", PRED_1)
        _write(tmp_path, "2026-wk04-00-0034857.json", PRED_2)

        predictions = load_predictions(tmp_path, player="mahomes")
        assert len(predictions) == 1
        assert predictions[0]["player_name"] == "Patrick Mahomes"

    def test_weeks_back_limits_to_last_n_distinct_weeks(self, tmp_path):
        pred_wk5 = dict(PRED_1, week=5, player_id="00-9999999")
        _write(tmp_path, "2026-wk03-00-0033873.json", PRED_1)
        _write(tmp_path, "2026-wk04-00-0034857.json", PRED_2)
        _write(tmp_path, "2026-wk05-00-9999999.json", pred_wk5)

        predictions = load_predictions(tmp_path, weeks_back=1)
        assert len(predictions) == 1
        assert predictions[0]["week"] == 5
