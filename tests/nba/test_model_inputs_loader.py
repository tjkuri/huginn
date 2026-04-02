import json
import pytest
from nba.data.model_inputs_loader import load_model_inputs


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _make_model_input(game_id="1001", home_name="Team A", away_name="Team B"):
    return {
        "id": game_id,
        "home_team": {"id": "1", "name": home_name, "abbreviation": "TA"},
        "away_team": {"id": "2", "name": away_name, "abbreviation": "TB"},
        "home_games": [
            {"date": "2026-01-08", "pointsScored": 110, "pointsAllowed": 100,
             "isHome": True, "wentToOT": False},
            {"date": "2026-01-07", "pointsScored": 105, "pointsAllowed": 108,
             "isHome": False, "wentToOT": False},
        ],
        "away_games": [
            {"date": "2026-01-08", "pointsScored": 100, "pointsAllowed": 115,
             "isHome": False, "wentToOT": False},
            {"date": "2026-01-07", "pointsScored": 108, "pointsAllowed": 105,
             "isHome": True, "wentToOT": False},
        ],
    }


def _make_result(game_id="1001", actual=225, status="final"):
    return {"game_id": game_id, "actual_total": actual, "went_to_ot": False,
            "status": status, "home_score": 115, "away_score": 110}


def _make_prediction(game_id="1001", dk_line=220.5):
    return {"game_id": game_id, "opening_dk_line": dk_line,
            "home_team": "Team A", "away_team": "Team B",
            "projected_total": 228.0}


class TestLoadModelInputs:
    def test_loads_paired_files(self, tmp_path):
        _write_json(tmp_path / "2026-01-10-nba-model-inputs.json",
                     [_make_model_input()])
        _write_json(tmp_path / "2026-01-10-nba-results.json",
                     [_make_result()])
        _write_json(tmp_path / "2026-01-10-nba-predictions.json",
                     {"games": [_make_prediction()]})

        games = load_model_inputs(tmp_path)
        assert len(games) == 1
        assert games[0]["game_id"] == "1001"
        assert games[0]["actual_total"] == 225
        assert games[0]["opening_dk_line"] == 220.5
        assert len(games[0]["home_games"]) == 2

    def test_skips_incomplete_results(self, tmp_path):
        _write_json(tmp_path / "2026-01-10-nba-model-inputs.json",
                     [_make_model_input()])
        _write_json(tmp_path / "2026-01-10-nba-results.json",
                     [_make_result(status="in_progress")])
        _write_json(tmp_path / "2026-01-10-nba-predictions.json",
                     {"games": [_make_prediction()]})

        games = load_model_inputs(tmp_path)
        assert len(games) == 0

    def test_skips_missing_results(self, tmp_path):
        _write_json(tmp_path / "2026-01-10-nba-model-inputs.json",
                     [_make_model_input()])
        games = load_model_inputs(tmp_path)
        assert len(games) == 0

    def test_skips_missing_dk_line(self, tmp_path):
        _write_json(tmp_path / "2026-01-10-nba-model-inputs.json",
                     [_make_model_input()])
        _write_json(tmp_path / "2026-01-10-nba-results.json",
                     [_make_result()])
        # No predictions file -> no DK line
        games = load_model_inputs(tmp_path)
        assert len(games) == 0

    def test_days_filter(self, tmp_path):
        for d in ("2026-01-08", "2026-01-09", "2026-01-10"):
            _write_json(tmp_path / f"{d}-nba-model-inputs.json",
                         [_make_model_input(game_id=d)])
            _write_json(tmp_path / f"{d}-nba-results.json",
                         [_make_result(game_id=d)])
            _write_json(tmp_path / f"{d}-nba-predictions.json",
                         {"games": [_make_prediction(game_id=d)]})

        games = load_model_inputs(tmp_path, days=2)
        dates = {g["date"] for g in games}
        assert dates == {"2026-01-09", "2026-01-10"}

    def test_game_id_coerced_to_string(self, tmp_path):
        _write_json(tmp_path / "2026-01-10-nba-model-inputs.json",
                     [_make_model_input(game_id=12345)])
        _write_json(tmp_path / "2026-01-10-nba-results.json",
                     [_make_result(game_id=12345)])
        _write_json(tmp_path / "2026-01-10-nba-predictions.json",
                     {"games": [_make_prediction(game_id=12345)]})

        games = load_model_inputs(tmp_path)
        assert games[0]["game_id"] == "12345"
