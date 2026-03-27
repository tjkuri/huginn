import json
from pathlib import Path
from data.loader import load_all_days


def _write_fixture_day(tmp_path, date_str, predictions, results):
    """Helper to write a day's prediction + results files."""
    pred_file = tmp_path / f"{date_str}-nba-predictions.json"
    pred_file.write_text(json.dumps({
        "date": date_str,
        "model_version": "v3",
        "config": {},
        "games": predictions,
    }))
    results_file = tmp_path / f"{date_str}-nba-results.json"
    results_file.write_text(json.dumps(results))


# --- Fixture data -----------------------------------------------------------

PRED_GAME_1 = {
    "game_id": "1001",
    "home_team": "Boston Celtics",
    "away_team": "New York Knicks",
    "projected_total": 228.0,
    "sd_total": 10.0,
    "opening_dk_line": 220.0,
    "opening_gap": 8.0,
    "opening_z_score": 0.8,
    "opening_confidence": "MEDIUM",
    "opening_recommendation": "O",
    "opening_win_prob": 0.7881,
    "opening_ev": 50.3,
    "v1_line": 225.0,
}

PRED_GAME_2 = {
    "game_id": "1002",
    "home_team": "Los Angeles Lakers",
    "away_team": "Golden State Warriors",
    "projected_total": 215.0,
    "sd_total": 10.0,
    "opening_dk_line": 225.0,
    "opening_gap": -10.0,
    "opening_z_score": -1.0,
    "opening_confidence": "MEDIUM",
    "opening_recommendation": "U",
    "opening_win_prob": 0.8413,
    "opening_ev": 60.5,
    "v1_line": 220.0,
}

PRED_GAME_3 = {
    "game_id": "1003",
    "home_team": "Chicago Bulls",
    "away_team": "Miami Heat",
    "projected_total": 222.0,
    "sd_total": 10.0,
    "opening_dk_line": 220.0,
    "opening_gap": 2.0,
    "opening_z_score": 0.2,
    "opening_confidence": "LOW",
    "opening_recommendation": "NO_BET",
    "opening_win_prob": 0.579,
    "opening_ev": -8.5,
    "v1_line": 218.0,
}

RESULT_GAME_1 = {
    "game_id": "1001",
    "home_team": "Boston Celtics",
    "away_team": "New York Knicks",
    "home_score": 120,
    "away_score": 110,
    "actual_total": 230,
    "home_score_regulation": 120,
    "away_score_regulation": 110,
    "actual_total_regulation": 230,
    "went_to_ot": False,
    "ot_periods": 0,
    "status": "final",
    "last_updated": "2026-01-02T03:00:00.000Z",
}

RESULT_GAME_2 = {
    "game_id": "1002",
    "home_team": "Los Angeles Lakers",
    "away_team": "Golden State Warriors",
    "home_score": 105,
    "away_score": 108,
    "actual_total": 213,
    "home_score_regulation": 105,
    "away_score_regulation": 108,
    "actual_total_regulation": 213,
    "went_to_ot": False,
    "ot_periods": 0,
    "status": "final",
    "last_updated": "2026-01-02T03:00:00.000Z",
}

RESULT_GAME_3 = {
    "game_id": "1003",
    "home_team": "Chicago Bulls",
    "away_team": "Miami Heat",
    "home_score": 105,
    "away_score": 110,
    "actual_total": 215,
    "home_score_regulation": 105,
    "away_score_regulation": 110,
    "actual_total_regulation": 215,
    "went_to_ot": False,
    "ot_periods": 0,
    "status": "final",
    "last_updated": "2026-01-03T03:00:00.000Z",
}


# --- Tests -------------------------------------------------------------------

class TestLoadAllDays:
    def test_loads_and_merges_paired_files(self, tmp_path):
        _write_fixture_day(tmp_path, "2026-01-01",
                           [PRED_GAME_1, PRED_GAME_2],
                           [RESULT_GAME_1, RESULT_GAME_2])

        games = load_all_days(tmp_path)

        assert len(games) == 2
        g1 = next(g for g in games if g["game_id"] == "1001")
        assert g1["date"] == "2026-01-01"
        assert g1["home_team"] == "Boston Celtics"
        assert g1["projected_total"] == 228.0
        assert g1["opening_dk_line"] == 220.0
        assert g1["opening_recommendation"] == "O"
        assert g1["v1_line"] == 225.0
        assert g1["actual_total"] == 230
        assert g1["went_to_ot"] is False

    def test_skips_day_with_no_results_file(self, tmp_path):
        pred_file = tmp_path / "2026-01-01-nba-predictions.json"
        pred_file.write_text(json.dumps({
            "date": "2026-01-01", "model_version": "v3",
            "config": {}, "games": [PRED_GAME_1],
        }))

        games = load_all_days(tmp_path)
        assert len(games) == 0

    def test_skips_incomplete_day(self, tmp_path):
        incomplete_result = dict(RESULT_GAME_1)
        incomplete_result["status"] = "in_progress"

        _write_fixture_day(tmp_path, "2026-01-01",
                           [PRED_GAME_1], [incomplete_result])

        games = load_all_days(tmp_path)
        assert len(games) == 0

    def test_days_filter_limits_to_last_n(self, tmp_path):
        _write_fixture_day(tmp_path, "2026-01-01",
                           [PRED_GAME_1], [RESULT_GAME_1])
        _write_fixture_day(tmp_path, "2026-01-02",
                           [PRED_GAME_3], [RESULT_GAME_3])

        games = load_all_days(tmp_path, days=1)
        assert len(games) == 1
        assert games[0]["date"] == "2026-01-02"

    def test_team_filter(self, tmp_path):
        _write_fixture_day(tmp_path, "2026-01-01",
                           [PRED_GAME_1, PRED_GAME_2],
                           [RESULT_GAME_1, RESULT_GAME_2])

        games = load_all_days(tmp_path, team="Lakers")
        assert len(games) == 1
        assert games[0]["game_id"] == "1002"

    def test_team_filter_case_insensitive(self, tmp_path):
        _write_fixture_day(tmp_path, "2026-01-01",
                           [PRED_GAME_1], [RESULT_GAME_1])

        games = load_all_days(tmp_path, team="celtics")
        assert len(games) == 1

    def test_skips_game_with_no_matching_result(self, tmp_path):
        mismatched_result = dict(RESULT_GAME_1)
        mismatched_result["game_id"] = "9999"

        _write_fixture_day(tmp_path, "2026-01-01",
                           [PRED_GAME_1], [mismatched_result])

        games = load_all_days(tmp_path)
        assert len(games) == 0

    def test_handles_legacy_status_null_with_scores(self, tmp_path):
        legacy_result = dict(RESULT_GAME_1)
        del legacy_result["status"]

        _write_fixture_day(tmp_path, "2026-01-01",
                           [PRED_GAME_1], [legacy_result])

        games = load_all_days(tmp_path)
        assert len(games) == 1

    def test_returns_empty_for_empty_cache(self, tmp_path):
        games = load_all_days(tmp_path)
        assert games == []
