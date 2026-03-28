import json
import pytest
from pathlib import Path
from optimizer.fitness import evaluate, evaluate_config
from models.v2_current import compute_my_line, compute_confidence_and_ev


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


class TestEvaluateConfig:
    def test_returns_flat_dict(self, tmp_path):
        _make_cache(tmp_path)
        result = evaluate_config(DEFAULT_CFG, tmp_path)
        assert "beat_rate" in result
        assert "avg_miss" in result
        assert "avg_advantage" in result
        assert "roi" in result
        assert "win_rate" in result
        assert "total_bets" in result
        assert "total_games" in result

    def test_empty_cache_returns_penalties(self, tmp_path):
        result = evaluate_config(DEFAULT_CFG, tmp_path)
        assert result["beat_rate"] == 0.0
        assert result["avg_miss"] == 999.0
        assert result["roi"] == -100.0
        assert result["total_games"] == 0

    def test_sample_size_truncates_games(self, tmp_path):
        _make_cache(tmp_path)
        small_cfg = {**DEFAULT_CFG, "sample_size": 1}
        result = evaluate_config(small_cfg, tmp_path)
        assert result["total_games"] == 1

    def test_values_are_numeric(self, tmp_path):
        _make_cache(tmp_path)
        result = evaluate_config(DEFAULT_CFG, tmp_path)
        for key in ("beat_rate", "avg_miss", "avg_advantage", "roi", "win_rate"):
            assert isinstance(result[key], (int, float)), f"{key} should be numeric"


YGGDRASIL_CACHE = Path(__file__).parent.parent.parent / "yggdrasil" / "cache"

PRODUCTION_CFG = {
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


class TestParity:
    @pytest.mark.skipif(
        not YGGDRASIL_CACHE.exists(),
        reason="Yggdrasil cache not available",
    )
    def test_projections_match_cached_predictions(self):
        """Verify ported model produces same projected_total as Yggdrasil."""
        cache_path = YGGDRASIL_CACHE
        input_files = sorted(cache_path.glob("????-??-??-nba-model-inputs.json"))

        if not input_files:
            pytest.skip("No model-inputs files in cache")

        tested = 0
        mismatches = []

        for input_file in input_files[-3:]:
            date_str = input_file.name[:10]
            pred_file = cache_path / f"{date_str}-nba-predictions.json"

            if not pred_file.exists():
                continue

            with open(input_file) as f:
                model_inputs = json.load(f)
            with open(pred_file) as f:
                predictions = json.load(f)

            pred_by_id = {
                str(g["game_id"]): g for g in predictions.get("games", [])
            }

            for mi in model_inputs:
                game_id = str(mi["id"])
                pred = pred_by_id.get(game_id)
                if not pred or pred.get("projected_total") is None:
                    continue

                result = compute_my_line(
                    mi["home_games"], mi["away_games"],
                    date_str, PRODUCTION_CFG,
                )
                if result["my_line"] is None:
                    continue

                computed = round(result["my_line"], 1)
                expected = pred["projected_total"]
                tested += 1

                if abs(computed - expected) > 0.15:
                    mismatches.append({
                        "date": date_str,
                        "game_id": game_id,
                        "computed": computed,
                        "expected": expected,
                        "diff": round(computed - expected, 2),
                    })

        assert tested > 0, "No games tested -- check cache data"
        assert not mismatches, (
            f"{len(mismatches)}/{tested} projection mismatches:\n"
            + "\n".join(str(m) for m in mismatches[:5])
        )

    @pytest.mark.skipif(
        not YGGDRASIL_CACHE.exists(),
        reason="Yggdrasil cache not available",
    )
    def test_z_scores_match_cached_predictions(self):
        """Verify ported model produces same z_scores as Yggdrasil."""
        cache_path = YGGDRASIL_CACHE
        input_files = sorted(cache_path.glob("????-??-??-nba-model-inputs.json"))

        if not input_files:
            pytest.skip("No model-inputs files in cache")

        tested = 0
        mismatches = []

        for input_file in input_files[-3:]:
            date_str = input_file.name[:10]
            pred_file = cache_path / f"{date_str}-nba-predictions.json"

            if not pred_file.exists():
                continue

            with open(input_file) as f:
                model_inputs = json.load(f)
            with open(pred_file) as f:
                predictions = json.load(f)

            pred_by_id = {
                str(g["game_id"]): g for g in predictions.get("games", [])
            }

            for mi in model_inputs:
                game_id = str(mi["id"])
                pred = pred_by_id.get(game_id)
                if not pred or pred.get("opening_z_score") is None:
                    continue

                result = compute_my_line(
                    mi["home_games"], mi["away_games"],
                    date_str, PRODUCTION_CFG,
                )
                if result["my_line"] is None or result["sd_total"] is None:
                    continue

                discrepancy = result["my_line"] - pred["opening_dk_line"]
                conf = compute_confidence_and_ev(
                    discrepancy, result["sd_total"], PRODUCTION_CFG,
                )
                if conf["z_score"] is None:
                    continue

                tested += 1
                if abs(conf["z_score"] - pred["opening_z_score"]) > 0.002:
                    mismatches.append({
                        "date": date_str,
                        "game_id": game_id,
                        "computed_z": conf["z_score"],
                        "expected_z": pred["opening_z_score"],
                    })

        assert tested > 0, "No games tested -- check cache data"
        assert not mismatches, (
            f"{len(mismatches)}/{tested} z-score mismatches:\n"
            + "\n".join(str(m) for m in mismatches[:5])
        )
