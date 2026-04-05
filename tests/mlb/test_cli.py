import json

from mlb.config import LEAGUE_AVERAGES
from mlb.engine.aggregate import aggregate_simulations
from mlb.scripts.diagnose_calibration import build_league_average_context
from mlb.scripts.simulate_game import build_parser, filter_games, serialize_simulation_result
from mlb.scripts.test_smoke import build_synthetic_game_context


class TestSyntheticSmoke:
    def test_synthetic_context_runs(self):
        context = build_synthetic_game_context()
        result = aggregate_simulations(context, LEAGUE_AVERAGES, n_simulations=100, base_seed=11)
        assert result.game_id == "smoke-game"
        assert result.n_simulations == 100
        assert result.total_runs_mean >= 0
        assert len(result.player_stats) > 0

    def test_league_average_calibration_is_reasonable(self):
        context = build_league_average_context()
        result = aggregate_simulations(context, LEAGUE_AVERAGES, n_simulations=1000, base_seed=42)
        assert 8.5 <= result.total_runs_mean <= 9.5


class TestJsonSerialization:
    def test_round_trip(self):
        context = build_synthetic_game_context()
        result = aggregate_simulations(context, LEAGUE_AVERAGES, n_simulations=50, base_seed=3)
        payload = serialize_simulation_result(result, context, seed=3, data_warnings=["fallback used"])
        encoded = json.dumps(payload)
        decoded = json.loads(encoded)

        assert decoded["game_id"] == "smoke-game"
        assert decoded["date"] == "2026-04-03"
        assert decoded["venue"] == "Smoke Test Park"
        assert isinstance(decoded["player_stats"], dict)
        assert isinstance(decoded["betting_lines"], dict)
        assert decoded["metadata"]["seed"] == 3
        assert isinstance(decoded["metadata"]["data_warnings"], list)
        assert "inning_scores" not in decoded


class TestArgparseDefaults:
    def test_defaults(self):
        args = build_parser().parse_args([])
        assert isinstance(args.date, str)
        assert args.sims == 10000
        assert args.json is False
        assert args.verbose is False


class TestTeamFilter:
    def test_team_filter(self):
        games = [
            {"game_id": "1", "away_team": "New York Yankees", "home_team": "Boston Red Sox"},
            {"game_id": "2", "away_team": "Chicago Cubs", "home_team": "St. Louis Cardinals"},
            {"game_id": "3", "away_team": "Houston Astros", "home_team": "Seattle Mariners"},
        ]
        filtered = filter_games(games, team="Yankees")
        assert len(filtered) == 1
        assert filtered[0]["game_id"] == "1"
