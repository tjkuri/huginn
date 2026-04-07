import json
from argparse import Namespace
from types import SimpleNamespace

from mlb.config import LEAGUE_AVERAGES
from mlb.engine.aggregate import aggregate_simulations
from mlb.scripts.diagnose_calibration import build_league_average_context
from mlb.scripts.simulate_game import (
    build_parser,
    filter_games,
    load_schedule_and_stats,
    run_cli,
    serialize_simulation_result,
    simulate_game_context,
)
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
        assert 7.5 <= result.total_runs_mean <= 10.5


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


class TestVerboseBehavior:
    def test_load_schedule_and_stats_verbose_matches_non_verbose_fetch_behavior(self, monkeypatch):
        monkeypatch.setattr("mlb.scripts.simulate_game.fetch_todays_games", lambda date: [{"game_id": "1"}])
        monkeypatch.setattr("mlb.scripts.simulate_game.fetch_batting_splits", lambda season: {"batter": {"source": "marcel_1yr"}})
        monkeypatch.setattr("mlb.scripts.simulate_game.fetch_pitching_splits", lambda season: {"pitcher": {"source": "marcel_1yr"}})

        quiet_games, quiet_batting, quiet_pitching = load_schedule_and_stats("2026-04-06", verbose=False)
        verbose_games, verbose_batting, verbose_pitching = load_schedule_and_stats("2026-04-06", verbose=True)

        assert quiet_games == verbose_games
        assert quiet_batting == verbose_batting
        assert quiet_pitching == verbose_pitching

    def test_simulate_game_context_uses_same_seed_stream_in_verbose_mode(self, monkeypatch):
        seed_calls: list[tuple[int, int]] = []

        def fake_run_simulations(game_context, league_averages, n_simulations, base_seed):
            del game_context, league_averages
            seed_calls.append((n_simulations, base_seed))
            return [SimpleNamespace()] * n_simulations

        monkeypatch.setattr("mlb.scripts.simulate_game.run_simulations", fake_run_simulations)
        monkeypatch.setattr("mlb.scripts.simulate_game.random.randint", lambda a, b: 12345)
        monkeypatch.setattr("mlb.scripts.simulate_game.compute_run_distributions", lambda games: {
            "away_runs": {"mean": 1.0, "std": 0.0},
            "home_runs": {"mean": 1.0, "std": 0.0},
            "total_runs": {"mean": 2.0, "std": 0.0},
        })
        monkeypatch.setattr("mlb.scripts.simulate_game.compute_win_probability", lambda games: {
            "home_win_pct": 0.5,
            "away_win_pct": 0.5,
        })
        monkeypatch.setattr("mlb.scripts.simulate_game.compute_player_stats", lambda games: {})
        monkeypatch.setattr("mlb.scripts.simulate_game.compute_betting_lines", lambda games, run_dists: {})

        context = SimpleNamespace(
            game_id="game-1",
            away_lineup=SimpleNamespace(team_name="Away"),
            home_lineup=SimpleNamespace(team_name="Home"),
        )

        simulate_game_context(context, n_simulations=2500, base_seed=None, verbose=True, progress_label="label")

        assert seed_calls == [
            (1000, 12345),
            (1000, 13345),
            (500, 14345),
        ]

    def test_run_cli_uses_effective_seed_for_sample_selection(self, monkeypatch, capsys):
        captured_render = {}

        monkeypatch.setattr("mlb.scripts.simulate_game.random.randint", lambda a, b: 24680)
        monkeypatch.setattr(
            "mlb.scripts.simulate_game.load_schedule_and_stats",
            lambda target_date, verbose=False: (
                [{"game_id": "1", "away_team": "MIL", "home_team": "BOS"}],
                {"batter": {}},
                {"pitcher": {}},
            ),
        )
        monkeypatch.setattr("mlb.scripts.simulate_game.build_game_context", lambda game, batting, pitching: SimpleNamespace(
            game_id="1",
            date="2026-04-06",
            away_lineup=SimpleNamespace(team_name="MIL"),
            home_lineup=SimpleNamespace(team_name="BOS"),
            park_factors=SimpleNamespace(venue_name="Fenway Park"),
        ))

        result = SimpleNamespace(
            game_id="1",
            n_simulations=3,
            away_team="MIL",
            home_team="BOS",
            away_runs_mean=1.0,
            away_runs_std=0.0,
            home_runs_mean=1.0,
            home_runs_std=0.0,
            total_runs_mean=2.0,
            total_runs_std=0.0,
            home_win_pct=0.5,
            away_win_pct=0.5,
            player_stats={},
            betting_lines={},
            run_distributions={},
        )
        simulated_games = ["game-a", "game-b", "game-c"]
        monkeypatch.setattr(
            "mlb.scripts.simulate_game.simulate_game_context",
            lambda context, n_simulations, base_seed, verbose=False, progress_label=None: (result, simulated_games),
        )
        monkeypatch.setattr(
            "mlb.scripts.simulate_game.serialize_simulation_result",
            lambda result, game_context, seed, data_warnings: {"game_id": "1", "seed": seed},
        )
        monkeypatch.setattr(
            "mlb.scripts.simulate_game.format_terminal_report",
            lambda result, game_context, data_warnings, sample_game=None, sample_index=None, simulated_games=None: captured_render.update(
                {"sample_game": sample_game, "sample_index": sample_index}
            ) or "report",
        )

        args = Namespace(
            date="2026-04-06",
            game_id=None,
            team=None,
            sims=3,
            seed=None,
            json=False,
            verbose=False,
        )
        exit_code = run_cli(args)

        assert exit_code == 0
        assert captured_render["sample_index"] == 3
        assert captured_render["sample_game"] == "game-c"
        assert "report" in capsys.readouterr().out
