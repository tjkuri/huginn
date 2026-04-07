"""CLI entry point for MLB game simulation."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum

from mlb.config import LEAGUE_AVERAGES, NUM_SIMULATIONS, SEASON
from mlb.data.builder import build_game_context
from mlb.data.lineups import fetch_todays_games
from mlb.data.models import GameContext, SimulatedGame, SimulationResult
from mlb.data.stats import fetch_batting_splits, fetch_pitching_splits
from mlb.engine.aggregate import (
    compute_betting_lines,
    compute_player_stats,
    compute_run_distributions,
    compute_win_probability,
    run_simulations,
)
from mlb.scripts.format_output import build_terminal_output

try:
    from rich.console import Console
    HAS_RICH = True
except ImportError:  # pragma: no cover
    Console = None
    HAS_RICH = False

logger = logging.getLogger(__name__)

_PROGRESS_CHUNK_SIZE = 1000
TERMINAL_WIDTH = 90
TERMINAL_CONSOLE = Console(width=TERMINAL_WIDTH) if HAS_RICH else None


def _today_str() -> str:
    return date.today().isoformat()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Run MLB Monte Carlo simulations.")
    parser.add_argument("--date", default=_today_str(), help="Date to simulate (YYYY-MM-DD).")
    parser.add_argument("--game-id", dest="game_id", help="Simulate a single game by game id.")
    parser.add_argument("--team", help="Filter to games involving a team name.")
    parser.add_argument("--sims", type=int, default=NUM_SIMULATIONS, help="Number of simulations to run.")
    parser.add_argument("--seed", type=int, help="Base random seed for reproducibility.")
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout only.")
    parser.add_argument("--verbose", action="store_true", help="Show progress and detailed logging.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    return build_parser().parse_args(argv)


def configure_logging(verbose: bool, json_mode: bool) -> None:
    """Configure logging to stderr based on output mode."""
    level = logging.CRITICAL if json_mode else (logging.INFO if verbose else logging.ERROR)
    logging.basicConfig(level=level, stream=sys.stderr, format="%(levelname)s: %(message)s", force=True)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setLevel(level)


def _status_context(message: str, json_mode: bool):
    """Return a Rich status context when terminal output is enabled."""
    if json_mode or not HAS_RICH or TERMINAL_CONSOLE is None:
        return nullcontext()
    return TERMINAL_CONSOLE.status(f"[bold]{message}")


def filter_games(games: list[dict], team: str | None = None, game_id: str | None = None) -> list[dict]:
    """Apply team and game-id filters to a schedule list."""
    filtered = list(games)
    if team:
        needle = team.strip().lower()
        filtered = [
            game for game in filtered
            if needle in str(game.get("away_team", "")).lower()
            or needle in str(game.get("home_team", "")).lower()
        ]
    if game_id:
        filtered = [game for game in filtered if str(game.get("game_id")) == str(game_id)]
    return filtered


def _serialize_value(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _serialize_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    return value


def serialize_simulation_result(
    result: SimulationResult,
    game_context: GameContext,
    seed: int | None,
    data_warnings: list[str],
) -> dict:
    """Serialize a SimulationResult plus CLI metadata to a JSON-friendly dict."""
    payload = _serialize_value(result)
    payload["date"] = game_context.date
    payload["venue"] = game_context.park_factors.venue_name
    payload["metadata"] = {
        "n_simulations": result.n_simulations,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_warnings": list(data_warnings),
    }
    return payload


def format_terminal_report(
    result: SimulationResult,
    game_context: GameContext,
    data_warnings: list[str],
    sample_game: SimulatedGame | None = None,
    sample_index: int | None = None,
    simulated_games: list[SimulatedGame] | None = None,
):
    """Build a terminal report renderable or plain-text string."""
    return build_terminal_output(result, game_context, sample_game, sample_index, data_warnings, simulated_games)


@contextmanager
def capture_data_warnings() -> list[str]:
    """Capture data-source log records emitted during data assembly."""

    class _ListHandler(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.INFO)
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(record.getMessage())

    handler = _ListHandler()
    target_names = [
        "mlb.data.builder",
        "mlb.data.stats",
        "mlb.data.lineups",
        "mlb.data.models",
    ]
    target_loggers = [logging.getLogger(name) for name in target_names]
    previous_levels = {logger_obj.name: logger_obj.level for logger_obj in target_loggers}
    for logger_obj in target_loggers:
        logger_obj.addHandler(handler)
        logger_obj.setLevel(logging.INFO)
    try:
        yield handler.messages
    finally:
        for logger_obj in target_loggers:
            logger_obj.removeHandler(handler)
            logger_obj.setLevel(previous_levels[logger_obj.name])


def simulate_game_context(
    game_context: GameContext,
    n_simulations: int,
    base_seed: int | None = None,
    verbose: bool = False,
    progress_label: str | None = None,
) -> tuple[SimulationResult, list[SimulatedGame]]:
    """Run simulations for one game, optionally in chunks for progress reporting."""
    effective_seed = base_seed if base_seed is not None else random.randint(0, 2**31 - 1)

    if not verbose or n_simulations <= _PROGRESS_CHUNK_SIZE:
        all_games = run_simulations(game_context, LEAGUE_AVERAGES, n_simulations, effective_seed)
    else:
        remaining = n_simulations
        completed = 0
        all_games = []
        label = progress_label or game_context.game_id
        start_seed = effective_seed

        while remaining > 0:
            chunk_size = min(_PROGRESS_CHUNK_SIZE, remaining)
            chunk_seed = start_seed + completed
            chunk_games = run_simulations(game_context, LEAGUE_AVERAGES, chunk_size, chunk_seed)
            all_games.extend(chunk_games)
            completed += chunk_size
            remaining -= chunk_size
            logger.info("  progress %s/%s sims for %s", completed, n_simulations, label)

    run_dists = compute_run_distributions(all_games)
    win_probs = compute_win_probability(all_games)
    player_stats = compute_player_stats(all_games)
    betting_lines = compute_betting_lines(all_games, run_dists)

    result = SimulationResult(
        game_id=game_context.game_id,
        n_simulations=n_simulations,
        away_team=game_context.away_lineup.team_name,
        home_team=game_context.home_lineup.team_name,
        away_runs_mean=run_dists["away_runs"]["mean"],
        away_runs_std=run_dists["away_runs"]["std"],
        home_runs_mean=run_dists["home_runs"]["mean"],
        home_runs_std=run_dists["home_runs"]["std"],
        total_runs_mean=run_dists["total_runs"]["mean"],
        total_runs_std=run_dists["total_runs"]["std"],
        home_win_pct=win_probs["home_win_pct"],
        away_win_pct=win_probs["away_win_pct"],
        player_stats=player_stats,
        betting_lines=betting_lines,
        run_distributions=run_dists,
    )
    return result, all_games


def load_schedule_and_stats(target_date: str, verbose: bool = False) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    """Fetch schedule and cached/fresh stat inputs."""
    if verbose:
        logger.info("Fetching schedule for %s...", target_date)
    with _status_context("Fetching schedule...", json_mode=False):
        games = fetch_todays_games(date=target_date)
    if verbose:
        logger.info("%s games found", len(games))
        logger.info("Loading batting stats...")
    with _status_context("Loading batting stats...", json_mode=False):
        batting_data = fetch_batting_splits(season=SEASON)
    if verbose:
        counts: dict[str, int] = {}
        for player in batting_data.values():
            src = player.get("source", "other")
            counts[src] = counts.get(src, 0) + 1
        logger.info("Batting stats loaded: %s", counts)

    if verbose:
        logger.info("Loading pitching stats...")
    with _status_context("Loading pitching stats...", json_mode=False):
        pitching_data = fetch_pitching_splits(season=SEASON)
    if verbose:
        counts = {}
        for player in pitching_data.values():
            src = player.get("source", "other")
            counts[src] = counts.get(src, 0) + 1
        logger.info("Pitching stats loaded: %s", counts)
    return games, batting_data, pitching_data


def run_cli(args: argparse.Namespace) -> int:
    """Execute the CLI workflow."""
    configure_logging(args.verbose, args.json)

    try:
        if args.json:
            games = fetch_todays_games(date=args.date)
            batting_data = fetch_batting_splits(season=SEASON)
            pitching_data = fetch_pitching_splits(season=SEASON)
        else:
            games, batting_data, pitching_data = load_schedule_and_stats(args.date, verbose=args.verbose)
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"Failed to load schedule or stats: {exc}", file=sys.stderr)
        return 1

    games = filter_games(games, team=args.team, game_id=args.game_id)
    if not games:
        message = f"No games found for {args.date}."
        if args.team:
            message = f"No games found for {args.team} on {args.date}."
        if args.game_id:
            message = f"Game {args.game_id} was not found for {args.date}."
        if args.json:
            print(json.dumps([]))
        else:
            print(message)
        return 0

    outputs: list[dict] = []
    had_success = False

    for index, game in enumerate(games, start=1):
        label = f"{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}"
        start_time = time.time()
        requested_game_seed = None if args.seed is None else args.seed + index - 1
        effective_game_seed = requested_game_seed if requested_game_seed is not None else random.randint(0, 2**31 - 1)
        if args.verbose:
            logger.info("Simulating Game %s/%s: %s...", index, len(games), label)

        try:
            with capture_data_warnings() as data_warnings:
                context = build_game_context(game, batting_data, pitching_data)
            with _status_context(f"Simulating {args.sims:,} games - {label}...", json_mode=args.json):
                result, simulated_games = simulate_game_context(
                    context,
                    n_simulations=args.sims,
                    base_seed=effective_game_seed,
                    verbose=args.verbose and not args.json,
                    progress_label=label,
                )
        except Exception as exc:
            logger.warning("Skipping %s due to simulation error: %s", label, exc)
            if args.verbose and not args.json:
                logger.info("... skipped")
            continue

        had_success = True
        outputs.append(serialize_simulation_result(result, context, requested_game_seed, data_warnings))
        if args.verbose:
            logger.info("done (%.1fs)", time.time() - start_time)
        if not args.json:
            if index > 1:
                print("-" * 90)
            sample_index = None
            sample_game = None
            if simulated_games:
                rng = random.Random(effective_game_seed)
                sample_index = rng.randrange(len(simulated_games)) + 1
                sample_game = simulated_games[sample_index - 1]
            renderable = format_terminal_report(
                result,
                context,
                data_warnings,
                sample_game,
                sample_index,
                simulated_games,
            )
            if isinstance(renderable, str):
                print(renderable)
            else:
                TERMINAL_CONSOLE.print(renderable)

    if args.json:
        print(json.dumps(outputs, indent=None))
    elif not had_success:
        print("No games could be simulated with the available data.")

    return 0 if had_success or not games else 1


def main(argv: list[str] | None = None) -> int:
    """CLI main entry point."""
    args = parse_args(argv)
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
