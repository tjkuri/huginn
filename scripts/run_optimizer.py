#!/usr/bin/env python3
"""
Huginn NBA Parameter Optimizer CLI.

Uses Optuna to search a 6-parameter space and find configs that maximize/minimize
a target metric. Supports temporal cross-validation to reduce overfitting.

Usage:
    python scripts/run_optimizer.py                       # 500 trials, beat_rate target
    python scripts/run_optimizer.py --trials 200          # fewer trials
    python scripts/run_optimizer.py --target roi          # optimize for ROI
    python scripts/run_optimizer.py --target avg_miss     # minimize average miss
    python scripts/run_optimizer.py --no-cv               # skip cross-validation
    python scripts/run_optimizer.py --export              # write best config to JSON
    python scripts/run_optimizer.py --cache /path/to/cache
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
from rich.console import Console
from rich.table import Table
from rich import box

from config import YGGDRASIL_CACHE, NBA_MODEL
from optimizer.fitness import evaluate_config, evaluate_config_cv


console = Console()

# ── Target definitions ────────────────────────────────────────────────────────

TARGETS = {
    "beat_rate": {
        "direction": "maximize",
        "key": "beat_rate",
        "format": ".1%",
        "label": "Beat rate",
        "higher_is_better": True,
    },
    "avg_miss": {
        "direction": "minimize",
        "key": "avg_miss",
        "format": ".2f",
        "label": "Avg miss",
        "higher_is_better": False,
    },
    "roi": {
        "direction": "maximize",
        "key": "roi",
        "format": ".2f",
        "label": "ROI",
        "higher_is_better": True,
    },
    "advantage": {
        "direction": "maximize",
        "key": "avg_advantage",
        "format": ".2f",
        "label": "Advantage",
        "higher_is_better": True,
    },
}

# ── Parameter space ───────────────────────────────────────────────────────────

FIXED_PARAMS = {
    "min_home_away_games": 4,
    "vig_win": 0.9091,
    "vig_risk": 1.0,
}

SEARCH_PARAMS = (
    "sample_size",
    "decay_factor",
    "min_z_threshold",
    "z_medium",
    "z_high",
    "home_boost",
)


# ── Core functions ────────────────────────────────────────────────────────────

def build_config(trial):
    """Sample a config from the search space and merge with fixed params."""
    search = {
        "sample_size": trial.suggest_int("sample_size", 5, 20),
        "decay_factor": trial.suggest_float("decay_factor", 0.90, 0.99),
        "min_z_threshold": trial.suggest_float("min_z_threshold", 0.0, 2.0),
        "z_medium": trial.suggest_float("z_medium", 0.3, 2.5),
        "z_high": trial.suggest_float("z_high", 0.8, 3.5),
        "home_boost": trial.suggest_float("home_boost", 0.0, 4.0),
    }
    return {**search, **FIXED_PARAMS}


def validate_config(config):
    """
    Return True if config z thresholds are ordered correctly.
    Requirements: z_medium < z_high AND min_z_threshold <= z_medium
    """
    z_min = config["min_z_threshold"]
    z_med = config["z_medium"]
    z_high = config["z_high"]
    return z_med < z_high and z_min <= z_med


def make_objective(cache_dir, target_key, use_cv):
    """
    Return an Optuna objective function for the given target and cache dir.
    """
    target_info = TARGETS[target_key]
    direction = target_info["direction"]

    def objective(trial):
        config = build_config(trial)

        if not validate_config(config):
            # Penalty for invalid config
            if direction == "minimize":
                return 999.0
            elif target_key == "roi":
                return -100.0
            else:
                return 0.0

        if use_cv:
            result = evaluate_config_cv(config, cache_dir)
        else:
            result = evaluate_config(config, cache_dir)

        trial.set_user_attr("result", result)

        # Special penalty for roi when no bets were placed
        if target_key == "roi" and result.get("total_bets", 0) == 0:
            return -100.0

        return result[target_info["key"]]

    return objective


# ── Display helpers ───────────────────────────────────────────────────────────

def _fmt(value, fmt):
    """Format a display value. Handles .1% specially (multiply by 100)."""
    if fmt == ".1%":
        return f"{value * 100:.1f}%"
    return f"{value:{fmt}}"


def _compare(best_val, current_val, higher_is_better):
    """Return 'green' if best beats current, 'red' otherwise."""
    if higher_is_better:
        return "green" if best_val > current_val else "red"
    else:
        return "green" if best_val < current_val else "red"


def print_results(study, target_key, current_result, use_cv, best_full_result=None):
    """Print optimizer results with rich formatting."""
    target_info = TARGETS[target_key]
    direction = target_info["direction"]
    n_trials = len(study.trials)
    best_trial = study.best_trial
    best_config = best_trial.params
    best_result = best_trial.user_attrs.get("result", {})

    # Use full result for display if available (CV + full eval)
    display_result = best_full_result if best_full_result is not None else best_result

    console.print()
    console.print(" [bold white]NBA PARAMETER OPTIMIZER[/]")
    console.print(f" [dim]{'─' * 60}[/]")
    console.print(f" [dim]Trials      : {n_trials}[/]")
    console.print(f" [dim]Target      : {target_info['label']} ({direction})[/]")
    if use_cv:
        console.print(f" [dim]Mode        : Cross-validation (k=5)[/]")
    else:
        console.print(f" [dim]Mode        : Full dataset[/]")

    # ── Best config ───────────────────────────────────────────────────────────
    console.print()
    console.print(" [bold white]── BEST CONFIG FOUND ──────────────────────────────────────[/]")
    config_table = Table(show_header=True, header_style="dim", box=box.ROUNDED,
                         border_style="dim")
    config_table.add_column("Parameter", style="white")
    config_table.add_column("Value", justify="right")
    config_table.add_column("Current", justify="right", style="dim")

    param_formats = {
        "sample_size": ("d", int),
        "decay_factor": (".4f", float),
        "min_z_threshold": (".3f", float),
        "z_medium": (".3f", float),
        "z_high": (".3f", float),
        "home_boost": (".3f", float),
    }

    for param in SEARCH_PARAMS:
        val = best_config.get(param)
        current_val = NBA_MODEL.get(param)
        fmt, _ = param_formats[param]
        val_str = f"{val:{fmt}}" if val is not None else "[dim]N/A[/]"
        cur_str = f"{current_val:{fmt}}" if current_val is not None else "[dim]N/A[/]"
        config_table.add_row(param, val_str, cur_str)

    console.print(config_table)

    # ── Best result ───────────────────────────────────────────────────────────
    console.print()
    console.print(" [bold white]── BEST RESULT ─────────────────────────────────────────────[/]")

    result_table = Table(show_header=True, header_style="dim", box=box.ROUNDED,
                         border_style="dim")
    result_table.add_column("Metric", style="white")
    result_table.add_column("Best", justify="right")
    result_table.add_column("Current", justify="right", style="dim")

    metrics_to_show = [
        ("beat_rate", "Beat rate", ".1%", True),
        ("avg_miss", "Avg miss", ".2f", False),
        ("avg_advantage", "Advantage", ".2f", True),
        ("total_games", "Games", "d", True),
    ]

    for key, label, fmt, higher in metrics_to_show:
        best_val = display_result.get(key)
        cur_val = current_result.get(key)

        if best_val is None:
            best_str = "[dim]N/A[/]"
            color = "dim"
        else:
            if fmt == ".1%":
                best_str = f"{best_val * 100:.1f}%"
            else:
                best_str = f"{best_val:{fmt}}"
            if cur_val is not None:
                color = _compare(best_val, cur_val, higher)
                best_str = f"[{color}]{best_str}[/]"

        if cur_val is None:
            cur_str = "[dim]N/A[/]"
        elif fmt == ".1%":
            cur_str = f"{cur_val * 100:.1f}%"
        else:
            cur_str = f"{cur_val:{fmt}}"

        result_table.add_row(label, best_str, cur_str)

    console.print(result_table)

    # ── CV overfitting analysis ───────────────────────────────────────────────
    if use_cv and best_full_result is not None:
        console.print()
        console.print(" [bold white]── OVERFITTING ANALYSIS ────────────────────────────────────[/]")
        cv_score = best_result.get(target_info["key"])
        full_score = best_full_result.get(target_info["key"])
        fmt = target_info["format"]

        cv_str = _fmt(cv_score, fmt) if cv_score is not None else "N/A"
        full_str = _fmt(full_score, fmt) if full_score is not None else "N/A"

        console.print(f"  CV score    {cv_str}  [dim](trust this)[/]")
        console.print(f"  Full score  {full_str}  [dim](likely overfit)[/]")

        if cv_score is not None and full_score is not None:
            # Gap: for maximize targets, gap = full - cv. For minimize: gap = cv - full
            if direction == "maximize":
                gap = full_score - cv_score
            else:
                gap = cv_score - full_score

            # Display in same units (for .1% format, multiply by 100)
            if fmt == ".1%":
                gap_display = gap * 100
                gap_str = f"{gap_display:.1f}%"
                gap_color = "red" if abs(gap_display) > 5 else "dim"
            else:
                gap_str = f"{gap:{fmt}}"
                gap_color = "red" if abs(gap) > 5 else "dim"
            console.print(f"  Gap         [{gap_color}]{gap_str}[/]  [dim](>5 = concerning)[/]")

    # ── Sample size warning ───────────────────────────────────────────────────
    total_games = display_result.get("total_games", 0)
    if total_games < 200:
        console.print()
        console.print(f" [yellow]Warning: Only {total_games} games evaluated. Results may not be reliable (need 200+).[/]")

    # ── Current config for comparison ─────────────────────────────────────────
    console.print()
    console.print(" [bold white]── CURRENT CONFIG (NBA_MODEL) ──────────────────────────────[/]")
    cur_config_table = Table(show_header=True, header_style="dim", box=box.ROUNDED,
                             border_style="dim")
    cur_config_table.add_column("Parameter", style="white")
    cur_config_table.add_column("Value", justify="right", style="dim")

    for param in SEARCH_PARAMS:
        val = NBA_MODEL.get(param)
        fmt, _ = param_formats[param]
        val_str = f"{val:{fmt}}" if val is not None else "[dim]N/A[/]"
        cur_config_table.add_row(param, val_str)

    console.print(cur_config_table)

    console.print()
    console.print(" [bold white]── CURRENT RESULT ──────────────────────────────────────────[/]")
    cur_result_table = Table(show_header=True, header_style="dim", box=box.ROUNDED,
                             border_style="dim")
    cur_result_table.add_column("Metric", style="white")
    cur_result_table.add_column("Value", justify="right", style="dim")

    for key, label, fmt, _ in metrics_to_show:
        val = current_result.get(key)
        if val is None:
            val_str = "[dim]N/A[/]"
        elif fmt == ".1%":
            val_str = f"{val * 100:.1f}%"
        else:
            val_str = f"{val:{fmt}}"
        cur_result_table.add_row(label, val_str)

    console.print(cur_result_table)
    console.print()


def export_config(study, target_key, total_games, output_path, full_result=None):
    """Export the best config and result to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_trial = study.best_trial
    best_config = best_trial.params
    best_result = full_result if full_result is not None else best_trial.user_attrs.get("result", {})

    data = {
        "optimized_at": datetime.now(timezone.utc).isoformat(),
        "target": target_key,
        "trials": len(study.trials),
        "games_evaluated": total_games,
        "config": {param: best_config.get(param) for param in SEARCH_PARAMS},
        "result": {
            "beat_rate": best_result.get("beat_rate"),
            "avg_miss": best_result.get("avg_miss"),
            "avg_advantage": best_result.get("avg_advantage"),
        },
    }

    if total_games < 200:
        data["warning"] = f"Only {total_games} games evaluated. Results may not be reliable (need 200+)."

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f" [dim]Exported to: {output_path}[/]")


def main():
    parser = argparse.ArgumentParser(description="NBA Parameter Optimizer")
    parser.add_argument("--trials", type=int, default=500,
                        help="Number of Optuna trials (default: 500)")
    parser.add_argument("--target", choices=list(TARGETS.keys()), default="beat_rate",
                        help="Optimization target (default: beat_rate)")
    parser.add_argument("--cache", type=str, default=None,
                        help="Override cache directory path")
    parser.add_argument("--no-cv", action="store_true", dest="no_cv",
                        help="Skip cross-validation, use full dataset")
    parser.add_argument("--export", action="store_true",
                        help="Export best config to output/nba_config.json")
    args = parser.parse_args()

    cache_dir = Path(args.cache) if args.cache else YGGDRASIL_CACHE
    use_cv = not args.no_cv

    if not cache_dir.exists():
        print(f"Error: cache directory not found: {cache_dir}", file=sys.stderr)
        sys.exit(1)

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Evaluate current config for baseline
    console.print()
    console.print(" [dim]Evaluating current config baseline...[/]")
    current_cfg = {**NBA_MODEL, **FIXED_PARAMS}
    current_result = evaluate_config(current_cfg, cache_dir)

    # Run optimization
    target_info = TARGETS[args.target]
    study = optuna.create_study(direction=target_info["direction"])
    objective = make_objective(cache_dir, args.target, use_cv)

    with console.status(f"[bold]Optimizing... (target: {target_info['label']}, {args.trials} trials)[/]"):
        study.optimize(objective, n_trials=args.trials)

    # If CV was used, re-evaluate best config on full dataset for overfitting comparison
    best_full_result = None
    if use_cv:
        console.print(" [dim]Re-evaluating best config on full dataset...[/]")
        best_config = {**study.best_trial.params, **FIXED_PARAMS}
        best_full_result = evaluate_config(best_config, cache_dir)

    print_results(study, args.target, current_result, use_cv, best_full_result)

    if args.export:
        display_result = best_full_result if best_full_result is not None else \
            study.best_trial.user_attrs.get("result", {})
        total_games = int(display_result.get("total_games", 0))
        project_root = Path(__file__).resolve().parent.parent
        export_config(study, args.target, total_games,
                      project_root / "output" / "nba_config.json",
                      full_result=best_full_result)


if __name__ == "__main__":
    main()
