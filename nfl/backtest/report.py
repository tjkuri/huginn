from rich.console import Console
from rich.table import Table

console = Console()


def _fmt_pct(rate):
    return f"{rate * 100:.1f}%" if rate is not None else "n/a"


def _fmt_yds(value):
    return f"{value} yds" if value is not None else "n/a"


def _hit_icon(hit):
    if hit is True:
        return "[green]✓[/]"
    if hit is False:
        return "[red]✗[/]"
    return "[dim]–[/]"


def print_report(graded_predictions, metrics):
    console.print(
        f"\n[bold]NFL QB Verdict Backtest[/bold]  "
        f"({metrics['gradeable']}/{metrics['total_predictions']} gradeable)\n"
    )

    overall = metrics["overall_hit_rate"]
    color = "green" if overall["hit_rate"] is not None and overall["hit_rate"] >= 0.5 else "red" if overall["hit_rate"] is not None else "dim"
    console.print(f"Overall: n={overall['n']}  [{color}]{_fmt_pct(overall['hit_rate'])}[/{color}]")

    if metrics["by_confidence"]:
        table = Table(title="By confidence tier")
        table.add_column("Tier")
        table.add_column("n", justify="right")
        table.add_column("Hit rate", justify="right")
        for tier, stats in metrics["by_confidence"].items():
            table.add_row(tier, str(stats["n"]), _fmt_pct(stats["hit_rate"]))
        console.print(table)

    no_bet = metrics["no_bet_hypothetical"]
    console.print(
        f"\nNO_BET hypothetical: n={no_bet['n']}  hit rate={_fmt_pct(no_bet['hit_rate'])}  "
        f"[dim](near 50% = the gate is filtering noise correctly)[/dim]"
    )

    console.print(f"\nMean absolute error: {_fmt_yds(metrics['mean_abs_error'])}")
    for scope, mae in metrics["mean_abs_error_by_scope"].items():
        console.print(f"  {scope}: {_fmt_yds(mae)}")

    console.print()
    if not graded_predictions:
        console.print("[dim]No predictions logged yet.[/]")
        return

    table = Table(title="Predictions")
    table.add_column("Week", justify="right")
    table.add_column("Player")
    table.add_column("Line", justify="right")
    table.add_column("Proj", justify="right")
    table.add_column("Actual", justify="right")
    table.add_column("Rec")
    table.add_column("Hit")
    for p in graded_predictions:
        actual = p.get("actual_yards")
        adjusted_mean = p.get("adjusted_mean")
        table.add_row(
            str(p.get("week", "-")),
            p.get("player_name", "-"),
            str(p.get("line", "-")),
            f"{adjusted_mean:.1f}" if adjusted_mean is not None else "[dim]-[/]",
            f"{actual:.1f}" if actual is not None else "[dim]-[/]",
            p.get("verdict", {}).get("recommendation", "-"),
            _hit_icon(p.get("hit")),
        )
    console.print(table)


def print_comparison_report(results):
    """
    Side-by-side comparison of multiple strategies' metrics (from compare_strategies).
    results: dict[str, dict] — strategy name -> compute_metrics() output.
    """
    if not results:
        console.print("[dim]No strategy results to compare.[/]")
        return

    console.print("\n[bold]NFL Strategy Comparison[/bold]\n")

    overall_table = Table(title="Overall hit rate by strategy")
    overall_table.add_column("Strategy")
    overall_table.add_column("n", justify="right")
    overall_table.add_column("Hit rate", justify="right")
    overall_table.add_column("Gradeable", justify="right")
    overall_table.add_column("Total", justify="right")
    for name, metrics in results.items():
        overall = metrics["overall_hit_rate"]
        color = "green" if overall["hit_rate"] is not None and overall["hit_rate"] >= 0.5 else "red" if overall["hit_rate"] is not None else "dim"
        overall_table.add_row(
            name,
            str(overall["n"]),
            f"[{color}]{_fmt_pct(overall['hit_rate'])}[/{color}]",
            str(metrics["gradeable"]),
            str(metrics["total_predictions"]),
        )
    console.print(overall_table)

    all_tiers = sorted({tier for metrics in results.values() for tier in metrics["by_confidence"]})
    if all_tiers:
        tier_table = Table(title="By confidence tier")
        tier_table.add_column("Strategy")
        for tier in all_tiers:
            tier_table.add_column(tier, justify="right")
        for name, metrics in results.items():
            row = [name]
            for tier in all_tiers:
                stats = metrics["by_confidence"].get(tier)
                row.append(f"{_fmt_pct(stats['hit_rate'])} (n={stats['n']})" if stats else "[dim]-[/]")
            tier_table.add_row(*row)
        console.print(tier_table)

    mae_table = Table(title="Mean absolute error")
    mae_table.add_column("Strategy")
    mae_table.add_column("MAE (yds)", justify="right")
    for name, metrics in results.items():
        mae_table.add_row(name, _fmt_yds(metrics["mean_abs_error"]))
    console.print(mae_table)
    console.print()


def print_common_games_comparison(result):
    """Companion to print_comparison_report: hit rate restricted to games every strategy
    actually picked, so a reader can see a same-games comparison, not just each strategy's
    own (possibly very different) pick set."""
    n = result.get("n", 0)
    if n == 0:
        console.print("[dim]No games where every strategy made a gradeable, actionable pick -- no same-games comparison available.[/]")
        return
    table = Table(title=f"Same-games comparison (n={n} games every strategy picked)")
    table.add_column("Strategy")
    table.add_column("Hits", justify="right")
    table.add_column("Hit rate", justify="right")
    for name, stats in result["by_strategy"].items():
        table.add_row(name, str(stats["hits"]), _fmt_pct(stats["hit_rate"]))
    console.print(table)
