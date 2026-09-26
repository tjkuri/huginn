from rich.console import Console
from rich.table import Table

console = Console()


def _fmt_pct(rate):
    return f"{rate * 100:.1f}%" if rate is not None else "n/a"


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

    console.print(f"\nMean absolute error: {metrics['mean_abs_error']} yds")
    for scope, mae in metrics["mean_abs_error_by_scope"].items():
        console.print(f"  {scope}: {mae} yds")
