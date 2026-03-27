from rich.console import Console
from rich.table import Table
from rich.text import Text
from config import WIN_THRESHOLD


console = Console()


def print_report(graded_games, metrics):
    """Print formatted backtest report to terminal."""
    date_range = metrics["date_range"]
    total_games = metrics["total_games"]
    v2 = metrics["v2"]
    v1 = metrics["v1"]

    # ── Banner ─────────────────────────────────────────────────────────────────
    console.print()
    console.print(" [bold white]NBA BACKTEST REPORT[/]")
    console.print(f" [dim]{'─' * 60}[/]")
    if date_range:
        console.print(f" [dim]Date range : {date_range['from']} → {date_range['to']}[/]")
    console.print(f" [dim]Total games: {total_games}[/]")

    # ── V2 model ───────────────────────────────────────────────────────────────
    _section_header("V2 MODEL")
    if not v2:
        console.print("  [dim]No V2 data.[/]")
    else:
        console.print(f"  Overall    {_stats_line(v2)}")
        miss_str = f"{v2['avg_miss']:.2f} pts" if v2["avg_miss"] is not None else "[dim]N/A[/]"
        console.print(f"  Avg miss   {miss_str}")

        # By confidence
        console.print()
        conf_table = Table(show_header=True, header_style="dim", box=None,
                           pad_edge=False, show_edge=False)
        conf_table.add_column("Confidence", style="white")
        conf_table.add_column("Record")
        conf_table.add_column("Bets", justify="right")
        conf_table.add_column("Win%", justify="right")
        conf_table.add_column("ROI", justify="right")

        for lvl, s in v2["by_confidence"].items():
            if s.get("note"):
                label = Text(f"{lvl}  ← hypothetical", style="dim")
            elif lvl == "HIGH":
                label = Text(lvl, style="green")
            elif lvl == "MEDIUM":
                label = Text(lvl, style="yellow")
            else:
                label = Text(lvl, style="dim")

            conf_table.add_row(
                label,
                _color_record(s),
                str(s["total_bets"]) if s["total_bets"] > 0 else "[dim]0[/]",
                _color_win_rate(s["win_rate"]),
                _color_roi(s["roi"]),
            )
        console.print(conf_table)

        # By direction
        console.print()
        console.print("  [dim]By direction:[/]")
        dir_table = Table(show_header=True, header_style="dim", box=None,
                          pad_edge=False, show_edge=False)
        dir_table.add_column("Direction", style="white")
        dir_table.add_column("Record")
        dir_table.add_column("Bets", justify="right")
        dir_table.add_column("Win%", justify="right")
        dir_table.add_column("ROI", justify="right")
        for d, s in v2["by_direction"].items():
            dir_table.add_row(d, _color_record(s), str(s["total_bets"]),
                              _color_win_rate(s["win_rate"]), _color_roi(s["roi"]))
        console.print(dir_table)

        # By gap size
        console.print()
        console.print("  [dim]By gap size (|proj - DK|):[/]")
        gap_table = Table(show_header=True, header_style="dim", box=None,
                          pad_edge=False, show_edge=False)
        gap_table.add_column("Gap |pts|", style="white")
        gap_table.add_column("Record")
        gap_table.add_column("Bets", justify="right")
        gap_table.add_column("Win%", justify="right")
        gap_table.add_column("ROI", justify="right")
        for bucket, s in v2["by_gap_size"].items():
            gap_table.add_row(bucket, _color_record(s), str(s["total_bets"]),
                              _color_win_rate(s["win_rate"]), _color_roi(s["roi"]))
        console.print(gap_table)

        # Calibration
        if v2.get("calibration"):
            console.print()
            console.print("  [dim]Calibration  (|z| bucket -> predicted vs actual win%):[/]")
            cal_table = Table(show_header=True, header_style="dim", box=None,
                              pad_edge=False, show_edge=False)
            cal_table.add_column("|z| bucket", style="white")
            cal_table.add_column("Predicted", justify="right")
            cal_table.add_column("Actual", justify="right")
            cal_table.add_column("n", justify="right")
            for bucket, c in v2["calibration"].items():
                pred = f"{c['predicted_win_prob'] * 100:.1f}%" if c["predicted_win_prob"] else "[dim]N/A[/]"
                n_str = str(c["count"])
                warn = f" [yellow]! n={c['count']}[/]" if c["count"] < 10 else f" [dim]n={c['count']}[/]"
                cal_table.add_row(
                    bucket,
                    f"[dim]{pred}[/]",
                    f"{_color_win_rate(c['actual_win_rate'])}{warn}",
                    n_str,
                )
            console.print(cal_table)

    # ── V1 baseline ────────────────────────────────────────────────────────────
    _section_header("V1 BASELINE")
    if not v1:
        console.print("  [dim]No V1 data.[/]")
    else:
        console.print(f"  Overall    {_stats_line(v1)}")
        miss_str = f"{v1['avg_miss']:.2f} pts" if v1["avg_miss"] is not None else "[dim]N/A[/]"
        console.print(f"  Avg miss   {miss_str}")

    # ── Game log ───────────────────────────────────────────────────────────────
    _section_header("GAME LOG")
    log_table = Table(show_header=True, header_style="dim", box=None,
                      pad_edge=False, show_edge=False)
    log_table.add_column("Date", style="dim", width=10)
    log_table.add_column("Matchup", width=36)
    log_table.add_column("DK", justify="right", width=6)
    log_table.add_column("Proj", justify="right", width=6)
    log_table.add_column("Actual", justify="right", width=6)
    log_table.add_column("Rec", width=8)
    log_table.add_column("V2", width=6)
    log_table.add_column("V1 line", justify="right", width=7)
    log_table.add_column("V1", width=6)
    log_table.add_column("OT", width=3)

    for g in graded_games:
        matchup = f"{g['away_team']} @ {g['home_team']}"
        if len(matchup) > 35:
            matchup = matchup[:34] + "…"

        rec = g.get("opening_recommendation", "-")
        if rec == "O":
            rec_str = "[cyan]O[/]"
        elif rec == "U":
            rec_str = "[magenta]U[/]"
        else:
            rec_str = f"[dim]{rec}[/]"

        v2_icon = _result_icon(g.get("v2_result"))
        rec_col = f"{rec_str} {v2_icon}"

        proj = f"{g['projected_total']:.1f}" if g.get("projected_total") is not None else "[dim]-[/]"
        v1l = f"{g['v1_line']:.1f}" if g.get("v1_line") is not None else "[dim]-[/]"
        ot_str = "[yellow]OT[/]" if g.get("went_to_ot") else ""

        log_table.add_row(
            g["date"],
            matchup,
            str(g.get("opening_dk_line", "-")),
            proj,
            str(g.get("actual_total", "-")),
            rec_col,
            _color_result(g.get("v2_result")),
            v1l,
            _color_result(g.get("v1_result")),
            ot_str,
        )

    console.print(log_table)
    console.print()


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _section_header(title):
    padding = max(0, 56 - len(title))
    console.print(f"\n [bold white]{'─' * 3} {title} {'─' * padding}[/]")


def _color_roi(roi):
    if roi is None:
        return "[dim]N/A[/]"
    sign = "+" if roi >= 0 else ""
    style = "green" if roi > 0 else "red"
    return f"[{style}]{sign}{roi:.2f}%[/]"


def _color_win_rate(rate):
    if rate is None:
        return "[dim]N/A[/]"
    style = "green" if rate >= WIN_THRESHOLD else "red"
    return f"[{style}]{rate * 100:.1f}%[/]"


def _color_record(stats):
    if stats["total_bets"] == 0 and stats["pushes"] == 0:
        return "[dim]0W-0L-0P[/]"
    return f"[green]{stats['wins']}W[/]-[red]{stats['losses']}L[/]-[dim]{stats['pushes']}P[/]"


def _color_result(result):
    if result == "WIN":
        return "[green]WIN[/]"
    if result == "LOSS":
        return "[red]LOSS[/]"
    if result == "PUSH":
        return "[yellow]PUSH[/]"
    return "[dim]-[/]"


def _result_icon(result):
    if result == "WIN":
        return "[green]✓[/]"
    if result == "LOSS":
        return "[red]✗[/]"
    return "[dim]–[/]"


def _stats_line(stats):
    """Format a one-line stats summary."""
    if stats["total_bets"] == 0:
        return "[dim]0W-0L-0P  (0 bets)  N/A  N/A[/]"
    record = _color_record(stats)
    return f"{record}  ({stats['total_bets']} bets)  {_color_win_rate(stats['win_rate'])}  {_color_roi(stats['roi'])}"
