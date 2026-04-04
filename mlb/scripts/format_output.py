"""Terminal formatting helpers for MLB simulation CLI output."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from mlb.config import WindDirection
from mlb.data.models import GameContext, SimulatedGame, SimulationResult

try:
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:  # pragma: no cover
    Group = Panel = Table = Text = None
    HAS_RICH = False


TEAM_ABBREVIATIONS = {
    "Arizona Diamondbacks": "ARI",
    "Athletics": "ATH",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
    "Synthetic Away": "AWY",
    "Synthetic Home": "HME",
}


def _abbrev(team_name: str) -> str:
    return TEAM_ABBREVIATIONS.get(team_name, team_name[:3].upper())


def _american(probability: float) -> str:
    probability = max(0.001, min(0.999, probability))
    if probability > 0.5:
        value = -(probability / (1.0 - probability)) * 100.0
    elif probability < 0.5:
        value = ((1.0 - probability) / probability) * 100.0
    else:
        value = 100.0
    return f"{value:+.0f}"


def _count_batter_sources(game_context: GameContext) -> dict[str, int]:
    counts = {"2026": 0, "2025": 0, "league_avg": 0, "other": 0}
    for batter in game_context.away_lineup.batting_order + game_context.home_lineup.batting_order:
        source = getattr(batter, "data_source", "other")
        counts[source if source in counts else "other"] += 1
    return counts


def _group_missing_players(data_warnings: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for warning in data_warnings:
        if not warning.startswith("Missing "):
            continue
        if "Missing batting data for " in warning:
            name = warning.split("Missing batting data for ", 1)[1].split(";", 1)[0]
            grouped["batting"].append(name)
        elif "Missing pitching data for " in warning:
            name = warning.split("Missing pitching data for ", 1)[1].split(";", 1)[0]
            grouped["pitching"].append(name)
    return grouped


def _is_default_weather(game_context: GameContext) -> bool:
    weather = game_context.weather
    if weather is None:
        return False
    return (
        abs(weather.temperature_f - 72.0) < 1e-9
        and abs(weather.wind_speed_mph - 5.0) < 1e-9
        and abs(weather.humidity_pct - 50.0) < 1e-9
        and weather.wind_direction == WindDirection.CALM
    )


def _weather_text(game_context: GameContext) -> str:
    weather = game_context.weather
    if weather is None:
        return "No weather data"
    default_note = " (default - no real data)" if _is_default_weather(game_context) else ""
    indoor_label = "indoor" if weather.is_indoor else "outdoor"
    return (
        f"{weather.temperature_f:.0f}F, {weather.wind_speed_mph:.0f}mph "
        f"{weather.wind_direction.value.replace('_', ' ')} ({indoor_label}){default_note}"
    )


def _pad_display(text: str, width: int) -> str:
    trimmed = text[:width]
    return trimmed.ljust(width)


_CONTENT_WIDTH = 86  # Console width (90) minus 4 for panel borders

def _center_line(text: str, width: int = _CONTENT_WIDTH) -> Text:
    return Text(_pad_display(text.center(width), width), style="bold")


def _column_line(
    values: list[tuple[str, int, str]],
    style: str | None = None,
    dot_leader_name: str | None = None,
    dot_width: int = 0,
) -> Text:
    line = Text(style=style or "")
    for index, (value, width, align) in enumerate(values):
        if index == 0 and dot_leader_name is not None:
            name = dot_leader_name[:width]
            filler = max(0, width - len(name))
            line.append(name, style=style or "white")
            if filler:
                line.append("." * filler, style="dim")
            continue

        text = value[:width] if align == "left" else value
        if len(text) > width:
            text = text[:width]
        if align == "right":
            padded = text.rjust(width)
        elif align == "center":
            padded = text.center(width)
        else:
            padded = text.ljust(width)
        line.append(padded, style=style or "")
    return line


def _batter_rate_rows(
    game_context: GameContext,
    result: SimulationResult,
    simulated_games: list[SimulatedGame],
    is_home: bool,
) -> list[dict[str, str]]:
    lineup = game_context.home_lineup if is_home else game_context.away_lineup
    two_plus_hits: dict[str, float] = {}
    if simulated_games:
        hit_outcomes = {"1B", "2B", "3B", "HR"}
        two_plus_counts = Counter()
        for game in simulated_games:
            per_game_hits = Counter(
                pa.batter_id for pa in game.pa_results if pa.outcome.value in hit_outcomes
            )
            for player_id, hits in per_game_hits.items():
                if hits >= 2:
                    two_plus_counts[player_id] += 1
        total_games = len(simulated_games)
        two_plus_hits = {pid: count / total_games for pid, count in two_plus_counts.items()}

    rows = []
    for batter in lineup.batting_order:
        stats = result.player_stats.get(batter.player_id)
        if stats is None:
            rows.append(
                {
                    "name": batter.name,
                    "avg": ".000",
                    "obp": ".000",
                    "slg": ".000",
                    "k_pct": "0.0%",
                    "hr_pct": "0.0%",
                    "two_plus_hits": "0.0%",
                }
            )
            continue

        hbp = getattr(stats, "hbp_per_game", 0.0)
        pa = max(0.001, stats.pa_per_game)
        ab = max(0.001, pa - stats.bb_per_game - hbp)
        avg = stats.hits_per_game / ab
        obp = (stats.hits_per_game + stats.bb_per_game + hbp) / pa
        slg = stats.total_bases_per_game / ab
        rows.append(
            {
                "name": batter.name,
                "avg": f"{avg:.3f}".lstrip("0"),
                "obp": f"{obp:.3f}".lstrip("0"),
                "slg": f"{slg:.3f}".lstrip("0"),
                "k_pct": f"{(stats.k_per_game / pa) * 100:.1f}%",
                "hr_pct": f"{(stats.hr_per_game / pa) * 100:.1f}%",
                "two_plus_hits": f"{two_plus_hits.get(batter.player_id, 0.0) * 100:.1f}%",
            }
        )
    return rows


def _starter_rows(game_context: GameContext, simulated_games: list[SimulatedGame]) -> list[dict[str, str]]:
    starters = [
        (game_context.away_lineup.starting_pitcher, _abbrev(game_context.away_lineup.team_name)),
        (game_context.home_lineup.starting_pitcher, _abbrev(game_context.home_lineup.team_name)),
    ]
    rows = []
    for starter, team in starters:
        if not simulated_games:
            rows.append(
                {
                    "name": starter.name,
                    "team": team,
                    "ip": "--",
                    "k9": f"{starter.rates.get('K', 0.0) * 100:.1f}%",
                    "h9": f"{(starter.rates.get('1B', 0.0) + starter.rates.get('2B', 0.0) + starter.rates.get('3B', 0.0) + starter.rates.get('HR', 0.0)) * 100:.1f}%",
                    "era": "input",
                    "note": "TODO simulated starter line requires full simulated_games input",
                }
            )
            continue

        outs = strikeouts = hits_allowed = runs_allowed = 0
        for game in simulated_games:
            for pa in game.pa_results:
                if pa.pitcher_id != starter.player_id:
                    continue
                if pa.outcome.value in {"K", "OUT"}:
                    outs += 1
                if pa.outcome.value == "K":
                    strikeouts += 1
                if pa.outcome.value in {"1B", "2B", "3B", "HR"}:
                    hits_allowed += 1
                runs_allowed += pa.runs_scored
        games_n = len(simulated_games)
        ip = (outs / 3.0) / games_n if games_n else 0.0
        if ip > 0:
            k9 = ((strikeouts / games_n) / ip) * 9.0
            h9 = ((hits_allowed / games_n) / ip) * 9.0
            era = ((runs_allowed / games_n) / ip) * 9.0
        else:
            k9 = h9 = era = 0.0
        rows.append(
            {
                "name": starter.name,
                "team": team,
                "ip": f"{ip:.1f}",
                "k9": f"{k9:.1f}",
                "h9": f"{h9:.1f}",
                "era": f"{era:.2f}",
                "note": "",
            }
        )
    return rows


def _build_plain_report(
    result: SimulationResult,
    game_context: GameContext,
    sample_game: SimulatedGame | None,
    sample_index: int | None,
    data_warnings: list[str],
    simulated_games: list[SimulatedGame],
) -> str:
    lines = []
    header = (
        f"{_abbrev(result.away_team)} @ {_abbrev(result.home_team)}  |  "
        f"{game_context.park_factors.venue_name}  |  {game_context.date}  |  {result.n_simulations:,} sims"
    )
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("=" * len(header))
    if sample_game is not None and sample_index is not None:
        lines.append(f"Sample game: sim #{sample_index:,}")
        lines.append(f"{result.away_team}: {' '.join(str(v) for v in sample_game.inning_scores.get('away', []))}")
        lines.append(f"{result.home_team}: {' '.join(str(v) for v in sample_game.inning_scores.get('home', []))}")
    lines.append(f"Total runs: {result.total_runs_mean:.1f} +/- {result.total_runs_std:.1f}")
    lines.append(f"Moneyline: {_abbrev(result.away_team)} {_american(result.away_win_pct)} | {_abbrev(result.home_team)} {_american(result.home_win_pct)}")
    lines.append("Batters:")
    for row in _batter_rate_rows(game_context, result, simulated_games, is_home=False):
        lines.append(
            f"{row['name']}: AVG {row['avg']} OBP {row['obp']} SLG {row['slg']} K% {row['k_pct']} HR% {row['hr_pct']} 2+H {row['two_plus_hits']}"
        )
    return "\n".join(lines)


def _build_linescore_table(result: SimulationResult, sample_game: SimulatedGame, sample_index: int):
    innings = max(9, len(sample_game.inning_scores.get("away", [])), len(sample_game.inning_scores.get("home", [])))
    table = Table(box=None, expand=True, show_header=True, pad_edge=False)
    table.add_column("", style="bold")
    for inning in range(1, innings + 1):
        table.add_column(str(inning), justify="center", width=3)
    table.add_column("R", justify="right", style="bold cyan")
    table.add_column("H", justify="right", style="bold cyan")

    away_cells = [
        str(sample_game.inning_scores.get("away", [])[idx]) if idx < len(sample_game.inning_scores.get("away", [])) else ""
        for idx in range(innings)
    ]
    home_cells = []
    home_scores = sample_game.inning_scores.get("home", [])
    away_scores = sample_game.inning_scores.get("away", [])
    for idx in range(innings):
        if idx < len(home_scores):
            home_cells.append(str(home_scores[idx]))
        elif idx == len(away_scores) - 1 and idx >= 8 and sample_game.home_runs > sample_game.away_runs:
            home_cells.append("X")
        else:
            home_cells.append("")

    table.add_row(result.away_team, *away_cells, str(sample_game.away_runs), str(sample_game.away_hits))
    table.add_row(result.home_team, *home_cells, str(sample_game.home_runs), str(sample_game.home_hits))
    return Panel(table, title=f"SAMPLE GAME (sim #{sample_index:,} of {result.n_simulations:,})", border_style="cyan")


def _build_summary_panel(result: SimulationResult, game_context: GameContext):
    grid = Table.grid(padding=(0, 1))
    grid.add_row("Total runs:", f"{result.total_runs_mean:.1f} +/- {result.total_runs_std:.1f}")
    grid.add_row(f"{result.away_team}:", f"{result.away_runs_mean:.1f} +/- {result.away_runs_std:.1f}")
    grid.add_row(f"{result.home_team}:", f"{result.home_runs_mean:.1f} +/- {result.home_runs_std:.1f}")
    grid.add_row("", "")
    grid.add_row(
        "Win probability:",
        f"{_abbrev(result.away_team)} {result.away_win_pct * 100:.1f}%  ·  {_abbrev(result.home_team)} {result.home_win_pct * 100:.1f}%",
    )
    grid.add_row("", "")
    grid.add_row("Over/under:", "")
    for line in (8.5, 9.0, 9.5):
        market = result.betting_lines.get("totals", {}).get(line)
        if market:
            over = market["over_pct"]
            under = market["under_pct"]
            denom = over + under
            over_display = over / denom * 100 if denom > 0 else 50.0
            under_display = under / denom * 100 if denom > 0 else 50.0
            grid.add_row(
                f"  O/U {line:.1f}",
                f"Over {over_display:.1f}%  |  Under {under_display:.1f}%",
            )
    grid.add_row("", "")
    grid.add_row(
        "Moneyline (no-vig):",
        f"{_abbrev(result.home_team)} {_american(result.home_win_pct)}  ·  {_abbrev(result.away_team)} {_american(result.away_win_pct)}",
    )
    grid.add_row("", "")
    lhr = game_context.park_factors.factors_vs_lhb.get("HR", 1.0)
    rhr = game_context.park_factors.factors_vs_rhb.get("HR", 1.0)
    grid.add_row("Modifiers applied:", "")
    grid.add_row("  Park:", f"{game_context.park_factors.venue_name} (HRx{lhr:.2f} LHB, HRx{rhr:.2f} RHB)")
    grid.add_row("  Weather:", _weather_text(game_context))
    return Panel(grid, title="PROJECTION SUMMARY", border_style="blue")


def _build_quality_panel(game_context: GameContext, data_warnings: list[str]):
    lines = []
    source_counts = _count_batter_sources(game_context)
    if source_counts["2026"]:
        lines.append(Text(f"✓ {source_counts['2026']} batters using 2026 stats", style="green"))
    if source_counts["2025"]:
        lines.append(Text(f"⚠ {source_counts['2025']} batters using 2025 fallback", style="yellow"))
    if source_counts["league_avg"]:
        lines.append(Text(f"⚠ {source_counts['league_avg']} batters using league-average fallback", style="yellow"))
    lines.append(Text(f"✓ Park factors: {game_context.park_factors.venue_name}", style="green"))
    weather_style = "yellow" if _is_default_weather(game_context) else "green"
    weather_prefix = "⚠" if _is_default_weather(game_context) else "✓"
    weather_note = "default placeholder (no real data)" if _is_default_weather(game_context) else _weather_text(game_context)
    lines.append(Text(f"{weather_prefix} Weather: {weather_note}", style=weather_style))

    missing = _group_missing_players(data_warnings)
    if missing:
        lines.append(Text(""))
        lines.append(Text("Missing players:", style="bold"))
        for category in ("batting", "pitching"):
            if missing.get(category):
                lines.append(Text(f"  {', '.join(missing[category])} ({category})", style="yellow"))
    return Panel(Group(*lines), title="DATA QUALITY", border_style="yellow")


def _build_batter_block(team_name: str, rows: list[dict[str, str]]):
    lines: list[Text] = [
        _center_line(team_name),
        _column_line(
            [
                ("Player", 32, "left"),
                ("AVG", 7, "right"),
                ("OBP", 7, "right"),
                ("SLG", 7, "right"),
                ("K%", 7, "right"),
                ("HR%", 7, "right"),
                ("2+ H%", 8, "right"),
            ],
            style="bold",
        ),
    ]
    for row in rows:
        lines.append(
            _column_line(
                [
                    ("", 32, "left"),
                    (row["avg"], 7, "right"),
                    (row["obp"], 7, "right"),
                    (row["slg"], 7, "right"),
                    (row["k_pct"], 7, "right"),
                    (row["hr_pct"], 7, "right"),
                    (row["two_plus_hits"], 8, "right"),
                ],
                dot_leader_name=row["name"],
            )
        )
    return Group(*lines)


def _build_pitcher_block(rows: list[dict[str, str]]):
    lines: list[Text] = [
        _center_line("Starting pitchers"),
        _column_line(
            [
                ("Pitcher", 32, "left"),
                ("Team", 6, "right"),
                ("IP", 6, "right"),
                ("K/9", 7, "right"),
                ("H/9", 7, "right"),
                ("ERA*", 8, "right"),
            ],
            style="bold",
        ),
    ]
    note = ""
    for row in rows:
        lines.append(
            _column_line(
                [
                    ("", 32, "left"),
                    (row["team"], 6, "right"),
                    (row["ip"], 6, "right"),
                    (row["k9"], 7, "right"),
                    (row["h9"], 7, "right"),
                    (row["era"], 8, "right"),
                ],
                dot_leader_name=row["name"],
            )
        )
        if row.get("note"):
            note = row["note"]
    if note:
        lines.append(Text(note, style="dim"))
    else:
        lines.append(Text("* projected from simulation averages", style="dim"))
    return Group(*lines)


def build_terminal_output(
    result: SimulationResult,
    game_context: GameContext,
    sample_game: SimulatedGame | None,
    sample_index: int | None,
    data_warnings: list[str],
    simulated_games: list[SimulatedGame] | None = None,
):
    """Build a Rich renderable or plain-text report for one game."""
    all_games = simulated_games or []
    if not HAS_RICH:
        return _build_plain_report(result, game_context, sample_game, sample_index, data_warnings, all_games)

    header = Text(
        f"{_abbrev(result.away_team)} @ {_abbrev(result.home_team)}  ·  "
        f"{game_context.park_factors.venue_name}  ·  {game_context.date}  ·  {result.n_simulations:,} sims",
        style="bold white",
        justify="center",
    )
    sections = [Panel(header, border_style="bright_black")]
    if sample_game is not None and sample_index is not None:
        sections.append(_build_linescore_table(result, sample_game, sample_index))
    sections.append(_build_summary_panel(result, game_context))
    sections.append(_build_quality_panel(game_context, data_warnings))

    away_rows = _batter_rate_rows(game_context, result, all_games, is_home=False)
    home_rows = _batter_rate_rows(game_context, result, all_games, is_home=True)
    pitcher_rows = _starter_rows(game_context, all_games)
    sections.append(
        Panel(
            Group(
                _build_batter_block(result.away_team, away_rows),
                Text(""),
                _build_batter_block(result.home_team, home_rows),
                Text(""),
                _build_pitcher_block(pitcher_rows),
            ),
            title=f"PLAYER PROJECTIONS (per-game rates from {result.n_simulations:,} sims)",
            border_style="magenta",
        )
    )
    return Group(*sections)
