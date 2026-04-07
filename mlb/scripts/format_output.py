"""Terminal formatting helpers for MLB simulation CLI output."""
from __future__ import annotations

from collections import Counter
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


def _players_by_source(players: list[Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for player in players:
        source = str(getattr(player, "data_source", "unknown") or "unknown")
        # Normalize year-suffixed sources so "2026_split" and "2026_overall" both
        # group under "2026", matching what _build_quality_panel looks up.
        for suffix in ("_overall", "_split"):
            if source.endswith(suffix):
                source = source[: -len(suffix)]
                break
        grouped.setdefault(source, []).append(str(getattr(player, "name", "Unknown")))
    return grouped


def _wrap_names(names: list[str], indent: str = "    ", width: int | None = None) -> list[str]:
    if not names:
        return []
    if width is None:
        width = _CONTENT_WIDTH

    lines: list[str] = []
    current = indent
    for name in names:
        separator = "" if current == indent else ", "
        candidate = f"{current}{separator}{name}"
        if len(candidate) <= width:
            current = candidate
            continue
        lines.append(current)
        current = f"{indent}{name}"
    lines.append(current)
    return lines


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
                    "tb": "0.0",
                    "k_pct": "0.0%",
                    "two_plus_hits": "0.0%",
                    "hr_pct": "0.0%",
                }
            )
            continue

        hbp = getattr(stats, "hbp_per_game", 0.0)
        pa = max(0.001, stats.pa_per_game)
        ab = max(0.001, pa - stats.bb_per_game - hbp)
        avg = stats.hits_per_game / ab
        obp = (stats.hits_per_game + stats.bb_per_game + hbp) / pa
        rows.append(
            {
                "name": batter.name,
                "avg": f"{avg:.3f}".lstrip("0"),
                "obp": f"{obp:.3f}".lstrip("0"),
                "tb": f"{stats.total_bases_per_game:.1f}",
                "k_pct": f"{(stats.k_per_game / pa) * 100:.1f}%",
                "two_plus_hits": f"{two_plus_hits.get(batter.player_id, 0.0) * 100:.1f}%",
                "hr_pct": f"{(stats.hr_per_game / pa) * 100:.1f}%",
            }
        )
    return rows


def _starter_rows(game_context: GameContext, result: SimulationResult) -> list[dict[str, str]]:
    starters = [
        game_context.away_lineup.starting_pitcher,
        game_context.home_lineup.starting_pitcher,
    ]
    rows = []
    for starter in starters:
        stats = result.player_stats.get(starter.player_id)
        if stats is None:
            rows.append(
                {
                    "name": starter.name,
                    "ip": "--",
                    "k": "--",
                    "five_plus_k": "--",
                    "er": "--",
                    "qs_pct": "--",
                }
            )
            continue

        rows.append(
            {
                "name": starter.name,
                "ip": f"{getattr(stats, 'innings_pitched_per_game', 0.0):.1f}",
                "k": f"{stats.k_per_game:.1f}",
                "five_plus_k": f"{getattr(stats, 'k_5_plus_pct', 0.0) * 100:.1f}%",
                "er": f"{stats.runs_per_game:.1f}",
                "qs_pct": f"{getattr(stats, 'quality_start_pct', 0.0) * 100:.1f}%",
            }
        )
    return rows


def _bullpen_rows(game_context: GameContext, result: SimulationResult) -> list[dict[str, str]]:
    bullpens = []
    if game_context.away_lineup.bullpen:
        bullpens.append(game_context.away_lineup.bullpen[0])
    if game_context.home_lineup.bullpen:
        bullpens.append(game_context.home_lineup.bullpen[0])

    rows = []
    for bullpen in bullpens:
        stats = result.player_stats.get(bullpen.player_id)
        if stats is None:
            rows.append({"name": bullpen.name, "ip": "--", "k": "--", "er": "--"})
            continue
        rows.append(
            {
                "name": bullpen.name,
                "ip": f"{getattr(stats, 'innings_pitched_per_game', 0.0):.1f}",
                "k": f"{stats.k_per_game:.1f}",
                "er": f"{stats.runs_per_game:.1f}",
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
            f"{row['name']}: AVG {row['avg']} OBP {row['obp']} TB {row['tb']} K% {row['k_pct']} 2+H {row['two_plus_hits']} HR% {row['hr_pct']}"
        )
    lines.append("Starting pitchers:")
    for row in _starter_rows(game_context, result):
        lines.append(
            f"{row['name']}: IP {row['ip']} K {row['k']} 5+K% {row['five_plus_k']} ER {row['er']} QS% {row['qs_pct']}"
        )
    bullpen_rows = _bullpen_rows(game_context, result)
    if bullpen_rows:
        lines.append("Bullpen:")
        for row in bullpen_rows:
            lines.append(f"{row['name']}: IP {row['ip']} K {row['k']} ER {row['er']}")
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
    batters = game_context.away_lineup.batting_order + game_context.home_lineup.batting_order
    pitchers = [
        game_context.away_lineup.starting_pitcher,
        game_context.home_lineup.starting_pitcher,
    ]
    batter_sources = _players_by_source(batters)
    pitcher_sources = _players_by_source(pitchers)
    default_weather = _is_default_weather(game_context)
    away_lineup_fallback = game_context.away_lineup_source == "fallback_roster_order"
    home_lineup_fallback = game_context.home_lineup_source == "fallback_roster_order"
    away_starter_fallback = game_context.away_starter_source == "first_roster_arm"
    home_starter_fallback = game_context.home_starter_source == "first_roster_arm"

    has_player_warnings = bool(
        batter_sources.get("2025") or batter_sources.get("league_avg") or
        pitcher_sources.get("2025") or pitcher_sources.get("league_avg")
    )
    has_lineup_warnings = away_lineup_fallback or home_lineup_fallback or away_starter_fallback or home_starter_fallback
    if not has_player_warnings and not data_warnings and not default_weather and not has_lineup_warnings:
        return None

    lines: list[Text] = []

    # Source-based player stats — current year shown without names, fallbacks listed by name
    def add_source_group(
        grouped: dict[str, list[str]],
        source: str,
        label: str,
        description: str,
        icon: str,
        style: str,
    ) -> None:
        names = grouped.get(source, [])
        if not names:
            return
        noun = label if len(names) == 1 else f"{label}s"
        lines.append(Text(f"{icon} {len(names)} {noun} using {description}:", style=style))
        for wrapped_line in _wrap_names(names):
            lines.append(Text(f"  {wrapped_line}", style=style))

    for source, label in (("2026", "batter"), ("2026", "pitcher")):
        grouped = batter_sources if label == "batter" else pitcher_sources
        names = grouped.get(source, [])
        if names:
            noun = label if len(names) == 1 else f"{label}s"
            lines.append(Text(f"✓ {len(names)} {noun} using 2026 stats", style="green"))
    add_source_group(batter_sources, "2025", "batter", "2025 stats", "⚠", "yellow")
    add_source_group(batter_sources, "league_avg", "batter", "league-average fallback", "⚠", "bright_yellow")
    add_source_group(pitcher_sources, "2025", "pitcher", "2025 stats", "⚠", "yellow")
    add_source_group(pitcher_sources, "league_avg", "pitcher", "league-average fallback", "⚠", "bright_yellow")

    if away_lineup_fallback:
        lines.append(Text("⚠ Away lineup: fallback active-roster order (confirmed lineup unavailable)", style="yellow"))
    if home_lineup_fallback:
        lines.append(Text("⚠ Home lineup: fallback active-roster order (confirmed lineup unavailable)", style="yellow"))
    if away_lineup_fallback or home_lineup_fallback:
        lines.append(Text("  Fallback lineups may differ materially from the eventual batting order.", style="yellow"))
    if away_starter_fallback:
        lines.append(Text("⚠ Away starter: roster fallback (no probable pitcher listed)", style="yellow"))
    if home_starter_fallback:
        lines.append(Text("⚠ Home starter: roster fallback (no probable pitcher listed)", style="yellow"))

    lines.append(Text(f"✓ Park factors: {game_context.park_factors.venue_name}", style="green"))
    if default_weather:
        lines.append(Text("⚠ Weather: default placeholder (no real data)", style="yellow"))

    return Panel(Group(*lines), title="DATA QUALITY", border_style="yellow")


def _build_batter_block(team_name: str, rows: list[dict[str, str]]):
    lines: list[Text] = [
        _center_line(team_name),
        _column_line(
            [
                ("Player", 32, "left"),
                ("AVG", 7, "right"),
                ("OBP", 7, "right"),
                ("TB", 7, "right"),
                ("K%", 7, "right"),
                ("2+H%", 8, "right"),
                ("HR%", 7, "right"),
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
                    (row["tb"], 7, "right"),
                    (row["k_pct"], 7, "right"),
                    (row["two_plus_hits"], 8, "right"),
                    (row["hr_pct"], 7, "right"),
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
                ("IP", 6, "right"),
                ("K", 6, "right"),
                ("5+K%", 8, "right"),
                ("ER", 6, "right"),
                ("QS%", 8, "right"),
            ],
            style="bold",
        ),
    ]
    for row in rows:
        lines.append(
            _column_line(
                [
                    ("", 32, "left"),
                    (row["ip"], 6, "right"),
                    (row["k"], 6, "right"),
                    (row["five_plus_k"], 8, "right"),
                    (row["er"], 6, "right"),
                    (row["qs_pct"], 8, "right"),
                ],
                dot_leader_name=row["name"],
            )
        )
    return Group(*lines)


def _build_bullpen_block(rows: list[dict[str, str]]):
    lines: list[Text] = [
        _center_line("Bullpen"),
        _column_line(
            [
                ("Team", 32, "left"),
                ("IP", 6, "right"),
                ("K", 6, "right"),
                ("ER", 6, "right"),
            ],
            style="bold",
        ),
    ]
    for row in rows:
        lines.append(
            _column_line(
                [
                    ("", 32, "left"),
                    (row["ip"], 6, "right"),
                    (row["k"], 6, "right"),
                    (row["er"], 6, "right"),
                ],
                dot_leader_name=row["name"],
            )
        )
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
    quality_panel = _build_quality_panel(game_context, data_warnings)
    if quality_panel is not None:
        sections.append(quality_panel)

    away_rows = _batter_rate_rows(game_context, result, all_games, is_home=False)
    home_rows = _batter_rate_rows(game_context, result, all_games, is_home=True)
    pitcher_rows = _starter_rows(game_context, result)
    bullpen_rows = _bullpen_rows(game_context, result)
    player_blocks: list[Any] = [
        _build_batter_block(result.away_team, away_rows),
        Text(""),
        _build_batter_block(result.home_team, home_rows),
        Text(""),
        _build_pitcher_block(pitcher_rows),
    ]
    if bullpen_rows:
        player_blocks.extend([Text(""), _build_bullpen_block(bullpen_rows)])
    sections.append(
        Panel(
            Group(*player_blocks),
            title=f"PLAYER PROJECTIONS (matchup-adjusted, {result.n_simulations:,} sims)",
            border_style="magenta",
        )
    )
    return Group(*sections)
