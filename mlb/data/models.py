"""MLB data models.

Typed dataclasses for all simulation data structures: player stats, park factors,
weather, lineups, game context, simulation state, and results.
"""
import logging
from dataclasses import dataclass, field
from typing import Any

from mlb.config import Hand, Outcome, WindDirection

logger = logging.getLogger(__name__)


# ── Validation ───────────────────────────────────────────────────────────────

def _validate_rates(
    rates: dict[str, float], tolerance: float, label: str
) -> None:
    """Check that rates dict has all Outcome keys and sums to ~1.0."""
    expected_keys = {o.value for o in Outcome}
    actual_keys = set(rates.keys())

    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing:
        logger.warning("%s: missing outcome keys: %s", label, missing)
    if extra:
        logger.warning("%s: unexpected outcome keys: %s", label, extra)

    total = sum(rates.values())
    if abs(total - 1.0) > tolerance:
        logger.warning(
            "%s: rates sum to %.4f, expected ~1.0 (tolerance=%.3f)",
            label, total, tolerance,
        )


# ── Handedness resolution ────────────────────────────────────────────────────

def resolve_batter_hand(bats: Hand, pitcher_throws: Hand) -> Hand:
    """Resolve effective batting hand. Switch hitters bat opposite the pitcher."""
    if bats == Hand.SWITCH:
        return Hand.LEFT if pitcher_throws == Hand.RIGHT else Hand.RIGHT
    return bats


# ── Base state ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BaseState:
    """Baserunner occupancy. Immutable — each PA produces a new state."""
    first: bool = False
    second: bool = False
    third: bool = False


# ── Player stats ─────────────────────────────────────────────────────────────

@dataclass
class BatterStats:
    """A batter's PA outcome rates for a specific handedness split."""
    player_id: str
    name: str
    bats: Hand              # L, R, or S (switch)
    pa: int                 # plate appearances in this split
    rates: dict[str, float] # outcome rates keyed by OUTCOMES, should sum to ~1.0
    data_source: str = "unknown"
    split_profile: dict[str, Any] | None = None

    def __post_init__(self):
        _validate_rates(self.rates, tolerance=0.02, label=f"BatterStats({self.name})")


@dataclass
class PitcherStats:
    """A pitcher's PA outcome rates allowed for a specific handedness split."""
    player_id: str
    name: str
    throws: Hand                # L or R
    pa_against: int             # plate appearances against in this split
    rates: dict[str, float]     # outcome rates allowed, keyed by OUTCOMES
    avg_pitch_count: float = 0.0  # average pitches per start (stamina modeling)
    data_source: str = "unknown"
    split_profile: dict[str, Any] | None = None

    def __post_init__(self):
        _validate_rates(self.rates, tolerance=0.02, label=f"PitcherStats({self.name})")


# ── Park & weather ───────────────────────────────────────────────────────────

@dataclass
class ParkFactors:
    """Park adjustments by batter handedness. Multipliers centered on 1.0."""
    venue_id: str
    venue_name: str
    factors_vs_lhb: dict[str, float]  # outcome multipliers for left-handed batters
    factors_vs_rhb: dict[str, float]  # outcome multipliers for right-handed batters

    def get_factors(self, batter_hand: Hand) -> dict[str, float]:
        """Return the factors dict for the given batter handedness."""
        if batter_hand == Hand.LEFT:
            return self.factors_vs_lhb
        return self.factors_vs_rhb


@dataclass
class Weather:
    """Game-day weather conditions."""
    temperature_f: float
    wind_speed_mph: float
    wind_direction: WindDirection
    humidity_pct: float
    is_indoor: bool = False  # True for domes or closed retractable roofs


# ── Lineup & game context ────────────────────────────────────────────────────

@dataclass
class Lineup:
    """A team's confirmed lineup for a game."""
    team_id: str
    team_name: str
    batting_order: list[BatterStats]      # 9 batters in order
    starting_pitcher: PitcherStats
    bullpen: list[PitcherStats]


@dataclass
class GameContext:
    """Everything needed to simulate one game."""
    game_id: str
    date: str
    away_lineup: Lineup
    home_lineup: Lineup
    park_factors: ParkFactors
    weather: Weather | None = None
    away_lineup_source: str = "confirmed"
    home_lineup_source: str = "confirmed"
    away_starter_source: str = "boxscore"
    home_starter_source: str = "boxscore"


# ── Simulation state ─────────────────────────────────────────────────────────

@dataclass
class GameState:
    """Mutable in-game state for the simulation engine."""
    inning: int = 1
    is_top: bool = True          # True = away batting
    outs: int = 0
    bases: BaseState = field(default_factory=BaseState)
    away_score: int = 0
    home_score: int = 0
    away_batting_index: int = 0  # position in batting order (0-8)
    home_batting_index: int = 0
    away_pitch_count: int = 0
    home_pitch_count: int = 0
    away_pitcher_outs: int = 0
    home_pitcher_outs: int = 0
    away_pitcher_runs_allowed: int = 0
    home_pitcher_runs_allowed: int = 0
    away_bullpen_index: int = -1  # -1 = starter still in, 0+ = bullpen index
    home_bullpen_index: int = -1


# ── Simulation results ───────────────────────────────────────────────────────

@dataclass
class PAResult:
    """Result of a single plate appearance."""
    outcome: Outcome
    batter_id: str
    pitcher_id: str
    inning: int
    runners_before: BaseState
    runs_scored: int


@dataclass
class SimulatedGame:
    """Result of one complete game simulation (single run)."""
    game_id: str
    away_runs: int
    home_runs: int
    away_hits: int
    home_hits: int
    pa_results: list[PAResult]
    innings_played: int
    inning_scores: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class PlayerSimStats:
    """Per-player aggregated stats across N simulations."""
    player_id: str
    name: str
    pa_per_game: float       # mean plate appearances per game
    hits_per_game: float     # mean hits (1B+2B+3B+HR) per game
    hr_per_game: float       # mean home runs per game
    bb_per_game: float       # mean walks per game
    k_per_game: float        # mean strikeouts per game
    runs_per_game: float     # mean runs scored (batters) or allowed (pitchers)
    doubles_per_game: float = 0.0
    hbp_per_game: float = 0.0
    total_bases_per_game: float = 0.0
    hits_per_game_std: float = 0.0
    hr_per_game_std: float = 0.0
    bb_per_game_std: float = 0.0
    k_per_game_std: float = 0.0
    total_bases_per_game_std: float = 0.0


@dataclass
class SimulationResult:
    """Aggregated result of N simulations for one game."""
    game_id: str
    n_simulations: int
    away_team: str
    home_team: str
    away_runs_mean: float
    away_runs_std: float
    home_runs_mean: float
    home_runs_std: float
    total_runs_mean: float
    total_runs_std: float
    home_win_pct: float
    away_win_pct: float
    player_stats: dict[str, PlayerSimStats]  # keyed by player_id
    betting_lines: dict = field(default_factory=dict)
    run_distributions: dict = field(default_factory=dict)
