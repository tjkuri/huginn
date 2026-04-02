from pathlib import Path

# Yggdrasil cache directory (sibling repo)
YGGDRASIL_CACHE = Path(__file__).parent.parent.parent / "yggdrasil" / "cache"

# Vig constants for ROI at -110
VIG_RISK = 100       # dollars risked per bet
VIG_WIN = 90.91      # dollars won per bet at -110

# Break-even win rate at -110
WIN_THRESHOLD = 0.524

# Model config reference (mirrors yggdrasil/config/nba.js)
# Not used by backtest — included for future optimization phases.
NBA_MODEL = {
    "sample_size": 10,
    "decay_factor": 0.96,
    "min_z_threshold": 0.5,
    "z_medium": 0.8,
    "z_high": 1.5,
    "home_boost": 1.5,
    "min_home_away_games": 4,
}
