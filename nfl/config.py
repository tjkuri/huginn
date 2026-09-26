from pathlib import Path

# Yggdrasil cache directory (sibling repo) — where nflLogger.js writes prediction snapshots
YGGDRASIL_CACHE = Path(__file__).parent.parent.parent / "yggdrasil" / "cache" / "nfl-predictions"

# Same URL template Yggdrasil's services/nflverseStats.js uses for weekly player stats
NFLVERSE_STATS_URL_TMPL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
