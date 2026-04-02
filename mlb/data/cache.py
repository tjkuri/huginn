"""File-based JSON caching with TTL support.

Cache files are stored under CACHE_DIR (baseball_cache/ at repo root).
Pattern: {CACHE_DIR}/{category}/{date}-{key}.json  (dated)
         {CACHE_DIR}/{category}/{key}.json          (undated)
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from mlb.config import CACHE_DIR

logger = logging.getLogger(__name__)


def get_cache_path(
    category: str, key: str, date: str | None = None
) -> Path:
    """Return the filesystem path for a cache entry."""
    if date:
        return CACHE_DIR / category / f"{date}-{key}.json"
    return CACHE_DIR / category / f"{key}.json"


def read_cache(
    category: str,
    key: str,
    date: str | None = None,
    max_age_days: int | None = None,
) -> dict | None:
    """Read a cached JSON file. Returns None if missing or expired."""
    path = get_cache_path(category, key, date)
    if not path.exists():
        return None

    if max_age_days is not None:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        age = datetime.now(tz=timezone.utc) - mtime
        if age.days > max_age_days:
            logger.debug(
                "Cache expired: %s (age=%dd, max=%dd)", path, age.days, max_age_days
            )
            return None

    with open(path) as f:
        return json.load(f)


def write_cache(
    category: str,
    key: str,
    data: dict,
    date: str | None = None,
) -> Path:
    """Write data to cache as JSON. Creates directories as needed."""
    path = get_cache_path(category, key, date)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug("Cache written: %s", path)
    return path
