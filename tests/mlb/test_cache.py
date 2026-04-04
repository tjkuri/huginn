import json
import os
import time

import pytest

from mlb.config import SEASON
from mlb.data.stats import _load_raw_cache, _raw_cache_path, _save_raw_cache


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    """Point CACHE_DIR at a temp directory for each test."""
    import mlb.config as cfg
    import mlb.data.stats as stats_mod
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(stats_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(stats_mod, "SEASON", SEASON)
    return tmp_path


class TestRawCachePath:
    def test_flat_filename(self, cache_dir):
        path = _raw_cache_path("batting", 2025)
        assert path == cache_dir / "raw_batting-2025.json"

    def test_no_subdirectory(self, cache_dir):
        path = _raw_cache_path("pitching", 2026)
        assert path.parent == cache_dir


class TestSaveRawCache:
    def test_writes_json_file(self, cache_dir):
        players = {"aaron judge": {"name": "Aaron Judge", "pa": 600}}
        _save_raw_cache("batting", 2025, players)
        path = cache_dir / "raw_batting-2025.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["players"] == players

    def test_creates_cache_dir_if_missing(self, tmp_path, monkeypatch):
        import mlb.data.stats as stats_mod
        nested = tmp_path / "new_dir"
        monkeypatch.setattr(stats_mod, "CACHE_DIR", nested)
        _save_raw_cache("batting", 2025, {"p": {}})
        assert (nested / "raw_batting-2025.json").exists()


class TestLoadRawCache:
    def test_returns_none_when_missing(self, cache_dir):
        assert _load_raw_cache("batting", 2025) is None

    def test_round_trip_prior_season(self, cache_dir, monkeypatch):
        import mlb.data.stats as stats_mod
        monkeypatch.setattr(stats_mod, "SEASON", 2026)
        players = {"juan soto": {"name": "Juan Soto", "pa": 550}}
        _save_raw_cache("batting", 2025, players)
        result = _load_raw_cache("batting", 2025)
        assert result == players

    def test_prior_season_never_expires(self, cache_dir, monkeypatch):
        import mlb.data.stats as stats_mod
        monkeypatch.setattr(stats_mod, "SEASON", 2026)
        players = {"old player": {"name": "Old Player", "pa": 400}}
        _save_raw_cache("batting", 2025, players)
        # Backdate file by 10 years
        path = cache_dir / "raw_batting-2025.json"
        old_time = time.time() - (86400 * 3650)
        os.utime(path, (old_time, old_time))
        result = _load_raw_cache("batting", 2025)
        assert result == players

    def test_current_season_fresh_cache_returned(self, cache_dir, monkeypatch):
        import mlb.data.stats as stats_mod
        monkeypatch.setattr(stats_mod, "SEASON", 2026)
        players = {"new player": {"name": "New Player", "pa": 100}}
        _save_raw_cache("batting", 2026, players)
        result = _load_raw_cache("batting", 2026)
        assert result == players

    def test_current_season_stale_cache_returns_none(self, cache_dir, monkeypatch):
        import mlb.data.stats as stats_mod
        monkeypatch.setattr(stats_mod, "SEASON", 2026)
        monkeypatch.setattr(stats_mod, "STATS_CACHE_MAX_AGE_HOURS", 6)
        players = {"stale player": {"name": "Stale Player", "pa": 100}}
        _save_raw_cache("batting", 2026, players)
        # Backdate file by 8 hours (beyond the 6h TTL)
        path = cache_dir / "raw_batting-2026.json"
        old_time = time.time() - (3600 * 8)
        os.utime(path, (old_time, old_time))
        result = _load_raw_cache("batting", 2026)
        assert result is None

    def test_current_season_within_ttl_returned(self, cache_dir, monkeypatch):
        import mlb.data.stats as stats_mod
        monkeypatch.setattr(stats_mod, "SEASON", 2026)
        monkeypatch.setattr(stats_mod, "STATS_CACHE_MAX_AGE_HOURS", 6)
        players = {"fresh player": {"name": "Fresh Player", "pa": 100}}
        _save_raw_cache("batting", 2026, players)
        # Backdate by only 2 hours (within the 6h TTL)
        path = cache_dir / "raw_batting-2026.json"
        recent_time = time.time() - (3600 * 2)
        os.utime(path, (recent_time, recent_time))
        result = _load_raw_cache("batting", 2026)
        assert result == players
