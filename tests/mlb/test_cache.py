import os
import time

from mlb.data.cache import get_cache_path, read_cache, write_cache


class TestGetCachePath:
    def test_with_date(self, tmp_path, monkeypatch):
        monkeypatch.setattr('mlb.data.cache.CACHE_DIR', tmp_path)
        path = get_cache_path('stats', 'batting', date='2026-04-01')
        assert path == tmp_path / 'stats' / '2026-04-01-batting.json'

    def test_without_date(self, tmp_path, monkeypatch):
        monkeypatch.setattr('mlb.data.cache.CACHE_DIR', tmp_path)
        path = get_cache_path('stats', 'batting')
        assert path == tmp_path / 'stats' / 'batting.json'

    def test_nested_category(self, tmp_path, monkeypatch):
        monkeypatch.setattr('mlb.data.cache.CACHE_DIR', tmp_path)
        path = get_cache_path('players/splits', 'judge', date='2026-04-01')
        assert path == tmp_path / 'players/splits' / '2026-04-01-judge.json'


class TestWriteCache:
    def test_creates_directories(self, tmp_path, monkeypatch):
        monkeypatch.setattr('mlb.data.cache.CACHE_DIR', tmp_path)
        write_cache('deep/nested', 'file', {'x': 1})
        assert (tmp_path / 'deep' / 'nested' / 'file.json').exists()

    def test_returns_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr('mlb.data.cache.CACHE_DIR', tmp_path)
        path = write_cache('test', 'data', {'a': 1}, date='2026-04-01')
        assert path == tmp_path / 'test' / '2026-04-01-data.json'


class TestReadCache:
    def test_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr('mlb.data.cache.CACHE_DIR', tmp_path)
        data = {'key': 'value', 'number': 42, 'nested': {'a': [1, 2]}}
        write_cache('test', 'data', data, date='2026-04-01')
        result = read_cache('test', 'data', date='2026-04-01')
        assert result == data

    def test_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr('mlb.data.cache.CACHE_DIR', tmp_path)
        result = read_cache('test', 'nonexistent')
        assert result is None

    def test_expired_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr('mlb.data.cache.CACHE_DIR', tmp_path)
        write_cache('test', 'old', {'data': 1})
        # Backdate the file modification time by 10 days
        path = get_cache_path('test', 'old')
        old_time = time.time() - (86400 * 10)
        os.utime(path, (old_time, old_time))
        result = read_cache('test', 'old', max_age_days=7)
        assert result is None

    def test_fresh_respects_max_age(self, tmp_path, monkeypatch):
        monkeypatch.setattr('mlb.data.cache.CACHE_DIR', tmp_path)
        write_cache('test', 'fresh', {'data': 1})
        result = read_cache('test', 'fresh', max_age_days=7)
        assert result == {'data': 1}

    def test_no_max_age_ignores_staleness(self, tmp_path, monkeypatch):
        monkeypatch.setattr('mlb.data.cache.CACHE_DIR', tmp_path)
        write_cache('test', 'ancient', {'data': 1})
        path = get_cache_path('test', 'ancient')
        old_time = time.time() - (86400 * 365)  # 1 year old
        os.utime(path, (old_time, old_time))
        result = read_cache('test', 'ancient')  # no max_age_days
        assert result == {'data': 1}
