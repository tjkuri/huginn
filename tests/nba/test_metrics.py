import math
from nba.backtest.metrics import compute_metrics, _record_stats, _normal_cdf


# ─── Fixture: graded games with known outcomes ────────────────────────────────

def _graded(v2_result, v2_rec, v2_conf, v2_z, v2_miss,
            proj, dk, actual, v1_result, v1_miss, date, **kw):
    """Shorthand to build a graded game dict."""
    dk_miss = actual - dk
    dk_abs = abs(dk_miss)
    v2_abs_miss = abs(v2_miss) if v2_miss is not None else None
    v1_abs_miss = abs(v1_miss) if v1_miss is not None else None
    g = {
        "date": date,
        "game_id": kw.get("game_id", "1"),
        "home_team": kw.get("home_team", "Team A"),
        "away_team": kw.get("away_team", "Team B"),
        "projected_total": proj,
        "opening_dk_line": dk,
        "actual_total": actual,
        "opening_recommendation": v2_rec,
        "opening_confidence": v2_conf,
        "opening_z_score": v2_z,
        "v2_result": v2_result,
        "v2_miss": v2_miss,
        "v1_result": v1_result,
        "v1_miss": v1_miss,
        "went_to_ot": False,
        "dk_miss": dk_miss,
        "v2_abs_miss": v2_abs_miss,
        "v1_abs_miss": v1_abs_miss,
        "v2_beat_book": (v2_abs_miss < dk_abs) if v2_abs_miss is not None else None,
        "v1_beat_book": (v1_abs_miss < dk_abs) if v1_abs_miss is not None else None,
    }
    g.update(kw)
    return g


GAMES = [
    # Game 1: V2 O WIN, HIGH, z=1.6, gap=|235-220|=15 → 5+
    _graded("WIN",  "O",  "HIGH",   1.6,   -7.0,  235.0, 220.0, 228.0, "WIN",  3.0,  "2026-01-01", game_id="1"),
    # Game 2: V2 U WIN, MEDIUM, z=-0.9, gap=|227-230|=3 → 2-5
    _graded("WIN",  "U",  "MEDIUM", -0.9,   2.0,  227.0, 230.0, 225.0, "WIN",  0.0,  "2026-01-01", game_id="2"),
    # Game 3: V2 O LOSS, LOW, z=0.6, gap=|221-220|=1 → 0-2
    _graded("LOSS", "O",  "LOW",    0.6,   -6.0,  221.0, 220.0, 215.0, "LOSS", -10.0, "2026-01-02", game_id="3"),
    # Game 4: V2 NO_BET, z=0.3 (hyp O, actual>dk → hyp WIN)
    _graded(None,   "NO_BET", "LOW", 0.3,   2.0,  223.0, 220.0, 225.0, "WIN",  3.0,  "2026-01-02", game_id="4"),
    # Game 5: V2 O PUSH, HIGH, z=1.5, gap=|235-220|=15 → 5+
    _graded("PUSH", "O",  "HIGH",   1.5, -15.0,  235.0, 220.0, 220.0, "PUSH", -5.0,  "2026-01-03", game_id="5"),
]


# ─── Tests: _record_stats ─────────────────────────────────────────────────────

class TestRecordStats:
    def test_basic_stats(self):
        games = [{"r": "WIN"}, {"r": "WIN"}, {"r": "LOSS"}, {"r": "PUSH"}]
        stats = _record_stats(games, lambda g: g["r"])
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["pushes"] == 1
        assert stats["total_bets"] == 3
        assert stats["win_rate"] == round(2 / 3, 4)
        assert stats["roi"] == round(((2 * 90.91) - (1 * 100)) / (3 * 100) * 100, 2)

    def test_all_wins(self):
        stats = _record_stats([{"r": "WIN"}], lambda g: g["r"])
        assert stats["win_rate"] == 1.0
        assert stats["roi"] == round(90.91 / 100 * 100, 2)

    def test_all_losses(self):
        stats = _record_stats([{"r": "LOSS"}], lambda g: g["r"])
        assert stats["win_rate"] == 0.0
        assert stats["roi"] == -100.0

    def test_empty_returns_nulls(self):
        stats = _record_stats([], lambda g: g["r"])
        assert stats["total_bets"] == 0
        assert stats["win_rate"] is None
        assert stats["roi"] is None


# ─── Tests: compute_metrics ───────────────────────────────────────────────────

class TestComputeMetrics:
    def test_empty_games(self):
        m = compute_metrics([])
        assert m["total_games"] == 0
        assert m["v2"] is None
        assert m["v1"] is None

    def test_date_range(self):
        m = compute_metrics(GAMES)
        assert m["date_range"] == {"from": "2026-01-01", "to": "2026-01-03"}
        assert m["total_games"] == 5

    def test_v2_overall(self):
        m = compute_metrics(GAMES)
        v2 = m["v2"]
        assert v2["wins"] == 2
        assert v2["losses"] == 1
        assert v2["pushes"] == 1
        assert v2["total_bets"] == 3
        assert v2["win_rate"] == round(2 / 3, 4)
        assert v2["roi"] == round(((2 * 90.91) - (1 * 100)) / (3 * 100) * 100, 2)

    def test_v2_avg_miss(self):
        m = compute_metrics(GAMES)
        expected = round((7 + 2 + 6 + 2 + 15) / 5, 2)
        assert m["v2"]["avg_miss"] == expected

    def test_v2_by_confidence(self):
        m = compute_metrics(GAMES)
        bc = m["v2"]["by_confidence"]
        assert bc["HIGH"]["wins"] == 1
        assert bc["HIGH"]["losses"] == 0
        assert bc["HIGH"]["total_bets"] == 1
        assert bc["HIGH"]["win_rate"] == 1.0
        assert bc["MEDIUM"]["wins"] == 1
        assert bc["MEDIUM"]["total_bets"] == 1
        assert bc["LOW"]["wins"] == 0
        assert bc["LOW"]["losses"] == 1

    def test_v2_no_bet_hypothetical(self):
        m = compute_metrics(GAMES)
        hyp = m["v2"]["by_confidence"]["NO_BET_hypothetical"]
        assert hyp["wins"] == 1
        assert hyp["losses"] == 0
        assert hyp["total_bets"] == 1
        assert hyp["note"] == "NO BET (hypothetical)"

    def test_v2_by_direction(self):
        m = compute_metrics(GAMES)
        bd = m["v2"]["by_direction"]
        assert bd["O"]["wins"] == 1
        assert bd["O"]["losses"] == 1
        assert bd["O"]["total_bets"] == 2
        assert bd["O"]["win_rate"] == 0.5
        assert bd["U"]["wins"] == 1
        assert bd["U"]["total_bets"] == 1

    def test_v2_by_gap_size(self):
        m = compute_metrics(GAMES)
        bg = m["v2"]["by_gap_size"]
        assert bg["0-2"]["losses"] == 1
        assert bg["0-2"]["total_bets"] == 1
        assert bg["2-5"]["wins"] == 1
        assert bg["2-5"]["total_bets"] == 1
        assert bg["5+"]["wins"] == 1
        assert bg["5+"]["pushes"] == 1
        assert bg["5+"]["total_bets"] == 1

    def test_v2_calibration_buckets(self):
        m = compute_metrics(GAMES)
        cal = m["v2"]["calibration"]
        assert "0.5-0.8" in cal
        assert "0.8-1.0" in cal
        assert "1.5-2.0" in cal
        assert "1.0-1.5" not in cal
        assert "2.0+" not in cal
        assert cal["0.5-0.8"]["count"] == 1
        assert cal["0.8-1.0"]["count"] == 1
        assert cal["1.5-2.0"]["count"] == 2

    def test_v1_overall(self):
        m = compute_metrics(GAMES)
        v1 = m["v1"]
        assert v1["wins"] == 3
        assert v1["losses"] == 1
        assert v1["pushes"] == 1
        assert v1["total_bets"] == 4

    def test_v1_avg_miss(self):
        m = compute_metrics(GAMES)
        expected = round((3 + 0 + 10 + 3 + 5) / 5, 2)
        assert m["v1"]["avg_miss"] == expected


# ─── Tests: _normal_cdf ───────────────────────────────────────────────────────

class TestNormalCdf:
    def test_zero(self):
        assert abs(_normal_cdf(0) - 0.5) < 1e-6

    def test_positive(self):
        assert abs(_normal_cdf(1.0) - 0.8413) < 0.001

    def test_negative(self):
        assert abs(_normal_cdf(-1.0) - 0.1587) < 0.001

    def test_large(self):
        assert abs(_normal_cdf(3.0) - 0.9987) < 0.001

    def test_symmetry(self):
        assert abs(_normal_cdf(1.5) + _normal_cdf(-1.5) - 1.0) < 1e-6


class TestBookComparison:
    def test_book_comparison_present(self):
        m = compute_metrics(GAMES)
        assert "book_comparison" in m
        assert m["book_comparison"] is not None

    def test_dk_avg_miss(self):
        m = compute_metrics(GAMES)
        # dk_abs: |228-220|=8, |225-230|=5, |215-220|=5, |225-220|=5, |220-220|=0
        expected = round((8 + 5 + 5 + 5 + 0) / 5, 2)
        assert m["book_comparison"]["dk_avg_miss"] == expected

    def test_v2_beat_rate(self):
        m = compute_metrics(GAMES)
        # v2_abs: 7, 2, 6, 2, 15   dk_abs: 8, 5, 5, 5, 0
        # beats:  T, T, F, T, F  -> 3/5
        assert m["book_comparison"]["v2"]["beat_rate"] == round(3 / 5, 4)

    def test_v2_avg_advantage(self):
        m = compute_metrics(GAMES)
        # advantages (dk_abs - v2_abs): 8-7=1, 5-2=3, 5-6=-1, 5-2=3, 0-15=-15
        expected = round((1 + 3 + (-1) + 3 + (-15)) / 5, 2)
        assert m["book_comparison"]["v2"]["avg_advantage"] == expected

    def test_v1_beat_rate(self):
        m = compute_metrics(GAMES)
        # v1_abs: 3, 0, 10, 3, 5   dk_abs: 8, 5, 5, 5, 0
        # beats:  T, T, F,  T, F  -> 3/5
        assert m["book_comparison"]["v1"]["beat_rate"] == round(3 / 5, 4)

    def test_v1_avg_advantage(self):
        m = compute_metrics(GAMES)
        # advantages (dk_abs - v1_abs): 8-3=5, 5-0=5, 5-10=-5, 5-3=2, 0-5=-5
        expected = round((5 + 5 + (-5) + 2 + (-5)) / 5, 2)
        assert m["book_comparison"]["v1"]["avg_advantage"] == expected

    def test_book_comparison_none_when_empty(self):
        m = compute_metrics([])
        assert m["book_comparison"] is None
