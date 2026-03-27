from backtest.grader import grade_game, grade_all


def _make_game(**overrides):
    """Build a merged game dict with sensible defaults."""
    game = {
        "date": "2026-01-01",
        "game_id": "1001",
        "home_team": "Team A",
        "away_team": "Team B",
        "projected_total": 228.0,
        "sd_total": 10.0,
        "opening_dk_line": 220.0,
        "opening_z_score": 0.8,
        "opening_confidence": "MEDIUM",
        "opening_recommendation": "O",
        "opening_win_prob": 0.79,
        "opening_ev": 50.0,
        "v1_line": 225.0,
        "actual_total": 230.0,
        "went_to_ot": False,
    }
    game.update(overrides)
    return game


class TestGradeGameV2:
    def test_over_win(self):
        game = _make_game(opening_recommendation="O", opening_dk_line=220.0, actual_total=225.0)
        result = grade_game(game)
        assert result["v2_result"] == "WIN"

    def test_over_loss(self):
        game = _make_game(opening_recommendation="O", opening_dk_line=220.0, actual_total=215.0)
        result = grade_game(game)
        assert result["v2_result"] == "LOSS"

    def test_under_win(self):
        game = _make_game(opening_recommendation="U", opening_dk_line=220.0, actual_total=215.0)
        result = grade_game(game)
        assert result["v2_result"] == "WIN"

    def test_under_loss(self):
        game = _make_game(opening_recommendation="U", opening_dk_line=220.0, actual_total=225.0)
        result = grade_game(game)
        assert result["v2_result"] == "LOSS"

    def test_push(self):
        game = _make_game(opening_recommendation="O", opening_dk_line=220.0, actual_total=220.0)
        result = grade_game(game)
        assert result["v2_result"] == "PUSH"

    def test_no_bet_returns_none(self):
        game = _make_game(opening_recommendation="NO_BET", actual_total=230.0)
        result = grade_game(game)
        assert result["v2_result"] is None

    def test_v2_miss_positive(self):
        game = _make_game(projected_total=220.0, actual_total=230.0)
        result = grade_game(game)
        assert result["v2_miss"] == 10.0

    def test_v2_miss_negative(self):
        game = _make_game(projected_total=230.0, actual_total=220.0)
        result = grade_game(game)
        assert result["v2_miss"] == -10.0

    def test_v2_miss_none_when_no_projected(self):
        game = _make_game(projected_total=None, actual_total=220.0)
        result = grade_game(game)
        assert result["v2_miss"] is None


class TestGradeGameV1:
    def test_v1_over_win(self):
        game = _make_game(v1_line=225.0, opening_dk_line=220.0, actual_total=225.0)
        result = grade_game(game)
        assert result["v1_direction"] == "O"
        assert result["v1_result"] == "WIN"

    def test_v1_over_loss(self):
        game = _make_game(v1_line=225.0, opening_dk_line=220.0, actual_total=215.0)
        result = grade_game(game)
        assert result["v1_direction"] == "O"
        assert result["v1_result"] == "LOSS"

    def test_v1_under_win(self):
        game = _make_game(v1_line=215.0, opening_dk_line=220.0, actual_total=210.0)
        result = grade_game(game)
        assert result["v1_direction"] == "U"
        assert result["v1_result"] == "WIN"

    def test_v1_under_loss(self):
        game = _make_game(v1_line=215.0, opening_dk_line=220.0, actual_total=225.0)
        result = grade_game(game)
        assert result["v1_direction"] == "U"
        assert result["v1_result"] == "LOSS"

    def test_v1_push(self):
        game = _make_game(v1_line=225.0, opening_dk_line=220.0, actual_total=220.0)
        result = grade_game(game)
        assert result["v1_result"] == "PUSH"

    def test_v1_none_when_no_v1_line(self):
        game = _make_game(v1_line=None, actual_total=230.0)
        result = grade_game(game)
        assert result["v1_direction"] is None
        assert result["v1_result"] is None
        assert result["v1_miss"] is None

    def test_v1_skipped_when_zero_gap(self):
        game = _make_game(v1_line=220.0, opening_dk_line=220.0, actual_total=225.0)
        result = grade_game(game)
        assert result["v1_direction"] is None
        assert result["v1_result"] is None

    def test_v1_miss_computed(self):
        game = _make_game(v1_line=225.0, actual_total=230.0)
        result = grade_game(game)
        assert result["v1_miss"] == 5.0


class TestGradeGamePreservesInput:
    def test_does_not_mutate_original(self):
        game = _make_game()
        original_keys = set(game.keys())
        grade_game(game)
        assert set(game.keys()) == original_keys

    def test_preserves_all_original_fields(self):
        game = _make_game()
        result = grade_game(game)
        for key in game:
            assert result[key] == game[key]


class TestGradeAll:
    def test_grades_list_of_games(self):
        games = [
            _make_game(game_id="1", opening_recommendation="O",
                       opening_dk_line=220.0, actual_total=225.0),
            _make_game(game_id="2", opening_recommendation="U",
                       opening_dk_line=220.0, actual_total=215.0),
        ]
        graded = grade_all(games)
        assert len(graded) == 2
        assert graded[0]["v2_result"] == "WIN"
        assert graded[1]["v2_result"] == "WIN"


class TestGradeGameBeatBook:
    def test_dk_miss(self):
        game = _make_game(opening_dk_line=220.0, actual_total=230.0)
        result = grade_game(game)
        assert result["dk_miss"] == 10.0

    def test_dk_miss_negative(self):
        game = _make_game(opening_dk_line=220.0, actual_total=210.0)
        result = grade_game(game)
        assert result["dk_miss"] == -10.0

    def test_v2_abs_miss(self):
        game = _make_game(projected_total=228.0, actual_total=230.0)
        result = grade_game(game)
        assert result["v2_abs_miss"] == 2.0

    def test_v2_abs_miss_none_when_no_projected(self):
        game = _make_game(projected_total=None, actual_total=230.0)
        result = grade_game(game)
        assert result["v2_abs_miss"] is None

    def test_v1_abs_miss(self):
        game = _make_game(v1_line=225.0, actual_total=230.0)
        result = grade_game(game)
        assert result["v1_abs_miss"] == 5.0

    def test_v1_abs_miss_none_when_no_v1_line(self):
        game = _make_game(v1_line=None, actual_total=230.0)
        result = grade_game(game)
        assert result["v1_abs_miss"] is None

    def test_v2_beat_book_true(self):
        # |actual-proj| = 2, |actual-dk| = 10 -> model closer
        game = _make_game(projected_total=228.0, opening_dk_line=220.0, actual_total=230.0)
        result = grade_game(game)
        assert result["v2_beat_book"] is True

    def test_v2_beat_book_false(self):
        # |actual-proj| = 13, |actual-dk| = 5 -> DK closer
        game = _make_game(projected_total=228.0, opening_dk_line=220.0, actual_total=215.0)
        result = grade_game(game)
        assert result["v2_beat_book"] is False

    def test_v2_beat_book_false_on_tie(self):
        # |actual-proj| = 5, |actual-dk| = 5 -> tie is not a beat
        game = _make_game(projected_total=220.0, opening_dk_line=220.0, actual_total=225.0)
        result = grade_game(game)
        assert result["v2_beat_book"] is False

    def test_v2_beat_book_none_when_no_projected(self):
        game = _make_game(projected_total=None, actual_total=230.0)
        result = grade_game(game)
        assert result["v2_beat_book"] is None

    def test_v1_beat_book_true(self):
        # |actual-v1| = 1, |actual-dk| = 10 -> model closer
        game = _make_game(v1_line=229.0, opening_dk_line=220.0, actual_total=230.0)
        result = grade_game(game)
        assert result["v1_beat_book"] is True

    def test_v1_beat_book_none_when_no_v1_line(self):
        game = _make_game(v1_line=None, actual_total=230.0)
        result = grade_game(game)
        assert result["v1_beat_book"] is None
