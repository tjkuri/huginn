from nfl.backtest.grader import grade_prediction, grade_all


def _make_prediction(**overrides):
    pred = {
        "player_id": "00-0033873", "player_name": "Patrick Mahomes", "team_abbr": "KC",
        "opponent_abbr": "MIA", "season": 2026, "week": 3,
        "line": 235.5, "scope_used": "last_season",
        "verdict": {"recommendation": "OVER", "confidence": "HIGH", "win_probability": 78.6,
                    "market_fair_probability": 50.0, "edge": 28.6, "z_score": -0.68},
        "adjusted_mean": 273.4, "logged_at": "2026-09-26T15:00:00.000Z",
    }
    pred.update(overrides)
    return pred


class TestGradePredictionRealBets:
    def test_over_hit(self):
        result = grade_prediction(_make_prediction(), actual_yards=250.0)
        assert result["hit"] is True

    def test_over_miss(self):
        result = grade_prediction(_make_prediction(), actual_yards=200.0)
        assert result["hit"] is False

    def test_under_hit(self):
        pred = _make_prediction(verdict={**_make_prediction()["verdict"], "recommendation": "UNDER"})
        result = grade_prediction(pred, actual_yards=200.0)
        assert result["hit"] is True

    def test_under_miss(self):
        pred = _make_prediction(verdict={**_make_prediction()["verdict"], "recommendation": "UNDER"})
        result = grade_prediction(pred, actual_yards=250.0)
        assert result["hit"] is False


class TestGradePredictionNoBet:
    def test_hypothetical_hit_agrees_with_zscore_and_winprob(self):
        no_bet_verdict = {**_make_prediction()["verdict"], "recommendation": "NO_BET", "z_score": -0.3}
        pred = _make_prediction(verdict=no_bet_verdict)
        result = grade_prediction(pred, actual_yards=250.0)  # above line -> OVER actually happened
        assert result["hit"] is None
        # z<0 and win_probability=78.6>50 agree -> implied OVER, and it went OVER
        assert result["hypothetical_hit"] is True

    def test_hypothetical_miss(self):
        no_bet_verdict = {**_make_prediction()["verdict"], "recommendation": "NO_BET", "z_score": -0.3}
        pred = _make_prediction(verdict=no_bet_verdict)
        result = grade_prediction(pred, actual_yards=200.0)  # below line -> UNDER actually happened
        assert result["hypothetical_hit"] is False

    def test_hypothetical_hit_uses_win_probability_not_just_zscore_sign(self):
        # z_score<0 alone would (wrongly) imply OVER, but win_probability=45.0<=50 means the
        # verdict's actual de-vigged choice disagreed with the z-sign and picked UNDER instead
        # (e.g. book's fair_over was higher than the z-implied p_over would suggest). The old
        # z-sign-only formula would have guessed OVER here and scored this case False; the
        # correct (z<0) == (win_probability>50) formula recovers the true chosen side (UNDER).
        no_bet_verdict = {
            **_make_prediction()["verdict"],
            "recommendation": "NO_BET", "z_score": -0.3, "win_probability": 45.0,
        }
        pred = _make_prediction(verdict=no_bet_verdict)
        result = grade_prediction(pred, actual_yards=200.0)  # below line=235.5 -> UNDER actually happened
        assert result["hit"] is None
        assert result["hypothetical_hit"] is True


class TestGradePredictionPush:
    def test_exact_line_match_is_a_push_not_a_win_or_loss(self):
        result = grade_prediction(_make_prediction(), actual_yards=235.5)  # == line
        assert result["hit"] is None
        assert result["hypothetical_hit"] is None
        assert result["abs_error"] is not None  # abs_error compares to adjusted_mean, not line


class TestGradePredictionPendingResult:
    def test_all_none_when_actual_is_none(self):
        result = grade_prediction(_make_prediction(), actual_yards=None)
        assert result["hit"] is None
        assert result["hypothetical_hit"] is None
        assert result["abs_error"] is None
        assert result["actual_yards"] is None


class TestGradePredictionAbsError:
    def test_computed_against_adjusted_mean(self):
        pred = _make_prediction(adjusted_mean=270.0)
        result = grade_prediction(pred, actual_yards=250.0)
        assert result["abs_error"] == 20.0


class TestGradeAll:
    def test_grades_multiple_predictions_fetching_each_season_once(self, monkeypatch):
        calls = []

        def fake_fetch(season):
            calls.append(season)
            return [{"player_id": "00-0033873", "week": "3", "season_type": "REG", "passing_yards": "250"}]

        monkeypatch.setattr("nfl.backtest.grader.fetch_season_rows", fake_fetch)

        preds = [_make_prediction(), _make_prediction(player_id="00-other")]
        graded = grade_all(preds)

        assert len(graded) == 2
        assert calls == [2026]  # fetched once even though two predictions share the season
