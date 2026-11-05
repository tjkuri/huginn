from pathlib import Path

from nfl.data.loader import load_predictions
from nfl.backtest.grader import grade_all
from nfl.backtest.metrics import compute_metrics


def compare_strategies(backtest_dir, weeks_back=None, player=None):
    """
    Grade every strategy's output under backtest_dir (one subdirectory per strategy, each
    containing prediction files in the same shape Yggdrasil's live nflLogger.js writes — see
    scripts/backtest-nfl-strategies.js in the yggdrasil repo), and return each strategy's
    compute_metrics() result keyed by strategy name.

    load_predictions/grade_all/compute_metrics are reused completely unchanged — this function
    is pure orchestration, running the existing single-strategy pipeline once per subdirectory.

    Args:
        backtest_dir: Path to a directory containing one subdirectory per strategy
            (e.g. yggdrasil/cache/nfl-backtest/).
        weeks_back: forwarded to load_predictions.
        player: forwarded to load_predictions.
    Returns: dict[str, dict] — strategy name -> compute_metrics() output, plus an
        "out_of_sample" key: compute_metrics() over only the weeks on or after the strategy's
        out_of_sample_from stamp (with that stamp under "from"), or None if its files carry no
        stamp. Empty dict if backtest_dir doesn't exist or has no subdirectories.
    """
    base = Path(backtest_dir)
    if not base.exists():
        return {}

    results = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        predictions = load_predictions(entry, weeks_back=weeks_back, player=player)
        graded = grade_all(predictions)
        metrics = compute_metrics(graded)
        metrics["out_of_sample"] = _out_of_sample_metrics(graded)
        results[entry.name] = metrics

    return results


def _out_of_sample_metrics(graded):
    """
    Metrics over only the games a strategy could not have been fitted to. A strategy designed
    after looking at past results will look good on those same results, so Yggdrasil stamps
    each strategy's output with the first week it may honestly be judged on
    (out_of_sample_from, defined in services/nflBacktestStrategies.js). If files disagree --
    say the start was moved later -- the latest stamp wins, so earlier weeks never sneak back in.
    """
    stamps = [g["out_of_sample_from"] for g in graded if g.get("out_of_sample_from")]
    if not stamps:
        return None
    start = max(stamps, key=lambda s: (s["season"], s["week"]))
    since = [g for g in graded if (g["season"], g["week"]) >= (start["season"], start["week"])]
    return {"from": start, **compute_metrics(since)}


def compute_common_games_comparison(backtest_dir, weeks_back=None, player=None):
    """
    Among games where EVERY strategy subdirectory under backtest_dir produced a gradeable,
    actionable pick (hit is not None), compute each strategy's hit rate restricted to that
    common set. compare_strategies alone can't answer "which strategy actually wins on the
    same games" -- each strategy's own hit rate there is computed over ITS OWN pick set, which
    can differ substantially from another strategy's pick set (confirmed on real data: two
    strategies' actionable picks overlapped on only 12 of 30+19 total picks). This function
    answers that narrower, more defensible question instead.

    Returns: {"n": <size of the common gradeable+actionable set>,
              "by_strategy": {name: {"n": int, "hits": int, "hit_rate": float|None}}}
    Empty/degenerate inputs return {"n": 0, "by_strategy": {}}.
    """
    base = Path(backtest_dir)
    if not base.exists():
        return {"n": 0, "by_strategy": {}}

    strategy_dirs = [e for e in sorted(base.iterdir()) if e.is_dir()]
    if not strategy_dirs:
        return {"n": 0, "by_strategy": {}}

    graded_by_strategy = {}
    for entry in strategy_dirs:
        predictions = load_predictions(entry, weeks_back=weeks_back, player=player)
        graded = grade_all(predictions)
        graded_by_strategy[entry.name] = {
            (g["season"], g["week"], g["player_id"]): g["hit"]
            for g in graded if g["hit"] is not None
        }

    common_keys = None
    for keyed in graded_by_strategy.values():
        keys = set(keyed.keys())
        common_keys = keys if common_keys is None else (common_keys & keys)
    common_keys = common_keys or set()

    by_strategy = {}
    for name, keyed in graded_by_strategy.items():
        n = len(common_keys)
        hits = sum(1 for k in common_keys if keyed[k])
        by_strategy[name] = {"n": n, "hits": hits, "hit_rate": round(hits / n, 4) if n else None}

    return {"n": len(common_keys), "by_strategy": by_strategy}
