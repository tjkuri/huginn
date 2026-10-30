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
    Returns: dict[str, dict] — strategy name -> compute_metrics() output. Empty dict if
        backtest_dir doesn't exist or has no subdirectories.
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
        results[entry.name] = compute_metrics(graded)

    return results
