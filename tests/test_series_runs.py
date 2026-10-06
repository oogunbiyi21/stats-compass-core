"""detect_run: four or more weekly moves the same way, judged against the store's own history.

Under independent noise any run of four monotone moves has probability
2/5! (about 0.017) whatever its size, so a "joint probability < alpha" gate
would pass every run. Real weekly series drift and are autocorrelated, so runs
are commoner than that. The run is ranked instead among every earlier stretch
of the same length in the store's own seasonally adjusted weekly history.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from stats_compass_core.series import (  # noqa: E402
    Insufficient,
    RunFacts,
    decompose,
    detect_run,
)

DEMO = Path(__file__).parent / "fixtures" / "demo_store_daily.csv"


def _demo(column: str) -> pd.Series:
    return (
        pd.read_csv(DEMO, parse_dates=["date"]).set_index("date")[column].astype(float)
    )


def _null_counts(seed: int, days: int = 742, rate: float = 20.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-09-16", periods=days, freq="D")
    return pd.Series(rng.poisson(rate, days).astype(float), index=idx)


class TestDemoStore:
    def test_the_planted_acquisition_decline_is_a_significant_downward_run(self):
        """New customers per week: 157, 146, 139, 119, 94, 77, 55."""
        run = detect_run(decompose(_demo("new_customers")))
        assert isinstance(run, RunFacts)
        assert run.direction == "down"
        assert run.moves >= 4
        assert run.significant and run.p_value < 0.05
        assert run.change_abs < 0
        assert run.weeks[-1][0] == pd.Timestamp("2026-09-21")

    def test_returning_customers_holding_is_not_a_significant_run(self):
        run = detect_run(decompose(_demo("returning_customers")))
        assert run is None or not run.significant


class TestCalibration:
    def test_patternless_stores_rarely_show_a_significant_run(self):
        significant = 0
        seeds = range(40)
        for seed in seeds:
            run = detect_run(decompose(_null_counts(700 + seed), method="fourier"))
            significant += isinstance(run, RunFacts) and run.significant
        assert significant <= 4, significant

    def test_a_planted_decline_is_found(self):
        series = _null_counts(5).copy()
        weeks = pd.date_range(
            "2026-08-24", periods=5, freq="7D"
        )  # five Mondays, the last week included
        for i, monday in enumerate(weeks):
            series.loc[monday : monday + pd.Timedelta(days=6)] -= 6 * (i + 1)
        run = detect_run(decompose(series, method="fourier"))
        assert isinstance(run, RunFacts)
        assert run.direction == "down" and run.significant


class TestContract:
    def test_no_run_ending_at_the_last_week_is_none(self):
        series = _demo("new_customers").copy()
        series.loc["2026-09-21":"2026-09-27"] += 100  # the last week turns back up
        assert detect_run(decompose(series)) is None

    def test_too_little_history_to_compare_is_insufficient(self):
        result = detect_run(decompose(_demo("new_customers").iloc[-70:]))
        assert isinstance(result, Insufficient)
        assert result.reason == "TOO_SHORT"

    def test_deterministic_and_recorded(self):
        a = detect_run(decompose(_demo("new_customers")), min_moves=4)
        b = detect_run(decompose(_demo("new_customers")), min_moves=4)
        assert (a.p_value, a.moves, a.change_abs) == (b.p_value, b.moves, b.change_abs)
        assert a.params["min_moves"] == 4
        assert a.method == "lagged_seasonal+empirical_runs"
