"""forecast: seasonally adjusted daily forecast with intervals measured, not modelled.

The point is the same expectation the verdict uses ahead of the data, corrected
by the median of its past errors at each horizon. The intervals are the
quantiles of those errors, from every past origin where the forecast can be
tried, with the seasonal pattern read as it would have been known at the time.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from stats_compass_core.series import (  # noqa: E402
    ForecastFacts,
    Insufficient,
    decompose,
    forecast,
)

DEMO = Path(__file__).parent / "fixtures" / "demo_store_daily.csv"


def _demo(column: str = "net_revenue") -> pd.Series:
    return (
        pd.read_csv(DEMO, parse_dates=["date"]).set_index("date")[column].astype(float)
    )


def _null(seed: int, days: int = 742) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-09-16", periods=days, freq="D")
    counts = rng.poisson(30, days)
    return pd.Series(
        [rng.lognormal(np.log(2800), 0.55, c).sum() for c in counts], index=idx
    )


class TestShape:
    def test_ninety_daily_points_from_the_day_after_the_history(self):
        facts = forecast(decompose(_demo()))
        assert isinstance(facts, ForecastFacts)
        assert len(facts.points) == 90
        assert facts.points[0].day == date(2026, 9, 28)
        assert facts.points[-1].day == date(2026, 12, 26)
        assert all(p.lower <= p.mean <= p.upper for p in facts.points)

    def test_intervals_never_narrow_as_the_horizon_grows(self):
        facts = forecast(decompose(_demo()))
        below = [p.mean - p.lower for p in facts.points]
        above = [p.upper - p.mean for p in facts.points]
        assert all(b >= a - 1e-9 for a, b in zip(below, below[1:]))
        assert all(b >= a - 1e-9 for a, b in zip(above, above[1:]))

    def test_months_total_the_forecast_days_they_hold(self):
        facts = forecast(decompose(_demo()))
        assert [m.month for m in facts.months] == [
            "2026-09",
            "2026-10",
            "2026-11",
            "2026-12",
        ]
        assert sum(m.n_days for m in facts.months) == 90
        october = next(m for m in facts.months if m.month == "2026-10")
        in_october = [p.mean for p in facts.points if p.day.month == 10]
        assert october.n_days == 31
        assert october.mean == pytest.approx(sum(in_october))
        assert all(m.lower <= m.mean <= m.upper for m in facts.months)

    def test_wide_months_are_flagged_by_the_ratio(self):
        everything = forecast(decompose(_demo()), wide_ratio=0.0)
        nothing = forecast(decompose(_demo()), wide_ratio=100.0)
        assert all(m.wide for m in everything.months)
        assert not any(m.wide for m in nothing.months)


class TestCalibration:
    def test_held_out_days_fall_inside_their_intervals_at_about_the_level(self):
        """Forecast the last 90 days of patternless stores from what came before."""
        inside = total = 0
        for seed in range(20):
            series = _null(900 + seed)
            history, future = series.iloc[:-90], series.iloc[-90:]
            facts = forecast(decompose(history, method="fourier"), level=0.9)
            for point, actual in zip(facts.points, future.to_numpy()):
                inside += point.lower <= actual <= point.upper
                total += 1
        assert 0.84 <= inside / total <= 0.96, inside / total


class TestYoungAndShort:
    def test_a_six_month_store_gets_a_forecast(self):
        facts = forecast(decompose(_demo().iloc[-180:]))
        assert isinstance(facts, ForecastFacts)
        assert facts.n_origins >= 30

    def test_too_few_origins_is_insufficient(self):
        result = forecast(decompose(_demo().iloc[-100:]))
        assert isinstance(result, Insufficient)
        assert result.reason == "TOO_SHORT"


class TestContract:
    def test_deterministic(self):
        a = forecast(decompose(_demo()))
        b = forecast(decompose(_demo()))
        assert [(p.mean, p.lower, p.upper) for p in a.points] == [
            (p.mean, p.lower, p.upper) for p in b.points
        ]

    def test_params_record_every_argument(self):
        facts = forecast(decompose(_demo()), horizon_days=30, level=0.8, wide_ratio=0.4)
        for key in (
            "horizon_days",
            "level",
            "wide_ratio",
            "min_origins",
            "bias_corrected",
            "decomposition_method",
        ):
            assert key in facts.params, key
        assert len(facts.points) == 30
        assert facts.method == "mstl+empirical_horizon_errors"
