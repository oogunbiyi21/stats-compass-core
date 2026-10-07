"""The level ahead is read from real days, never from refilled ones.

When a recent promotion and its tail are excluded, the last days before the
period are refilled from the fit. Reading the level from them reads the fit
back to itself: on stores with an inert promotion ending in the last four
weeks, the verdict's test alone fired 12-18% of the time (mstl, stl). The level
now walks back past excluded and imputed days, as the promotion baseline does.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from stats_compass_core.series import decompose, verdict  # noqa: E402

DEMO = Path(__file__).parent / "fixtures" / "demo_store_daily.csv"


def _demo() -> pd.Series:
    return (
        pd.read_csv(DEMO, parse_dates=["date"])
        .set_index("date")["net_revenue"]
        .astype(float)
    )


def _null(seed: int, days: int = 742) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-09-16", periods=days, freq="D")
    counts = rng.poisson(30, days)
    return pd.Series(
        [rng.lognormal(np.log(2800), 0.55, c).sum() for c in counts], index=idx
    )


def test_the_level_ahead_skips_excluded_days():
    y = _demo().loc[:"2026-09-20"]
    excluded = (date(2026, 8, 31), date(2026, 9, 20))  # the last 21 days
    dec = decompose(y, method="fourier", exclude=[excluded])
    lagged = sum(c.shift(p) for p, c in dec.seasonal_by_period.items())
    real = (dec.daily - lagged).loc[: pd.Timestamp("2026-08-30")].iloc[-28:]
    seasonal_first_day = sum(
        c.loc[pd.Timestamp("2026-09-21") - pd.Timedelta(days=p)]
        for p, c in dec.seasonal_by_period.items()
    )
    ahead = dec.expected_daily(date(2026, 9, 21), date(2026, 9, 21)).iloc[0]
    assert ahead == pytest.approx(real.mean() + seasonal_first_day)
    assert dec.params["level_lookback_days"] == 84


@pytest.mark.parametrize("method", ["mstl", "fourier"])
def test_a_recent_excluded_promotion_does_not_make_the_test_fire(method):
    """80 stores at alpha 0.05: a calibrated test lands on at most 10."""
    fired = 0
    for seed in range(80):
        series = _null(9000 + seed)
        end = series.index[-1]
        start = end - pd.Timedelta(days=6)
        history = series.loc[: start - pd.Timedelta(days=1)]
        last = history.index[-1].date()
        promo = (
            last - timedelta(days=27),
            last - timedelta(days=21),
        )  # tail runs to the last day
        dec = decompose(
            history, method=method, exclude=[(promo[0], promo[1] + timedelta(days=21))]
        )
        fired += verdict(dec, series.loc[start:end], kind="money").p_value < 0.05
    assert fired <= 10, f"{method}: {fired}/80"
