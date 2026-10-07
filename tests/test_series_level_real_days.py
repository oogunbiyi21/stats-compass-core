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


class TestTooFewRealDays:
    """Back-to-back promotions with their tails can fill the whole lookback.

    Silently falling back to refilled days made the test fire 42% of the time
    with no warning. Below half a window of real days the level is not
    trusted: the decomposition says so, and verdict and forecast refuse.
    """

    def _filled(self, days_excluded: int):
        y = _demo().loc[:"2026-09-20"]
        last = y.index[-1].date()
        window = (last - timedelta(days=days_excluded - 1), last)
        return y, decompose(y, method="fourier", exclude=[window])

    def test_the_decomposition_records_how_many_real_days_it_had(self):
        _, dec = self._filled(77)  # 84-day lookback leaves 7 real days
        assert dec.params["level_real_days"] == 7
        assert "RECENT_DAYS_FILLED" in [w.code for w in dec.warnings]

    def test_verdict_refuses(self):
        from stats_compass_core.series import Insufficient

        _, dec = self._filled(77)
        period = _demo().loc["2026-09-21":"2026-09-27"]
        result = verdict(dec, period, kind="money")
        assert isinstance(result, Insufficient)
        assert (result.reason, result.needs, result.has) == (
            "RECENT_DAYS_FILLED",
            14,
            7,
        )

    def test_forecast_refuses(self):
        from stats_compass_core.series import Insufficient, forecast

        _, dec = self._filled(77)
        result = forecast(dec)
        assert (
            isinstance(result, Insufficient) and result.reason == "RECENT_DAYS_FILLED"
        )

    def test_half_a_window_is_enough(self):
        from stats_compass_core.series import VerdictFacts

        _, dec = self._filled(63)  # 21 real days left
        assert dec.params["level_real_days"] == 21
        assert "RECENT_DAYS_FILLED" not in [w.code for w in dec.warnings]
        period = _demo().loc["2026-09-21":"2026-09-27"]
        assert isinstance(verdict(dec, period, kind="money"), VerdictFacts)

    def test_a_level_read_across_the_first_year_says_so(self):
        """With the annual period kept at 370 days, the year-back seasonal is
        missing for most of the last 28 days."""
        dec = decompose(_demo().iloc[-370:], method="fourier")
        assert dec.params["periods_used"] == [7, 365]
        assert dec.params["level_real_days"] < 14
        assert "RECENT_DAYS_FILLED" in [w.code for w in dec.warnings]
