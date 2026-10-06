"""decompose and month_effects: series in, statistics out.

Two failures shaped these tests, both measured on patternless data before any
code was written:

- With two years of history, MSTL's annual component absorbs almost all of the
  daily noise. Its remainder had a tenth of the true noise's spread, so any band
  or test built on it calls ordinary weeks a real change. ``residual_sigma`` is
  held to the truth here.
- A bootstrap of that remainder gave month-effect intervals that excluded zero
  for 94% of months on stores with no seasonality at all. ``month_effects`` is
  held to its nominal rate here.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from stats_compass_core.series import (  # noqa: E402
    Decomposition,
    Insufficient,
    MonthEffects,
    decompose,
    month_effects,
)

DEMO = Path(__file__).parent / "fixtures" / "demo_store_daily.csv"
METHODS = ["mstl", "stl", "fourier"]

# Brief §3.1, measured on the demo data: November 2025, with and without the
# Black Friday weekend and Cyber Monday.
NOVEMBER_WITH_BLACK_FRIDAY = 31.28
NOVEMBER_WITHOUT = 18.94


def _demo(column: str = "net_revenue") -> pd.Series:
    frame = pd.read_csv(DEMO, parse_dates=["date"]).set_index("date")
    return frame[column].astype(float)


def _null(seed: int, days: int = 742) -> pd.Series:
    """A store with nothing to find: Poisson orders, iid order values."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-09-16", periods=days, freq="D")
    counts = rng.poisson(30, days)
    return pd.Series(
        [rng.lognormal(np.log(2800), 0.55, c).sum() for c in counts], index=idx
    )


# =============================================================================
# decompose
# =============================================================================


class TestDecompose:
    @pytest.mark.parametrize("method", METHODS)
    def test_components_add_back_to_the_series(self, method):
        y = _demo()
        dec = decompose(y, method=method)
        assert isinstance(dec, Decomposition)
        if dec.grain == "day":
            observed = y
        else:  # Monday-start weekly sums; the demo is 106 complete weeks
            observed = y.groupby(
                y.index - pd.to_timedelta(y.index.dayofweek, unit="D")
            ).sum()
        rebuilt = dec.trend + dec.seasonal + dec.remainder
        np.testing.assert_allclose(
            rebuilt.to_numpy(), observed.loc[rebuilt.index].to_numpy(), rtol=1e-9
        )

    @pytest.mark.parametrize(
        "method, periods", [("mstl", {7, 365}), ("stl", {52}), ("fourier", {7, 365})]
    )
    def test_one_seasonal_component_per_period(self, method, periods):
        dec = decompose(_demo(), method=method)
        assert set(dec.seasonal_by_period) == periods
        total = sum(dec.seasonal_by_period.values())
        np.testing.assert_allclose(total.to_numpy(), dec.seasonal.to_numpy(), atol=1e-6)

    def test_params_record_every_argument(self):
        dec = decompose(
            _demo(), method="mstl", max_gap_days=5, annual_smoothing_days=21
        )
        for key in (
            "method",
            "periods",
            "max_gap_days",
            "annual_smoothing_days",
            "fourier_terms",
            "trend_ahead",
            "trend_window_days",
            "start",
            "end",
        ):
            assert key in dec.params, key
        assert dec.params["annual_smoothing_days"] == 21
        assert dec.params["max_gap_days"] == 5

    @pytest.mark.parametrize(
        "method, needs", [("mstl", 730), ("stl", 728), ("fourier", 365)]
    )
    def test_too_little_history_is_a_returned_value(self, method, needs):
        result = decompose(_demo().iloc[-300:], method=method)
        assert isinstance(result, Insufficient)
        assert (result.needs, result.has, result.unit, result.reason) == (
            needs,
            300,
            "days",
            "TOO_SHORT",
        )

    def test_short_gaps_are_filled_and_named(self):
        y = _demo().copy()
        y.loc["2025-03-10":"2025-03-12"] = np.nan
        dec = decompose(y)
        assert dec.imputed == [date(2025, 3, 10), date(2025, 3, 11), date(2025, 3, 12)]
        assert [w.code for w in dec.warnings] == ["IMPUTED_DAYS"]
        assert not dec.trend.isna().any()

    def test_a_long_gap_is_insufficient_not_invented(self):
        y = _demo().copy()
        y.loc["2025-03-01":"2025-03-20"] = np.nan
        result = decompose(y, max_gap_days=7)
        assert isinstance(result, Insufficient)
        assert (result.reason, result.needs, result.has) == ("GAP_TOO_LONG", 7, 20)

    def test_leading_and_trailing_nulls_are_trimmed_not_filled(self):
        y = _demo().copy()
        y.iloc[:3] = np.nan
        dec = decompose(y)
        assert dec.params["start"] == "2024-09-19"
        assert dec.imputed == []

    def test_deterministic(self):
        a, b = decompose(_demo()), decompose(_demo())
        assert a.remainder.to_numpy().tobytes() == b.remainder.to_numpy().tobytes()


class TestResidualSigma:
    """The spread the verdict bands are built from must be the noise's, not less."""

    @pytest.mark.parametrize("method", METHODS)
    def test_weekly_sigma_matches_the_noise_on_a_patternless_store(self, method):
        ratios = []
        for seed in range(6):
            y = _null(100 + seed)
            truth = y.rolling(7).sum().iloc[6::7].std()
            ratios.append(decompose(y, method=method).residual_sigma(7) / truth)
        assert 0.75 <= np.mean(ratios) <= 1.25, ratios

    def test_without_annual_smoothing_mstl_understates_the_noise(self):
        """The failure the smoothing exists for, kept visible."""
        y = _null(100)
        truth = y.rolling(7).sum().iloc[6::7].std()
        raw = decompose(y, method="mstl", annual_smoothing_days=0).residual_sigma(7)
        assert raw / truth < 0.5


class TestExpected:
    def test_in_sample_is_trend_plus_seasonal(self):
        dec = decompose(_demo())
        start, end = date(2026, 9, 21), date(2026, 9, 27)
        window = slice(str(start), str(end))
        assert dec.expected_total(start, end) == pytest.approx(
            float((dec.trend[window] + dec.seasonal[window]).sum())
        )

    def test_ahead_repeats_each_seasonal_period_and_holds_the_trend(self):
        dec = decompose(_demo())
        last = dec.trend.index[-1]
        ahead = dec.expected_daily(date(2026, 9, 28), date(2026, 10, 4))
        held = dec.trend.iloc[-28:].mean()
        for day, value in ahead.items():
            seasonal = sum(
                comp.loc[day - pd.Timedelta(days=p)]
                for p, comp in dec.seasonal_by_period.items()
            )
            assert value == pytest.approx(held + seasonal)
        assert ahead.index[0] == last + pd.Timedelta(days=1)

    def test_weekly_grain_apportions_to_days(self):
        dec = decompose(_demo(), method="stl")
        week = dec.expected_daily(date(2026, 9, 21), date(2026, 9, 27))
        assert len(week) == 7
        weekly_value = float((dec.trend + dec.seasonal).loc["2026-09-21"])
        assert week.sum() == pytest.approx(weekly_value)


class TestWeeklyView:
    """Brief §5.3, settled: Monday-start weekly sums, complete weeks only, any method."""

    @pytest.mark.parametrize("method", METHODS)
    def test_demo_is_106_complete_weeks_for_every_method(self, method):
        weekly = decompose(_demo(), method=method).weekly()
        assert len(weekly.trend) == 106
        assert all(day.dayofweek == 0 for day in weekly.trend.index)
        assert weekly.trend.index[0] == pd.Timestamp("2024-09-16")
        rebuilt = weekly.trend + weekly.seasonal + weekly.remainder
        np.testing.assert_allclose(
            rebuilt.to_numpy(), weekly.observed.to_numpy(), rtol=1e-9
        )

    def test_daily_components_sum_to_the_week(self):
        dec = decompose(_demo(), method="mstl")
        weekly = dec.weekly()
        assert weekly.trend.loc["2026-09-21"] == pytest.approx(
            float(dec.trend.loc["2026-09-21":"2026-09-27"].sum())
        )

    def test_partial_weeks_at_either_end_are_dropped(self):
        y = _demo().loc[
            "2024-09-18":"2026-09-24"
        ]  # starts on a Wednesday, ends on a Thursday
        weekly = decompose(y, method="fourier").weekly()
        assert weekly.trend.index[0] == pd.Timestamp("2024-09-23")
        assert weekly.trend.index[-1] == pd.Timestamp("2026-09-14")


# =============================================================================
# month_effects
# =============================================================================


class TestMonthEffects:
    def test_twelve_months(self):
        effects = month_effects(decompose(_demo()))
        assert isinstance(effects, MonthEffects)
        assert [row.month for row in effects.rows] == list(range(1, 13))
        assert all(row.lower <= row.effect_pct <= row.upper for row in effects.rows)
        assert effects.interval_basis == "newey_west_t"

    def test_demo_november_is_recovered(self):
        """T7.2: within 5 pt of the measured figure.

        The effect is the month's mean deviation from trend, the same definition
        the measurement uses, so one-day spikes count: the figure with Black
        Friday applies (brief §5.2).
        """
        effects = month_effects(decompose(_demo(), method="mstl"))
        november = effects.rows[10]
        assert november.effect_pct == pytest.approx(NOVEMBER_WITH_BLACK_FRIDAY, abs=5)
        assert november.lower > 0

    def test_a_flat_series_has_every_interval_through_zero(self):
        y = pd.Series(1000.0, index=pd.date_range("2024-09-16", periods=742, freq="D"))
        effects = month_effects(decompose(y))
        assert all(row.lower <= 0 <= row.upper for row in effects.rows)

    def test_patternless_stores_exclude_zero_at_about_the_nominal_rate(self):
        """The bootstrap this replaced excluded zero for 94% of months."""
        excluded = total = 0
        for seed in range(20):
            effects = month_effects(decompose(_null(500 + seed)), level=0.9)
            excluded += sum(1 for r in effects.rows if r.lower > 0 or r.upper < 0)
            total += 12
        assert excluded / total <= 0.2, f"{excluded}/{total}"

    def test_records_its_basis(self):
        effects = month_effects(decompose(_demo()), level=0.8, hac_lags=5)
        assert effects.params["level"] == 0.8
        assert effects.hac_lags == 5
        assert effects.years_of_history == pytest.approx(742 / 365.25, abs=0.01)
