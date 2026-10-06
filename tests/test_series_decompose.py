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
        "method, days, needs",
        [("mstl", 10, 14), ("stl", 300, 728), ("fourier", 10, 14)],
    )
    def test_too_little_history_is_a_returned_value(self, method, days, needs):
        result = decompose(_demo().iloc[-days:], method=method)
        assert isinstance(result, Insufficient)
        assert (result.needs, result.has, result.unit, result.reason) == (
            needs,
            days,
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

    @pytest.mark.parametrize("method", METHODS)
    def test_deterministic(self, method):
        a, b = decompose(_demo(), method=method), decompose(_demo(), method=method)
        assert a.remainder.to_numpy().tobytes() == b.remainder.to_numpy().tobytes()
        assert a.trend.to_numpy().tobytes() == b.trend.to_numpy().tobytes()

    def test_trimmed_days_are_named_not_dropped_silently(self):
        y = _demo().copy()
        y.iloc[:2] = np.nan
        y.iloc[-1] = np.nan  # a partial last day sent as null
        dec = decompose(y)
        assert dec.trimmed == [date(2024, 9, 16), date(2024, 9, 17), date(2026, 9, 27)]
        assert "TRIMMED_DAYS" in [w.code for w in dec.warnings]

    def test_components_are_matched_to_their_period_by_name(self):
        dec = decompose(_demo(), method="mstl")
        weekly = dec.seasonal_by_period[7].to_numpy()[: 7 * 50].reshape(50, 7)
        # A weekly component repeats every 7 days; the annual one does not.
        assert (
            np.abs(weekly.sum(axis=1)).max() < 0.05 * np.abs(weekly).sum(axis=1).max()
        )

    @pytest.mark.parametrize("method", ["mstl", "stl"])
    def test_robustness_is_one_setting_for_both_stl_methods(self, method):
        plain = decompose(_demo(), method=method)
        robust = decompose(_demo(), method=method, robust=True)
        assert (plain.params["robust"], robust.params["robust"]) == (False, True)
        assert not np.allclose(plain.trend.to_numpy(), robust.trend.to_numpy())

    def test_records_the_statsmodels_version(self):
        import statsmodels

        assert (
            decompose(_demo()).params["statsmodels_version"] == statsmodels.__version__
        )


class TestExclude:
    """For promotion ITS: leave a window out of the fit without calling it a gap."""

    WINDOW = (date(2026, 5, 14), date(2026, 5, 24))

    def test_a_long_window_is_excluded_not_insufficient(self):
        dec = decompose(_demo(), exclude=[self.WINDOW])
        assert isinstance(dec, Decomposition)
        assert len(dec.excluded) == 11 and dec.imputed == []
        assert dec.remainder.loc["2026-05-14":"2026-05-24"].isna().all()
        assert dec.params["exclude"] == [["2026-05-14", "2026-05-24"]]

    def test_excluded_values_do_not_shape_the_fit(self):
        clean = _demo()
        planted = clean.copy()
        planted.loc["2026-05-14":"2026-05-24"] *= 3
        a = decompose(clean, exclude=[self.WINDOW]).expected_total(*self.WINDOW)
        b = decompose(planted, exclude=[self.WINDOW]).expected_total(*self.WINDOW)
        assert a == pytest.approx(b, rel=1e-12)

    def test_excluded_days_are_refilled_from_the_fit_not_a_straight_line(self):
        """A line between the two days either side of a long window is as noisy
        as those two days; on the demo it moved a promotion lift by 14 pt."""
        start, end = date(2026, 5, 14), date(2026, 6, 21)
        dec = decompose(_demo(), exclude=[(start, end)])
        inside = slice(str(start), str(end))
        refit_gap = (
            (dec.observed[inside] - (dec.trend + dec.seasonal)[inside]).abs().mean()
        )
        noise = dec.remainder.dropna().abs().mean()
        assert refit_gap < 0.25 * noise
        assert dec.params["exclude_iterations"] == 3

    def test_excluded_days_are_left_out_of_the_noise(self):
        dec = decompose(_demo(), exclude=[self.WINDOW])
        assert not np.isnan(dec.residual_sigma(7))


class TestResidualSigma:
    """The spread the verdict bands are built from must be the noise's, not less."""

    def test_robust_scale_ignores_spikes_but_matches_on_plain_noise(self):
        demo = decompose(_demo())
        assert demo.residual_sigma(7, robust=True) < demo.residual_sigma(7)
        noise = decompose(_null(100))
        assert noise.residual_sigma(7, robust=True) == pytest.approx(
            noise.residual_sigma(7), rel=0.15
        )

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

    def test_ahead_repeats_each_seasonal_period_and_holds_the_level(self):
        """Ahead, the level is the mean of the seasonally adjusted values,
        adjusted by the seasonal read one period earlier: the estimator that
        expectation_errors measures in the past.

        Not the trend: at the end of the data a smoother sees only one side,
        and its end value carried more error than the 28-day mean of the data.
        """
        dec = decompose(_demo())
        last = dec.trend.index[-1]
        ahead = dec.expected_daily(date(2026, 9, 28), date(2026, 10, 4))
        lagged = sum(c.shift(p) for p, c in dec.seasonal_by_period.items())
        level = (dec.daily - lagged).iloc[-28:].mean()
        for day, value in ahead.items():
            seasonal = sum(
                comp.loc[day - pd.Timedelta(days=p)]
                for p, comp in dec.seasonal_by_period.items()
            )
            assert value == pytest.approx(level + seasonal)
        assert ahead.index[0] == last + pd.Timedelta(days=1)
        assert dec.params["trend_ahead"] == "level"

    def test_weekly_grain_apportions_to_days(self):
        dec = decompose(_demo(), method="stl")
        week = dec.expected_daily(date(2026, 9, 21), date(2026, 9, 27))
        assert len(week) == 7
        weekly_value = float((dec.trend + dec.seasonal).loc["2026-09-21"])
        assert week.sum() == pytest.approx(weekly_value)


class TestYoungStores:
    """Under two years, fit the weekly pattern and say the annual one was dropped.

    Requiring the annual period made every headline of a six-month store
    insufficient, though its floors are 84 days. Dropping the annual period
    is a statistical choice (its variation lands in the remainder and widens
    the bands), so it is made here and recorded, not left to the caller.
    """

    @pytest.mark.parametrize("method", ["mstl", "fourier"])
    def test_a_young_store_gets_its_weekly_pattern(self, method):
        dec = decompose(_demo().iloc[-300:], method=method)
        assert isinstance(dec, Decomposition)
        assert dec.params["periods_used"] == [7]
        assert set(dec.seasonal_by_period) == {7}
        warning = next(w for w in dec.warnings if w.code == "ANNUAL_DROPPED")
        assert "365" in warning.message

    def test_two_years_keep_the_annual_period(self):
        dec = decompose(_demo())
        assert dec.params["periods_used"] == [7, 365]
        assert "ANNUAL_DROPPED" not in [w.code for w in dec.warnings]

    def test_month_effects_need_the_annual_period(self):
        result = month_effects(decompose(_demo().iloc[-300:]))
        assert isinstance(result, Insufficient)
        assert (result.reason, result.needs, result.has) == ("NO_ANNUAL", 730, 300)


class TestExpectationErrors:
    """The error of the expectation, measured where it can be: in the past.

    At each past position the same estimator the period is judged by: the
    seasonally adjusted level over the 28 days before, plus the seasonal
    pattern read one period earlier, as a forecast must read it.
    """

    def test_one_row_per_position_with_a_full_year_and_baseline_behind_it(self):
        dec = decompose(_demo())
        errors = dec.expectation_errors(7)
        assert list(errors.columns) == ["error", "expected"]
        earliest = dec.daily.index[0] + pd.Timedelta(days=365 + 28)
        assert errors.index.min() == earliest
        assert errors.index.max() == dec.daily.index[-1] - pd.Timedelta(days=6)
        assert not errors.isna().any().any()

    def test_each_error_is_the_estimator_applied_at_that_position(self):
        dec = decompose(_demo())
        errors = dec.expectation_errors(7)
        lagged = sum(c.shift(p) for p, c in dec.seasonal_by_period.items())
        adjusted = dec.daily - lagged
        at = pd.Timestamp("2026-03-02")
        level = adjusted.loc[
            at - pd.Timedelta(days=28) : at - pd.Timedelta(days=1)
        ].mean()
        week = slice(at, at + pd.Timedelta(days=6))
        expected = 7 * level + lagged.loc[week].sum()
        assert errors.loc[at, "expected"] == pytest.approx(expected)
        assert errors.loc[at, "error"] == pytest.approx(
            dec.daily.loc[week].sum() - expected
        )

    @pytest.mark.parametrize("method", METHODS)
    def test_every_method_provides_them(self, method):
        errors = decompose(_demo(), method=method).expectation_errors(11)
        assert len(errors) > 300


class TestOutOfSampleReference:
    """A past window's own data leaks into the seasonal it is judged against.

    With two cycles, the seasonal one period before a past position was fitted
    with that position's values in it; the period really being judged was not.
    For the Fourier fit the leak is removed exactly, by deleting each window's
    rows from the least-squares solution.
    """

    def test_fourier_errors_match_a_refit_without_the_window(self):
        y = _demo()
        dec = decompose(y, method="fourier")
        errors = dec.expectation_errors(7)
        assert errors.attrs["reference"] == "out_of_sample_exact"
        at = pd.Timestamp("2026-03-02")
        week = slice(at, at + pd.Timedelta(days=6))
        # Excluding the week and refilling it from the fit until it settles is
        # least squares without those rows: what the deletion computes directly.
        refit = decompose(
            y,
            method="fourier",
            exclude=[(at.date(), (at + pd.Timedelta(days=6)).date())],
            exclude_iterations=40,
        )
        lagged = sum(c.shift(p) for p, c in refit.seasonal_by_period.items())
        adjusted = y - lagged
        level = adjusted.loc[
            at - pd.Timedelta(days=28) : at - pd.Timedelta(days=1)
        ].mean()
        expected = 7 * level + lagged.loc[week].sum()
        assert errors.loc[at, "expected"] == pytest.approx(expected, rel=1e-6)

    def test_out_of_sample_errors_are_wider_than_in_sample(self):
        dec = decompose(_null(100), method="fourier")
        oos = dec.expectation_errors(7)["error"].std()
        ins = dec.expectation_errors(7, out_of_sample=False)["error"].std()
        assert oos > ins

    def test_other_methods_say_their_reference_is_in_sample(self):
        errors = decompose(_demo(), method="mstl").expectation_errors(7)
        assert errors.attrs["reference"] == "in_sample"


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
        """T7.2: within 5 pt of the measured figure. PROVISIONAL until §5.2 is decided.

        The effect is the month's mean deviation from trend, the same definition
        the measurement uses, so one-day spikes count and the figure with Black
        Friday is the one compared. The seasonal component and the expectation
        smooth Black Friday into the remainder instead, which is the other
        answer to §5.2. Which figure T7.2 is measured against is the founder's
        call (brief §5.2, T7.1).
        """
        effects = month_effects(decompose(_demo(), method="mstl"))
        november = effects.rows[10]
        assert november.effect_pct == pytest.approx(NOVEMBER_WITH_BLACK_FRIDAY, abs=5)
        assert november.lower > 0

    def test_a_flat_series_has_every_interval_through_zero(self):
        y = pd.Series(1000.0, index=pd.date_range("2024-09-16", periods=742, freq="D"))
        effects = month_effects(decompose(y))
        assert all(row.lower <= 0 <= row.upper for row in effects.rows)

    @pytest.mark.parametrize("method", METHODS)
    def test_patternless_stores_exclude_zero_at_about_the_nominal_rate(self, method):
        """The bootstrap this replaced excluded zero for 94% of months.

        Two-sided: an interval that never excludes zero is as useless as one
        that always does.
        """
        excluded = total = 0
        for seed in range(40):
            effects = month_effects(
                decompose(_null(500 + seed), method=method), level=0.9
            )
            excluded += sum(1 for r in effects.rows if r.lower > 0 or r.upper < 0)
            total += 12
        assert 0.05 <= excluded / total <= 0.15, f"{method}: {excluded}/{total}"

    def test_a_level_at_or_below_zero_is_insufficient(self):
        y = _demo() - 400_000.0  # net revenue below zero all year
        result = month_effects(decompose(y, method="fourier"))
        assert isinstance(result, Insufficient)
        assert (result.reason, result.needs, result.has) == (
            "NONPOSITIVE_LEVEL",
            None,
            None,
        )

    def test_a_month_without_enough_days_is_insufficient(self):
        dec = decompose(_demo(), method="fourier")
        short = Decomposition(
            method=dec.method,
            grain="day",
            params=dec.params,
            observed=dec.observed.loc[:"2025-08-31"].iloc[-200:],
            trend=dec.trend.loc[:"2025-08-31"].iloc[-200:],
            seasonal=dec.seasonal.loc[:"2025-08-31"].iloc[-200:],
            remainder=dec.remainder.loc[:"2025-08-31"].iloc[-200:],
            seasonal_by_period={
                365: dec.seasonal_by_period[365].loc[:"2025-08-31"].iloc[-200:]
            },
            imputed=[],
            warnings=[],
        )
        result = month_effects(short)
        assert isinstance(result, Insufficient)
        assert result.reason == "MONTH_TOO_SHORT"

    def test_the_days_per_month_floor_is_an_argument(self):
        result = month_effects(decompose(_demo()), min_days_per_month=200)
        assert isinstance(result, Insufficient) and result.reason == "MONTH_TOO_SHORT"

    def test_records_its_basis(self):
        effects = month_effects(decompose(_demo()), level=0.8, hac_lags=5)
        assert effects.params["level"] == 0.8
        assert effects.hac_lags == 5
        assert effects.years_of_history == pytest.approx(742 / 365.25, abs=0.01)
