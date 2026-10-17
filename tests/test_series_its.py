"""its_lift: did a promotion move a series, against what would have happened?

The counterfactual is the seasonally adjusted level over the days before the
window, plus the seasonal pattern from a fit that never saw the window or the
pull-forward tail after it. Its error is measured by applying the same
estimator at every past position, with the seasonal read one period earlier
as a forecast reads it. An earlier version read it in-sample and reported
p-values two to three times too small.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from stats_compass_core.series import (  # noqa: E402
    CounterfactualDay,
    Insufficient,
    LiftFacts,
    its_lift,
)

DEMO = Path(__file__).parent / "fixtures" / "demo_store_daily.csv"
SUMMER30 = (date(2026, 5, 14), date(2026, 5, 24))
PRIOR = (date(2025, 5, 15), date(2025, 5, 25))  # 364 days earlier, same weekdays


def _demo(column: str) -> pd.Series:
    frame = pd.read_csv(DEMO, parse_dates=["date"]).set_index("date")
    return frame[column].astype(float)


def _null(seed: int, days: int = 742) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-09-16", periods=days, freq="D")
    counts = rng.poisson(30, days)
    return pd.Series(
        [rng.lognormal(np.log(2800), 0.55, c).sum() for c in counts], index=idx
    )


# =============================================================================
# The demo store
# =============================================================================


class TestDemoPromotion:
    """SUMMER30. Measured on the data: revenue per day +24.7%, contribution
    margin per day -18.7%, against the 28 days before (unadjusted)."""

    def test_revenue_lift_is_positive_with_an_honest_interval(self):
        facts = its_lift(
            _demo("net_revenue"), SUMMER30, kind="money", prior_window=PRIOR
        )
        assert isinstance(facts, LiftFacts)
        assert 10 < facts.estimate_pct < 35
        lo, hi = facts.interval_pct
        assert lo < facts.estimate_pct < hi
        assert hi - lo > 20  # an 11-day window carries about 12 pt of noise

    def test_contribution_margin_lift_is_negative(self):
        facts = its_lift(
            _demo("contribution_margin"), SUMMER30, kind="money", prior_window=PRIOR
        )
        assert -35 < facts.estimate_pct < -10

    def test_new_customers_lift_is_a_real_change(self):
        """+67%, well clear of an 11-day window's noise, even with the noise band."""
        facts = its_lift(
            _demo("new_customers"), SUMMER30, kind="count", prior_window=PRIOR
        )
        assert facts.estimate_pct > 40
        assert facts.verdict == "real change"
        assert facts.interval_pct[0] > 0

    def test_money_lifts_are_not_real_changes_at_this_size(self):
        """PROVISIONAL. T7.7's AC expects real change for both money lifts; at
        about 20% over 11 days they are under 2σ (p about 0.1). Commerce is
        re-planting the promotion; this test records today's honest answer."""
        for column in ("net_revenue", "contribution_margin"):
            facts = its_lift(_demo(column), SUMMER30, kind="money", prior_window=PRIOR)
            assert facts.verdict != "real change", column
            assert facts.p_value > 0.05, column

    def test_a_larger_effect_is_a_real_change(self):
        series = _demo("net_revenue").copy()
        series.loc["2026-05-14":"2026-05-24"] *= 1.6
        facts = its_lift(
            series, SUMMER30, kind="money", prior_window=PRIOR, noise_band="none"
        )
        assert facts.verdict == "real change"
        assert facts.estimate_pct > 40


# =============================================================================
# What the counterfactual may and may not see
# =============================================================================


class TestCounterfactual:
    def test_the_windows_own_values_never_move_it(self):
        clean = _demo("net_revenue")
        planted = clean.copy()
        planted.loc["2026-05-14":"2026-05-24"] *= 3
        a = its_lift(clean, SUMMER30, kind="money")
        b = its_lift(planted, SUMMER30, kind="money")
        assert a.counterfactual_total == pytest.approx(b.counterfactual_total, rel=1e-9)

    def test_the_pull_forward_tail_is_out_of_the_fit_and_the_reference(self):
        clean = _demo("net_revenue")
        dipped = clean.copy()
        dipped.loc["2026-05-25":"2026-06-14"] *= 0.5  # 21 days after the window
        a = its_lift(clean, SUMMER30, kind="money")
        b = its_lift(dipped, SUMMER30, kind="money")
        assert a.counterfactual_total == pytest.approx(b.counterfactual_total, rel=1e-9)
        assert a.interval_pct == pytest.approx(b.interval_pct, rel=1e-9)
        assert a.n_reference == b.n_reference

    def test_other_promotions_are_kept_out_too(self):
        series = _demo("net_revenue")
        other = (date(2025, 11, 3), date(2025, 11, 9))
        plain = its_lift(series, SUMMER30, kind="money")
        both = its_lift(series, SUMMER30, kind="money", other_windows=[other])
        assert both.n_reference < plain.n_reference
        assert ["2025-11-03", "2025-11-30"] in both.params["exclude"]


# =============================================================================
# Calibration on stores with nothing to find
# =============================================================================


class TestCalibration:
    """No effect at all: the 90% interval should miss zero about 10% of the time.

    A trip-wire with fixed seeds, bounded by Binomial(n, 0.1)'s 99% region; the
    harness's runs over many stores are the calibration evidence. The first
    version read the window's seasonal in-sample while its reference read it a
    period earlier, and missed zero for 20% of inert promotions.
    """

    @pytest.mark.parametrize("method, n, bound", [("fourier", 60, 12), ("mstl", 20, 6)])
    def test_inert_promotions_on_patternless_stores(self, method, n, bound):
        excluded = real = 0
        for seed in range(n):
            facts = its_lift(_null(600 + seed), SUMMER30, kind="money", method=method)
            lo, hi = facts.interval_pct
            excluded += lo > 0 or hi < 0
            real += facts.verdict == "real change"
        assert excluded <= bound, f"{method}: {excluded}/{n}"
        assert real <= bound // 2, f"{method}: {real}/{n}"


# =============================================================================
# Units, insufficiency and the contract
# =============================================================================


class TestUnitsAndContract:
    def test_absolute_and_percent_agree(self):
        facts = its_lift(_demo("net_revenue"), SUMMER30, kind="money")
        assert facts.estimate_abs == pytest.approx(
            facts.observed_total - facts.counterfactual_total
        )
        assert facts.estimate_pct == pytest.approx(
            100 * facts.estimate_abs / facts.counterfactual_total
        )

    def test_percent_is_none_when_the_counterfactual_is_not_positive(self):
        series = _demo("net_revenue") - 300_000.0
        facts = its_lift(series, SUMMER30, kind="money", method="fourier")
        assert facts.counterfactual_total <= 0
        assert facts.estimate_pct is None and facts.interval_pct is None

    def test_the_noise_band_is_off_by_default(self):
        facts = its_lift(
            _demo("net_revenue"), SUMMER30, kind="money", prior_window=PRIOR
        )
        assert facts.params["noise_band"] == "none" and facts.noise_range_pct is None

    def test_a_prior_lift_needs_a_seasonal_period_before_it(self):
        """With two years, May 2025 has no May before it to adjust by."""
        facts = its_lift(
            _demo("net_revenue"),
            SUMMER30,
            kind="money",
            prior_window=PRIOR,
            noise_band="prior_window",
        )
        assert facts.prior_lift_pct is None and facts.noise_range_pct is None
        assert facts.params["prior_unavailable"] == "NO_SEASONAL_BEFORE_PRIOR"

    def test_with_three_years_the_prior_lift_centres_the_noise_band(self):
        series = _null(42, days=1200)
        window = (date(2027, 5, 13), date(2027, 5, 23))
        prior = (date(2026, 5, 14), date(2026, 5, 24))
        facts = its_lift(
            series,
            window,
            kind="money",
            method="fourier",
            prior_window=prior,
            noise_band="prior_window",
        )
        assert facts.prior_lift_pct is not None
        lo, hi = facts.noise_range_pct
        assert (lo + hi) / 2 == pytest.approx(facts.prior_lift_pct)

    def test_the_interval_excludes_zero_exactly_when_the_test_is_significant(self):
        """The interval is the test inverted, so the report cannot contradict itself."""
        for column, kind in (("net_revenue", "money"), ("new_customers", "count")):
            facts = its_lift(_demo(column), SUMMER30, kind=kind, level=0.9)
            lo, hi = facts.interval_pct
            excludes = lo > 0 or hi < 0
            assert excludes == (facts.p_value < 0.1), column

    def test_interval_and_band_agree_with_the_test_on_every_store(self):
        """Both are read from the same ranked error the p-value counts, so they
        can never disagree with it; an interpolated quantile did on 5 of 300."""
        for seed in range(30):
            facts = its_lift(
                _null(800 + seed), SUMMER30, kind="money", method="fourier"
            )
            lo, hi = facts.interval_pct
            assert (lo > 0 or hi < 0) == (facts.p_value < 0.1), seed
            band_lo, band_hi = facts.expected_range_pct
            in_band = band_lo <= facts.estimate_pct <= band_hi
            assert in_band == (facts.p_value >= 1 - facts.params["band_level"]), seed

    def test_floors_are_arguments(self):
        facts = its_lift(
            _demo("net_revenue"),
            SUMMER30,
            kind="money",
            min_baseline_share=0.5,
            min_reference=30,
        )
        assert facts.params["min_baseline_share"] == 0.5
        assert facts.params["min_reference"] == 30

    def test_band_and_interval_levels_are_recorded(self):
        facts = its_lift(_demo("net_revenue"), SUMMER30, kind="money")
        assert facts.params["band_basis"] == "abs_error_quantile"
        assert facts.params["interval_basis"] == "abs_error_quantile"
        assert facts.params["band_level"] == pytest.approx(0.9545, abs=1e-4)
        assert facts.params["interval_level"] == 0.9

    def test_deterministic(self):
        a = its_lift(_demo("net_revenue"), SUMMER30, kind="money", prior_window=PRIOR)
        b = its_lift(_demo("net_revenue"), SUMMER30, kind="money", prior_window=PRIOR)
        assert (a.estimate_pct, a.interval_pct, a.p_value) == (
            b.estimate_pct,
            b.interval_pct,
            b.p_value,
        )

    def test_params_record_every_argument(self):
        facts = its_lift(
            _demo("net_revenue"), SUMMER30, kind="money", baseline_days=21, post_days=14
        )
        for key in (
            "kind",
            "window",
            "baseline_days",
            "post_days",
            "prior_window",
            "other_windows",
            "noise_band",
            "level",
            "seasonal_band_z",
            "noise_band_z",
            "real_change_alpha",
            "decomposition",
            "baseline_lookback_days",
            "band_level",
            "interval_level",
        ):
            assert key in facts.params, key
        assert facts.method == "mstl+its_empirical_windows"

    def test_too_little_history_before_the_window(self):
        result = its_lift(
            _demo("net_revenue"), (date(2024, 9, 30), date(2024, 10, 6)), kind="money"
        )
        assert isinstance(result, Insufficient)
        assert result.reason == "BASELINE_TOO_SHORT"
        assert result.needs == 23  # 80% of 28, what is actually required

    def test_the_baseline_reaches_back_past_another_promotion(self):
        """Monthly promotions would otherwise leave every lift without a baseline."""
        earlier = (date(2026, 4, 1), date(2026, 4, 7))  # its tail runs to 28 April
        facts = its_lift(
            _demo("net_revenue"), SUMMER30, kind="money", other_windows=[earlier]
        )
        assert facts.params["baseline_real_days"] == 28
        assert facts.params["baseline_first_day"] < "2026-04-01"

    def test_too_few_days_with_data_in_the_window(self):
        series = _demo("net_revenue").copy()
        series.loc["2026-05-14":"2026-05-22"] = np.nan
        result = its_lift(series, SUMMER30, kind="money")
        assert isinstance(result, Insufficient)
        assert result.reason == "TOO_FEW_POINTS"


# =============================================================================
# The series a chart draws: observed against the counterfactual, day by day
# =============================================================================


def _gap(facts: LiftFacts) -> tuple[float, float]:
    """Observed minus expected, and expected, summed over the window's days with data."""
    days = [d for d in facts.series if d.in_window and d.actual is not None]
    return (
        sum(d.actual - d.expected for d in days),
        sum(d.expected for d in days),
    )


class TestCounterfactualSeries:
    def test_not_computed_unless_asked_for(self):
        facts = its_lift(_demo("net_revenue"), SUMMER30, kind="money")
        assert facts.series is None
        assert "series_band_half_width" not in facts.params

    @pytest.mark.parametrize("method", ["fourier", "mstl"])
    def test_the_gap_between_the_lines_is_the_lift(self, method):
        facts = its_lift(
            _demo("net_revenue"), SUMMER30, kind="money", method=method,
            series_days_before=28, series_days_after=21,
        )
        gap, expected = _gap(facts)
        assert gap == pytest.approx(facts.estimate_abs, rel=1e-9)
        assert expected == pytest.approx(facts.counterfactual_total, rel=1e-9)

    def test_each_day_from_28_before_to_21_after(self):
        facts = its_lift(
            _demo("net_revenue"), SUMMER30, kind="money",
            series_days_before=28, series_days_after=21,
        )
        dates = [d.date for d in facts.series]
        assert dates[0] == date(2026, 4, 16) and dates[-1] == date(2026, 6, 14)
        assert len(dates) == 28 + 11 + 21  # every day, none skipped
        assert [d.date for d in facts.series if d.in_window] == list(
            pd.date_range(*SUMMER30).date
        )
        assert all(isinstance(d, CounterfactualDay) for d in facts.series)

    def test_actual_is_the_series_itself(self):
        values = _demo("net_revenue")
        facts = its_lift(values, SUMMER30, kind="money", series_days_before=7)
        for day in facts.series:
            assert day.actual == values.loc[pd.Timestamp(day.date)]

    def test_the_band_is_one_days_range_at_the_lifts_level(self):
        facts = its_lift(
            _demo("net_revenue"), SUMMER30, kind="money", method="fourier",
            series_days_before=28, series_days_after=21,
        )
        half = facts.params["series_band_half_width"]
        assert facts.params["series_band_level"] == facts.params["interval_level"] == 0.9
        assert facts.params["series_band_reference"] == "out_of_sample_exact"
        for day in facts.series:
            assert day.low == pytest.approx(day.expected - half)
            assert day.high == pytest.approx(day.expected + half)
        # A day's range, not the window's: narrower than the lift's interval.
        lo, hi = facts.interval_abs
        assert 0 < 2 * half < hi - lo
        # About nine days in ten of the lead-in fall inside it.
        before = [d for d in facts.series if d.date < SUMMER30[0]]
        inside = sum(d.low <= d.actual <= d.high for d in before)
        assert inside >= 0.75 * len(before)

    def test_it_ends_at_the_last_day_of_data(self):
        values = _demo("net_revenue").loc[:"2026-06-01"]
        facts = its_lift(values, SUMMER30, kind="money", series_days_after=21)
        assert facts.series[-1].date == date(2026, 6, 1)
        assert facts.series[0].date == SUMMER30[0]  # nothing before was asked for

    def test_a_day_with_no_data_has_no_actual_and_stays_out_of_the_gap(self):
        values = _demo("net_revenue").copy()
        values.loc["2026-05-16":"2026-05-17"] = np.nan
        facts = its_lift(values, SUMMER30, kind="money", series_days_before=3)
        missing = [d for d in facts.series if d.actual is None]
        assert [d.date for d in missing] == [date(2026, 5, 16), date(2026, 5, 17)]
        assert all(np.isfinite(d.expected) for d in missing)
        gap, expected = _gap(facts)
        assert gap == pytest.approx(facts.estimate_abs, rel=1e-9)
        assert expected == pytest.approx(facts.counterfactual_total, rel=1e-9)

    def test_asking_for_it_changes_nothing_else(self):
        plain = its_lift(_demo("net_revenue"), SUMMER30, kind="money")
        drawn = its_lift(
            _demo("net_revenue"), SUMMER30, kind="money",
            series_days_before=28, series_days_after=21,
        )
        assert (plain.estimate_abs, plain.interval_abs, plain.p_value, plain.verdict) == (
            drawn.estimate_abs, drawn.interval_abs, drawn.p_value, drawn.verdict,
        )
        extra = set(drawn.params) - set(plain.params)
        assert extra == {
            "series_days_before", "series_days_after", "series_band_half_width",
            "series_band_level", "series_band_basis", "series_band_reference",
            "series_band_n_reference",
        }

    def test_negative_days_are_refused(self):
        with pytest.raises(ValueError, match="negative"):
            its_lift(_demo("net_revenue"), SUMMER30, kind="money", series_days_before=-1)
