"""verdict: architecture §7.3, steps in a fixed order.

A figure can only be a real change if it lies outside the expected band, then
outside the noise band, and then survives the test that fits its kind. The
expectation comes from a decomposition of the history before the period: one
that included the period would let the trend lean towards the week being
judged.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from stats_compass_core.series import (  # noqa: E402
    Insufficient,
    VerdictFacts,
    decompose,
    verdict,
)

DEMO = Path(__file__).parent / "fixtures" / "demo_store_daily.csv"
WEEK_START, WEEK_END = date(2026, 9, 21), date(2026, 9, 27)


def _demo(column: str) -> pd.Series:
    return (
        pd.read_csv(DEMO, parse_dates=["date"]).set_index("date")[column].astype(float)
    )


def _split(series: pd.Series, start: date = WEEK_START, end: date = WEEK_END, **kwargs):
    before = series.loc[: pd.Timestamp(start) - pd.Timedelta(days=1)]
    return decompose(before, **kwargs), series.loc[str(start) : str(end)]


def _null_money(seed: int, days: int = 742) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-09-16", periods=days, freq="D")
    counts = rng.poisson(30, days)
    return pd.Series(
        [rng.lognormal(np.log(2800), 0.55, c).sum() for c in counts], index=idx
    )


def _null_counts(seed: int, days: int = 742, rate: float = 20.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-09-16", periods=days, freq="D")
    return pd.Series(rng.poisson(rate, days).astype(float), index=idx)


def _last_week(series: pd.Series) -> tuple[date, date]:
    end = series.index[-1].date()
    return end - timedelta(days=6), end


# =============================================================================
# The order of the steps
# =============================================================================


class TestSteps:
    def test_a_week_at_its_expectation_is_expected(self):
        dec, _ = _split(_demo("net_revenue"))
        period = dec.expected_daily(WEEK_START, WEEK_END)
        facts = verdict(dec, period, kind="money")
        assert isinstance(facts, VerdictFacts)
        assert (facts.verdict, facts.decided_by) == ("expected", "expected_band")
        assert facts.observed == pytest.approx(facts.expected)

    def test_outside_the_expected_band_but_inside_the_noise_band_is_noise(self):
        dec, _ = _split(_demo("net_revenue"))
        period = dec.expected_daily(WEEK_START, WEEK_END) * 1.03
        facts = verdict(
            dec, period, kind="money", seasonal_band_z=0.0, noise_band_z=100.0
        )
        assert (facts.verdict, facts.decided_by) == ("noise", "noise_band")

    def test_far_outside_both_bands_and_significant_is_a_real_change(self):
        dec, _ = _split(_demo("net_revenue"))
        period = dec.expected_daily(WEEK_START, WEEK_END) * 1.8
        facts = verdict(dec, period, kind="money")
        assert (facts.verdict, facts.decided_by) == ("real change", "test")
        assert facts.p_value < 0.05

    def test_outside_both_bands_but_not_significant_is_noise(self):
        """With both bands collapsed, only the test stands between a figure and 'real change'."""
        dec, _ = _split(_demo("net_revenue"))
        period = dec.expected_daily(WEEK_START, WEEK_END) * 1.001
        facts = verdict(
            dec, period, kind="money", seasonal_band_z=0.0, noise_band_z=0.0
        )
        assert (facts.verdict, facts.decided_by) == ("noise", "test")

    def test_bands_are_centred_where_section_7_3_says(self):
        series = _demo("net_revenue")
        dec, period = _split(series)
        facts = verdict(dec, period, kind="money")
        lo, hi = facts.expected_range
        assert (lo + hi) / 2 == pytest.approx(facts.expected)
        prior = float(series.loc["2026-09-14":"2026-09-20"].sum())
        nlo, nhi = facts.noise_range
        assert (nlo + nhi) / 2 == pytest.approx(prior)
        assert facts.change_vs_prior[0] == pytest.approx(facts.observed - prior)
        assert facts.change_vs_prior[1] == pytest.approx(
            100 * (facts.observed - prior) / prior
        )


class TestPValueOnEveryFigure:
    def test_recorded_even_when_the_expected_band_decides(self):
        dec, _ = _split(_demo("net_revenue"))
        facts = verdict(dec, dec.expected_daily(WEEK_START, WEEK_END), kind="money")
        assert 0 < facts.p_value <= 1
        assert facts.test == "empirical_windows"
        assert facts.n_reference > 300

    def test_analytic_tests_have_no_reference_windows(self):
        dec, period = _split(_demo("order_count"))
        facts = verdict(dec, period, kind="count")
        assert facts.n_reference is None
        assert facts.test in ("poisson", "negative_binomial")

    def test_the_money_test_is_exact_not_a_random_draw(self):
        """A seeded draw from the reference windows moved p by 3.4× between seeds."""
        dec, period = _split(_demo("net_revenue"))
        facts = verdict(dec, period, kind="money")
        errors = dec.expectation_errors(7)["error"].to_numpy()
        deviation = facts.observed - facts.expected
        exact = (1 + int((abs(errors) >= abs(deviation)).sum())) / (len(errors) + 1)
        assert facts.p_value == exact


# =============================================================================
# Calibration on stores with nothing to find
# =============================================================================


def _null_rates(make, kind, seeds, method, **kwargs):
    significant = real = 0
    for seed in seeds:
        series = make(seed)
        start, end = _last_week(series)
        dec, period = _split(series, start, end, method=method)
        facts = verdict(dec, period, kind=kind, **kwargs)
        significant += facts.p_value < 0.05
        real += facts.verdict == "real change"
    return significant / len(seeds), real / len(seeds)


class TestCalibration:
    """Out of sample, the test alone should fire at about alpha; the bands cut it further.

    The first reference (in-sample remainder windows) fired 15-20% of the time
    at alpha 0.05 for mstl and stl: the fit had already absorbed part of each
    past window's noise, which the period being judged never gets.
    """

    # 80 stores at alpha 0.05: a calibrated test lands between 1 and 10 hits
    # 99% of the time (Binomial(80, 0.05)). On 480 stores fourier's money test
    # fired at 5.8% and 7.1%; these 80 happen to hit 9.
    LOW, HIGH = 1 / 80, 10 / 80

    @pytest.mark.parametrize("method", ["mstl", "stl", "fourier"])
    def test_money(self, method):
        test_only, gated = _null_rates(_null_money, "money", range(200, 280), method)
        assert self.LOW <= test_only <= self.HIGH, f"{method}: {test_only:.3f}"
        assert gated <= test_only

    @pytest.mark.parametrize("method", ["mstl", "stl", "fourier"])
    def test_count(self, method):
        """Discrete, so it may sit below alpha; only the upper bound applies."""
        test_only, gated = _null_rates(_null_counts, "count", range(300, 380), method)
        assert test_only <= self.HIGH, f"{method}: {test_only:.3f}"
        assert gated <= test_only


class TestCountTest:
    def test_poisson_counts_use_poisson(self):
        series = _null_counts(1)
        dec, period = _split(series, *_last_week(series), method="fourier")
        assert verdict(dec, period, kind="count").test == "poisson"

    def test_overdispersed_counts_use_negative_binomial(self):
        rng = np.random.default_rng(2)
        idx = pd.date_range("2024-09-16", periods=742, freq="D")
        series = pd.Series(rng.negative_binomial(2, 0.1, 742).astype(float), index=idx)
        dec, period = _split(series, *_last_week(series), method="fourier")
        assert verdict(dec, period, kind="count").test == "negative_binomial"


# =============================================================================
# Means and ratios
# =============================================================================


class TestWeightedMean:
    def test_observed_is_the_weighted_mean(self):
        series, orders = _demo("aov"), _demo("orders")
        dec, period = _split(series)
        weights = orders.loc[str(WEEK_START) : str(WEEK_END)]
        facts = verdict(dec, period, kind="money", aggregate="mean", weights=weights)
        assert facts.observed == pytest.approx(
            float((period * weights).sum() / weights.sum())
        )


class TestRatio:
    def _rates(self, seed: int, drop_last_week: float = 1.0):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2024-09-16", periods=742, freq="D")
        n = pd.Series(rng.poisson(40, 742).astype(float), index=idx)
        p = np.full(742, 0.25)
        p[-7:] *= drop_last_week
        rate = pd.Series(rng.binomial(n.astype(int), p) / n, index=idx)
        return rate, n

    def test_without_a_denominator_it_is_insufficient(self):
        rate, _ = self._rates(1)
        dec, period = _split(rate, *_last_week(rate), method="fourier")
        result = verdict(dec, period, kind="ratio", aggregate="mean")
        assert isinstance(result, Insufficient)
        assert result.reason == "NO_DENOMINATOR"

    def test_a_halved_rate_is_a_real_change(self):
        rate, n = self._rates(1, drop_last_week=0.5)
        start, end = _last_week(rate)
        dec, period = _split(rate, start, end, method="fourier")
        facts = verdict(
            dec,
            period,
            kind="ratio",
            aggregate="mean",
            weights=n.loc[str(start) : str(end)],
        )
        assert facts.test == "binomial"
        assert facts.verdict == "real change"


# =============================================================================
# Insufficiency and the contract
# =============================================================================


class TestInsufficientAndContract:
    def test_too_few_points_in_the_period(self):
        dec, period = _split(_demo("net_revenue"))
        period = period.copy()
        period.iloc[:3] = np.nan
        result = verdict(dec, period, kind="money")
        assert isinstance(result, Insufficient)
        assert (result.reason, result.needs, result.has) == ("TOO_FEW_POINTS", 5, 4)

    def test_a_decomposition_that_contains_the_period_is_refused(self):
        series = _demo("net_revenue")
        whole = decompose(series)
        with pytest.raises(ValueError, match="before the period"):
            verdict(whole, series.loc[str(WEEK_START) : str(WEEK_END)], kind="money")

    def test_deterministic(self):
        dec, period = _split(_demo("net_revenue"))
        a = verdict(dec, period, kind="money")
        b = verdict(dec, period, kind="money")
        assert (a.p_value, a.expected_range, a.noise_range) == (
            b.p_value,
            b.expected_range,
            b.noise_range,
        )

    def test_nothing_is_drawn_so_there_is_no_seed(self):
        dec, period = _split(_demo("net_revenue"))
        with pytest.raises(TypeError):
            verdict(dec, period, kind="money", seed=1)  # type: ignore[call-arg]

    def test_a_trimmed_last_day_of_history_is_read_across(self):
        """A zero-order day sends null AOV; it must not turn into an exception."""
        series = _demo("aov").copy()
        series.loc["2026-09-20"] = np.nan
        dec, period = _split(series)
        assert dec.trimmed == [date(2026, 9, 20)]
        facts = verdict(dec, period, kind="money", aggregate="mean")
        assert isinstance(facts, VerdictFacts)

    def test_history_that_ends_early_for_another_reason_is_insufficient(self):
        series = _demo("net_revenue")
        dec = decompose(series.loc[:"2026-09-13"], method="fourier")
        result = verdict(dec, series.loc["2026-09-21":"2026-09-27"], kind="money")
        assert isinstance(result, Insufficient)
        assert result.reason == "HISTORY_ENDS_EARLY"

    def test_robust_scale_is_an_option_and_recorded(self):
        dec, period = _split(_demo("net_revenue"))
        plain = verdict(dec, period, kind="money")
        robust = verdict(dec, period, kind="money", robust_scale=True)
        assert (plain.params["scale"], robust.params["scale"]) == ("std", "mad")
        width = lambda f: f.expected_range[1] - f.expected_range[0]  # noqa: E731
        assert width(robust) < width(plain)

    def test_params_record_every_argument(self):
        dec, period = _split(_demo("net_revenue"))
        facts = verdict(dec, period, kind="money", seasonal_band_z=1.5)
        for key in (
            "kind",
            "aggregate",
            "seasonal_band_z",
            "noise_band_z",
            "real_change_alpha",
            "min_points",
            "period_start",
            "period_end",
        ):
            assert key in facts.params, key
        assert facts.params["seasonal_band_z"] == 1.5
        assert facts.method == "mstl+empirical_windows"


# =============================================================================
# The demo store
# =============================================================================


class TestYoungStore:
    def test_a_store_with_four_months_still_gets_a_verdict(self):
        series = _demo("net_revenue").loc["2026-05-25":]
        dec, period = _split(series)
        facts = verdict(dec, period, kind="money")
        assert isinstance(facts, VerdictFacts)
        assert dec.params["periods_used"] == [7]


class TestDemoStore:
    def test_the_last_week_of_the_decline_is_below_expected_but_inside_the_noise_band(
        self,
    ):
        """The planted decline is gradual: 139, 119, 94, 77, then 55 new customers.

        The last week is far below the seasonal expectation, but only 22 below
        the week before, which is inside the noise band. Under §7.3's fixed
        order that makes it noise. Catching a gradual decline is the run
        detector's job (§7.6, brief T7.8), not the headline verdict's.
        """
        dec, period = _split(_demo("new_customers"))
        facts = verdict(dec, period, kind="count")
        assert facts.observed == 55
        assert facts.observed < facts.expected_range[0]
        assert (facts.verdict, facts.decided_by) == ("noise", "noise_band")
        assert facts.p_value < 0.05  # the test alone would have called it

    def test_judged_out_of_sample_the_last_week_is_far_below_expected(self):
        """Fitted including the week, expected was 79.5 and z −1.26; ahead, z < −3."""
        dec, period = _split(_demo("new_customers"))
        facts = verdict(dec, period, kind="count")
        sigma = (
            facts.expected_range[1] - facts.expected_range[0]
        ) / 4  # seasonal_band_z = 2
        assert (facts.observed - facts.expected) / sigma < -3

    def test_returning_customers_holding_is_not_a_real_change(self):
        dec, period = _split(_demo("returning_customers"))
        facts = verdict(dec, period, kind="count")
        assert facts.verdict != "real change"
