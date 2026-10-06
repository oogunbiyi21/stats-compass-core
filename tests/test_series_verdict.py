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
        facts = verdict(dec, period, kind="money", seed=1)
        assert isinstance(facts, VerdictFacts)
        assert (facts.verdict, facts.decided_by) == ("expected", "expected_band")
        assert facts.observed == pytest.approx(facts.expected)

    def test_outside_the_expected_band_but_inside_the_noise_band_is_noise(self):
        dec, _ = _split(_demo("net_revenue"))
        period = dec.expected_daily(WEEK_START, WEEK_END) * 1.03
        facts = verdict(
            dec, period, kind="money", seed=1, seasonal_band_z=0.0, noise_band_z=100.0
        )
        assert (facts.verdict, facts.decided_by) == ("noise", "noise_band")

    def test_far_outside_both_bands_and_significant_is_a_real_change(self):
        dec, _ = _split(_demo("net_revenue"))
        period = dec.expected_daily(WEEK_START, WEEK_END) * 1.8
        facts = verdict(dec, period, kind="money", seed=1)
        assert (facts.verdict, facts.decided_by) == ("real change", "test")
        assert facts.p_value < 0.05

    def test_outside_both_bands_but_not_significant_is_noise(self):
        """With both bands collapsed, only the test stands between a figure and 'real change'."""
        dec, _ = _split(_demo("net_revenue"))
        period = dec.expected_daily(WEEK_START, WEEK_END) * 1.001
        facts = verdict(
            dec, period, kind="money", seed=1, seasonal_band_z=0.0, noise_band_z=0.0
        )
        assert (facts.verdict, facts.decided_by) == ("noise", "test")

    def test_bands_are_centred_where_section_7_3_says(self):
        series = _demo("net_revenue")
        dec, period = _split(series)
        facts = verdict(dec, period, kind="money", seed=1)
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
        facts = verdict(
            dec,
            dec.expected_daily(WEEK_START, WEEK_END),
            kind="money",
            seed=1,
            n_boot=500,
        )
        assert 0 < facts.p_value <= 1
        assert facts.n_boot == 500
        assert facts.test == "block_bootstrap"

    def test_analytic_tests_record_no_n_boot(self):
        dec, period = _split(_demo("order_count"))
        facts = verdict(dec, period, kind="count", seed=1)
        assert facts.n_boot is None
        assert facts.test in ("poisson", "negative_binomial")


# =============================================================================
# Calibration on stores with nothing to find
# =============================================================================


def _null_rates(make, kind, seeds, **kwargs):
    significant = real = 0
    for seed in seeds:
        series = make(seed)
        start, end = _last_week(series)
        dec, period = _split(series, start, end, method="fourier")
        facts = verdict(dec, period, kind=kind, seed=seed, **kwargs)
        significant += facts.p_value < 0.05
        real += facts.verdict == "real change"
    return significant / len(seeds), real / len(seeds)


class TestCalibration:
    def test_money_test_alone_fires_at_about_alpha_and_the_bands_cut_it(self):
        test_only, gated = _null_rates(
            _null_money, "money", range(200, 260), n_boot=999
        )
        assert test_only <= 0.15, test_only
        assert gated <= test_only
        assert gated <= 0.05

    def test_count_test_alone_fires_at_about_alpha_and_the_bands_cut_it(self):
        test_only, gated = _null_rates(_null_counts, "count", range(300, 360))
        assert test_only <= 0.15, test_only
        assert gated <= test_only
        assert gated <= 0.05


class TestCountTest:
    def test_poisson_counts_use_poisson(self):
        series = _null_counts(1)
        dec, period = _split(series, *_last_week(series), method="fourier")
        assert verdict(dec, period, kind="count", seed=1).test == "poisson"

    def test_overdispersed_counts_use_negative_binomial(self):
        rng = np.random.default_rng(2)
        idx = pd.date_range("2024-09-16", periods=742, freq="D")
        series = pd.Series(rng.negative_binomial(2, 0.1, 742).astype(float), index=idx)
        dec, period = _split(series, *_last_week(series), method="fourier")
        assert verdict(dec, period, kind="count", seed=1).test == "negative_binomial"


# =============================================================================
# Means and ratios
# =============================================================================


class TestWeightedMean:
    def test_observed_is_the_weighted_mean(self):
        series, orders = _demo("aov"), _demo("orders")
        dec, period = _split(series)
        weights = orders.loc[str(WEEK_START) : str(WEEK_END)]
        facts = verdict(
            dec, period, kind="money", seed=1, aggregate="mean", weights=weights
        )
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
        result = verdict(dec, period, kind="ratio", seed=1, aggregate="mean")
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
            seed=1,
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
        result = verdict(dec, period, kind="money", seed=1)
        assert isinstance(result, Insufficient)
        assert (result.reason, result.needs, result.has) == ("TOO_FEW_POINTS", 5, 4)

    def test_a_decomposition_that_contains_the_period_is_refused(self):
        series = _demo("net_revenue")
        whole = decompose(series)
        with pytest.raises(ValueError, match="before the period"):
            verdict(
                whole, series.loc[str(WEEK_START) : str(WEEK_END)], kind="money", seed=1
            )

    def test_deterministic(self):
        dec, period = _split(_demo("net_revenue"))
        a = verdict(dec, period, kind="money", seed=7)
        b = verdict(dec, period, kind="money", seed=7)
        assert (a.p_value, a.expected_range, a.noise_range) == (
            b.p_value,
            b.expected_range,
            b.noise_range,
        )

    def test_seed_is_required(self):
        dec, period = _split(_demo("net_revenue"))
        with pytest.raises(TypeError):
            verdict(dec, period, kind="money")  # type: ignore[call-arg]

    def test_params_record_every_argument(self):
        dec, period = _split(_demo("net_revenue"))
        facts = verdict(dec, period, kind="money", seed=3, seasonal_band_z=1.5)
        for key in (
            "kind",
            "seed",
            "aggregate",
            "seasonal_band_z",
            "noise_band_z",
            "real_change_alpha",
            "n_boot",
            "min_points",
            "period_start",
            "period_end",
        ):
            assert key in facts.params, key
        assert facts.params["seasonal_band_z"] == 1.5
        assert facts.method == "mstl+block_bootstrap"


# =============================================================================
# The demo store
# =============================================================================


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
        facts = verdict(dec, period, kind="count", seed=1)
        assert facts.observed == 55
        assert facts.observed < facts.expected_range[0]
        assert (facts.verdict, facts.decided_by) == ("noise", "noise_band")
        assert facts.p_value < 0.05  # the test alone would have called it

    def test_judged_out_of_sample_the_last_week_is_far_below_expected(self):
        """Fitted including the week, expected was 79.5 and z −1.26; ahead, z < −3."""
        dec, period = _split(_demo("new_customers"))
        facts = verdict(dec, period, kind="count", seed=1)
        sigma = (
            facts.expected_range[1] - facts.expected_range[0]
        ) / 4  # seasonal_band_z = 2
        assert (facts.observed - facts.expected) / sigma < -3

    def test_returning_customers_holding_is_not_a_real_change(self):
        dec, period = _split(_demo("returning_customers"))
        facts = verdict(dec, period, kind="count", seed=1)
        assert facts.verdict != "real change"
