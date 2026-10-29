"""Weekly input: the series statistics on one value per week (0.1.42).

Retailer sell-out data arrives weekly, labelled by the week ending, per
retailer and product. ``freq="W"`` runs the same decomposition, expectation
and reference on weeks: every ``*_days`` argument is whole weeks, only the
annual pattern is fitted (52 weeks), and a promotion window covers every week
it touches. Daily results are pinned separately (test_series_daily_golden).
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from stats_compass_core.series import (  # noqa: E402
    Insufficient,
    decompose,
    detect_run,
    forecast,
    its_lift,
    month_effects,
    verdict,
)
from stats_compass_core.series._common import (  # noqa: E402
    prepare_weekly,
    weeks_overlapping,
)

FOURIER = {"method": "fourier", "freq": "W"}


def weekly(
    weeks: int = 156,
    seed: int = 0,
    level: float = 10_000.0,
    amplitude: float = 0.15,
    noise: float = 0.05,
    end: str = "2026-09-27",
    lift: tuple[str, str, float] | None = None,
) -> pd.Series:
    """Weekly sales ending on Sundays: a level, an annual swing and noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=end, periods=weeks, freq="W-SUN")
    days = (idx - idx[0]).days.to_numpy()
    season = 1 + amplitude * np.sin(2 * np.pi * days / 365.25)
    values = level * season * (1 + noise * rng.standard_normal(weeks))
    series = pd.Series(values, index=idx, name="sales")
    if lift is not None:
        start, stop, share = lift
        series.loc[start:stop] *= 1 + share
    return series


# =============================================================================
# Preparing weekly input
# =============================================================================


class TestPrepare:
    def test_one_value_per_week_on_one_weekday(self):
        series = weekly(60)
        prepared = prepare_weekly(series, 2)
        assert len(prepared.values) == 60
        assert prepared.values.index.freqstr == "W-SUN"

    def test_mixed_weekdays_are_refused_in_plain_words(self):
        series = weekly(60)
        series.index = series.index[:-1].append(pd.DatetimeIndex(["2026-09-26"]))
        with pytest.raises(ValueError, match="same weekday.*Sat, Sun"):
            prepare_weekly(series, 2)

    def test_repeated_weeks_are_refused(self):
        series = weekly(60)
        doubled = pd.concat([series, series.iloc[-1:]])
        with pytest.raises(ValueError, match="repeated weeks"):
            prepare_weekly(doubled, 2)

    def test_short_gaps_are_filled_and_named_long_ones_refused(self):
        series = weekly(60)
        gappy = series.drop(series.index[[20, 21]])
        prepared = prepare_weekly(gappy, 2)
        assert [pd.Timestamp(d) for d in prepared.imputed] == list(series.index[[20, 21]])
        refused = prepare_weekly(series.drop(series.index[[20, 21, 22]]), 2)
        assert refused == Insufficient(needs=2, has=3, unit="weeks", reason="GAP_TOO_LONG")

    @pytest.mark.parametrize(
        "week_label, expected",
        [
            # Thursday 14 May to Wednesday 3 June 2026, weeks ending Sunday:
            # the weeks ending 17, 24 and 31 May and 7 June all hold promotion days.
            ("end", ["2026-05-17", "2026-05-24", "2026-05-31", "2026-06-07"]),
            # The same dates read as week starts: weeks from 10 May to 31 May.
            ("start", ["2026-05-10", "2026-05-17", "2026-05-24", "2026-05-31"]),
        ],
    )
    def test_a_window_covers_every_week_it_touches(self, week_label, expected):
        index = pd.date_range("2026-04-05", "2026-07-26", freq="W-SUN")
        touched = weeks_overlapping(index, date(2026, 5, 14), date(2026, 6, 3), week_label)
        assert [d.date().isoformat() for d in touched] == expected


# =============================================================================
# Decomposing weekly input
# =============================================================================


class TestDecompose:
    def test_unit_and_params(self):
        dec = decompose(weekly(), **FOURIER)
        assert dec.unit == "week" and dec.grain == "week"
        assert dec.params["input_freq"] == "W"
        assert dec.params["periods_used"] == [52]
        assert dec.params["n_weeks"] == 156
        assert dec.params["level_real_days"] == 28  # four real weeks, kept in days
        assert dec.daily.empty

    def test_daily_input_is_unchanged_in_kind(self):
        idx = pd.date_range("2024-01-01", periods=800, freq="D")
        daily = pd.Series(100 + np.random.default_rng(1).normal(0, 5, 800), index=idx)
        dec = decompose(daily, method="fourier")
        assert dec.unit == "day" and "input_freq" not in dec.params

    @pytest.mark.parametrize("method, needs", [("fourier", 52), ("stl", 104)])
    def test_history_floors(self, method, needs):
        short = decompose(weekly(needs - 1), method=method, freq="W")
        assert short == Insufficient(needs=needs, has=needs - 1, unit="weeks", reason="TOO_SHORT")
        assert decompose(weekly(needs), method=method, freq="W").unit == "week"

    def test_mstl_is_refused(self):
        with pytest.raises(ValueError, match="'fourier' \\(recommended\\) or 'stl'"):
            decompose(weekly(), method="mstl", freq="W")

    def test_only_the_annual_period(self):
        assert decompose(weekly(), method="fourier", freq="W", periods=(52,)).unit == "week"
        with pytest.raises(ValueError, match="annual pattern only"):
            decompose(weekly(), method="fourier", freq="W", periods=(7,))

    def test_days_must_be_whole_weeks(self):
        with pytest.raises(ValueError, match="trend_window_days must be a multiple of 7"):
            decompose(weekly(), method="fourier", freq="W", trend_window_days=30)

    def test_no_daily_expectation(self):
        dec = decompose(weekly(), **FOURIER)
        with pytest.raises(ValueError, match="no daily expectation"):
            dec.expected_daily(date(2026, 10, 1), date(2026, 10, 7))
        ahead = pd.date_range("2026-10-04", periods=4, freq="W-SUN")
        assert len(dec.expected_steps(ahead)) == 4

    def test_the_annual_swing_is_found(self):
        dec = decompose(weekly(amplitude=0.2, noise=0.02), **FOURIER)
        swing = dec.seasonal.max() - dec.seasonal.min()
        assert swing == pytest.approx(2 * 0.2 * 10_000, rel=0.15)

    def test_excluded_weeks_are_left_out_of_the_fit(self):
        series = weekly(lift=("2026-05-17", "2026-06-07", 0.5))
        dec = decompose(series, **FOURIER, exclude=[(date(2026, 5, 14), date(2026, 6, 3))])
        assert [d.isoformat() for d in dec.excluded] == ["2026-05-17", "2026-05-24", "2026-05-31", "2026-06-07"]
        assert dec.remainder.loc["2026-05-17":"2026-06-07"].isna().all()

    def test_forecast_and_runs_need_daily_input(self):
        dec = decompose(weekly(), **FOURIER)
        with pytest.raises(ValueError, match="daily input"):
            forecast(dec)
        with pytest.raises(ValueError, match="daily input"):
            detect_run(dec)


# =============================================================================
# The verdict on a week
# =============================================================================


def judge(series: pd.Series, weeks: int = 1, **options):
    dec = decompose(series.iloc[:-weeks], **FOURIER)
    return verdict(dec, series.iloc[-weeks:], kind="money", min_points=1, **options)


class TestVerdict:
    def test_an_ordinary_week_is_not_a_real_change(self):
        facts = judge(weekly(seed=3))
        assert facts.verdict in ("expected", "noise")
        assert facts.params["input_freq"] == "W"

    def test_a_jump_is_a_real_change(self):
        series = weekly(seed=3)
        series.iloc[-1] *= 1.6
        facts = judge(series)
        assert facts.verdict == "real change" and facts.observed > facts.expected

    def test_four_weeks_together(self):
        series = weekly(seed=4)
        series.iloc[-4:] *= 1.4
        facts = judge(series, weeks=4)
        assert facts.verdict == "real change"
        assert facts.observed == pytest.approx(series.iloc[-4:].sum())

    def test_a_week_left_out_between_history_and_period(self):
        series = weekly(seed=5)
        dec = decompose(series.iloc[:-2], **FOURIER)
        facts = verdict(dec, series.iloc[-1:], kind="money", min_points=1)
        assert facts == Insufficient(needs=None, has=None, unit="weeks", reason="HISTORY_ENDS_EARLY")

    def test_the_period_must_share_the_weekday(self):
        series = weekly(seed=6)
        dec = decompose(series.iloc[:-1], **FOURIER)
        moved = series.iloc[-1:].copy()
        moved.index = moved.index - pd.Timedelta(days=1)
        with pytest.raises(ValueError, match="same weekday"):
            verdict(dec, moved, kind="money", min_points=1)

    def test_units_too(self):
        rng = np.random.default_rng(7)
        idx = pd.date_range(end="2026-09-27", periods=120, freq="W-SUN")
        units = pd.Series(rng.poisson(400, 120).astype(float), index=idx)
        dec = decompose(units.iloc[:-1], **FOURIER)
        facts = verdict(dec, units.iloc[-1:], kind="count", min_points=1)
        assert facts.test in ("poisson", "negative_binomial")


# =============================================================================
# A promotion's lift, in weeks
# =============================================================================


PROMO = (date(2026, 5, 14), date(2026, 6, 3))


class TestLift:
    def test_a_known_lift_is_found(self):
        series = weekly(seed=8, lift=("2026-05-17", "2026-06-07", 0.3))
        facts = its_lift(series, PROMO, kind="money", method="fourier", freq="W")
        assert facts.verdict == "real change"
        assert facts.estimate_pct == pytest.approx(30, abs=10)
        lo, hi = facts.interval_pct
        assert lo < 30 < hi
        assert facts.params["window_weeks"] == ["2026-05-17", "2026-05-24", "2026-05-31", "2026-06-07"]
        assert facts.params["n_window_weeks_with_data"] == 4
        assert "n_window_days_with_data" not in facts.params

    def test_an_inert_promotion_is_not_a_real_change(self):
        facts = its_lift(weekly(seed=9), PROMO, kind="money", method="fourier", freq="W")
        assert facts.verdict in ("expected", "noise")

    def test_the_weekly_series_for_a_chart(self):
        series = weekly(seed=10, lift=("2026-05-17", "2026-06-07", 0.3))
        facts = its_lift(
            series, PROMO, kind="money", method="fourier", freq="W",
            series_days_before=28, series_days_after=21,
        )
        rows = facts.series
        assert [r.date.isoformat() for r in rows][:5] == ["2026-04-19", "2026-04-26", "2026-05-03", "2026-05-10", "2026-05-17"]
        assert len(rows) == 4 + 4 + 3
        inside = [r for r in rows if r.in_window]
        assert len(inside) == 4
        # The gap between the lines over the window is the lift.
        assert sum(r.actual - r.expected for r in inside) == pytest.approx(facts.estimate_abs)

    def test_counts_of_days_must_be_whole_weeks(self):
        with pytest.raises(ValueError, match="post_days must be a multiple of 7"):
            its_lift(weekly(), PROMO, kind="money", method="fourier", freq="W", post_days=10)

    def test_a_window_with_no_data_yet(self):
        facts = its_lift(weekly(), (date(2026, 10, 1), date(2026, 10, 20)), kind="money", method="fourier", freq="W")
        assert facts == Insufficient(needs=2, has=0, unit="weeks", reason="TOO_FEW_POINTS")

    def test_a_window_too_early_for_a_baseline(self):
        series = weekly(80)
        first = series.index[0].date()
        facts = its_lift(series, (first, first), kind="money", method="fourier", freq="W")
        assert isinstance(facts, Insufficient) and facts.unit == "weeks"


# =============================================================================
# Seasonality, by month, from weeks
# =============================================================================


class TestMonths:
    def test_a_december_peak_is_found(self):
        series = weekly(seed=11, amplitude=0.0, noise=0.03)
        december = series.index.month == 12
        series[december] *= 1.3
        effects = month_effects(decompose(series, **FOURIER))
        dec_row = next(r for r in effects.rows if r.month == 12)
        assert dec_row.effect_pct > 15
        assert dec_row.lower > 0

    @pytest.mark.parametrize("week_label, boosted, other", [("end", 2, 3), ("start", 3, 2)])
    def test_a_week_counts_towards_the_month_of_its_middle(self, week_label, boosted, other):
        # Weeks dated on the 1st to 3rd of March: a week ending then is mostly
        # February (its middle is 26 to 28 February), a week starting then is
        # March. Triple only those weeks and see which month moves.
        series = weekly(seed=12, amplitude=0.0, noise=0.02)
        edge = (series.index.month == 3) & (series.index.day <= 3)
        assert edge.sum() >= 2
        series[edge] *= 3
        effects = month_effects(decompose(series, method="fourier", freq="W", week_label=week_label))
        by_month = {r.month: r.effect_pct for r in effects.rows}
        assert by_month[boosted] > by_month[other] + 10
