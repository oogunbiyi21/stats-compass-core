"""Promotion lift in the sales layer, 0.1.45: the percentage range as the test
inverted, closures left out as missing days, and notes on what a promotion was
judged against.

From the sale checker's validation on UCI Online Retail II (stats-compass-mcp,
validation/README.md): plus or minus q held a +80% lift 65% of the time, and a
Christmas closure read as zero sales made most January windows a real change.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from stats_compass_core.sales import analysis, wording  # noqa: E402
from stats_compass_core.state import DataFrameState  # noqa: E402
from stats_compass_core.workflows.sales import PromotionLiftInput, promotion_lift  # noqa: E402


def store(days: int = 900, end: str = "2026-10-31", closed: bool = True, seed: int = 3, zero_share: float = 0.0) -> pd.DataFrame:
    """A daily store, closed 24 Dec – 3 Jan each year when ``closed``, with a quiet weekly cycle."""
    rng = np.random.default_rng(seed)
    day = pd.date_range(end=end, periods=days, freq="D")
    sales = np.asarray(1000 * (1 + 0.15 * np.sin(2 * np.pi * day.dayofweek / 7))) * rng.lognormal(0, 0.12, days)
    if closed:
        shut = ((day.month == 12) & (day.day >= 24)) | ((day.month == 1) & (day.day <= 3))
        sales[shut] = 0.0
    if zero_share:
        sales[rng.random(days) < zero_share] = 0.0
    return pd.DataFrame({"Date": day.strftime("%Y-%m-%d"), "Sales": sales.round(2)})


def lift(df: pd.DataFrame, start: str, end: str):
    state = DataFrameState()
    state.set_dataframe(df, name="s", operation="upload")
    return promotion_lift(state, PromotionLiftInput(
        dataframe_name="s", date_column="Date", value_column="Sales",
        windows=[{"name": "Sale", "start": start, "end": end}],
    )).rows[0]


class TestPercentRange:
    def test_the_test_inverted(self):
        # +80% with a half-width of 60 points: from (0.8 − 0.6)/1.6 to (0.8 + 0.6)/0.4.
        assert analysis._percent_range(80.0, (20.0, 140.0)) == pytest.approx((12.5, 350.0))

    def test_it_excludes_zero_exactly_when_the_test_does(self):
        for e, q in ((30.0, 29.9), (30.0, 30.1), (-20.0, 19.0), (-20.0, 21.0)):
            low, high = analysis._percent_range(e, (e - q, e + q))
            assert (low > 0 or high < 0) == (abs(e) > q)

    def test_unbounded_above_when_the_half_width_reaches_100(self):
        low, high = analysis._percent_range(150.0, (30.0, 270.0))
        assert high is None and low == pytest.approx(100 * (1.5 - 1.2) / 2.2)
        assert wording.lift_range("real change", (low, high), (0.0, 0.0), "money", "£") == "likely above +14%"


class TestClosures:
    def test_a_closure_is_left_out_and_named(self):
        row = lift(store(), "2026-01-12", "2026-01-18")
        assert row.verdict != "insufficient"
        assert any(w.startswith("Runs of days with no sales were taken as closures") and "24 December 2025 – 3 January 2026" in w
                   for w in row.warnings)
        assert not any("filled in from the days either side" in w for w in row.warnings)  # closures aren't missing data

    def test_january_is_judged_against_trading_days(self):
        # A week in January with no sale, after a Christmas closure. Read as zeros, the
        # closure pulls what's expected down; as missing days it doesn't.
        masked = lift(store(), "2026-01-19", "2026-01-25")
        original = analysis.CLOSURE_MIN_DAYS
        analysis.CLOSURE_MIN_DAYS = 10_000
        try:
            zeros = lift(store(), "2026-01-19", "2026-01-25")
        finally:
            analysis.CLOSURE_MIN_DAYS = original
        assert abs(masked.estimate_pct) < 10
        assert abs(masked.estimate_pct) < abs(zeros.estimate_pct)

    def test_a_sale_just_after_a_closure_says_so(self):
        row = lift(store(), "2026-01-06", "2026-01-12")
        assert wording.after_closure(3) in row.warnings
        assert not any("after a closure" in w for w in lift(store(), "2026-03-02", "2026-03-08").warnings)

    def test_not_in_a_store_that_rarely_sells(self):
        row = lift(store(closed=False, zero_share=0.6), "2026-03-02", "2026-03-08")
        assert not any("closures" in w for w in row.warnings)

    def test_not_inside_the_promotion(self):
        df = store(closed=False)
        df.loc[df["Date"].between("2026-03-03", "2026-03-06"), "Sales"] = 0.0
        row = lift(df, "2026-03-02", "2026-03-08")
        assert not any("closures" in w for w in row.warnings)
        assert row.estimate_pct < -40  # the closed days count against it, as they happened

    def test_a_real_gap_is_still_too_long(self):
        df = store()
        df = df[~df["Date"].between("2025-06-01", "2025-06-20")]
        row = lift(df, "2026-03-02", "2026-03-08")
        assert row.verdict == "insufficient"


class TestOneEarlierYear:
    def test_said_under_two_years_of_history(self):
        assert wording.ONE_EARLIER_YEAR in lift(store(days=600), "2026-09-07", "2026-09-13").warnings

    def test_not_with_two_years(self):
        assert wording.ONE_EARLIER_YEAR not in lift(store(days=900), "2026-09-07", "2026-09-13").warnings


def lift_among(df: pd.DataFrame, start: str, end: str, others: list[tuple[str, str]]):
    """The row for one promotion, measured with others in the same data (each left out of the others' fits)."""
    state = DataFrameState()
    state.set_dataframe(df, name="s", operation="upload")
    windows = [{"name": "Sale", "start": start, "end": end}] + [{"name": f"Other {i}", "start": a, "end": b} for i, (a, b) in enumerate(others)]
    rows = promotion_lift(state, PromotionLiftInput(dataframe_name="s", date_column="Date", value_column="Sales", windows=windows)).rows
    return next(r for r in rows if r.promotion == "Sale")


class TestTheWeeksAfter:
    """Another promotion inside the 3 weeks after one isn't that one's after-effect (0.1.46).

    On the sale checker's demo store, SUMMER40's weeks after read "stayed 19%
    above expected": that was FLASH20, 12 days later.
    """

    def test_judged_up_to_the_next_promotion(self):
        row = lift_among(store(closed=False), "2026-03-02", "2026-03-08", [("2026-03-20", "2026-03-26")])
        assert row.after_summary.startswith("In the 11 days before the next promotion, ")

    def test_too_soon_to_judge(self):
        row = lift_among(store(closed=False), "2026-03-02", "2026-03-08", [("2026-03-13", "2026-03-19")])
        assert row.after_summary == "The next promotion started 5 days later, so the weeks after can't be judged."
        assert row.after_pct is None and row.after_difference is None

    def test_unchanged_without_one(self):
        for others in ([], [("2026-04-20", "2026-04-26")]):
            assert lift_among(store(closed=False), "2026-03-02", "2026-03-08", others).after_summary.startswith("In the 3 weeks after, ")

    def test_weekly(self):
        daily = store(closed=False)
        weekly = (daily.assign(Date=pd.to_datetime(daily["Date"])).set_index("Date")["Sales"].resample("W-SAT").sum()
                  .iloc[1:-1].reset_index())
        weekly["Date"] = weekly["Date"].dt.strftime("%Y-%m-%d")
        row = lift_among(weekly, "2026-04-05", "2026-04-18", [("2026-05-03", "2026-05-09")])
        assert row.after_summary.startswith("In the 2 weeks before the next promotion, ")
        row = lift_among(weekly, "2026-04-05", "2026-04-18", [("2026-04-22", "2026-04-28")])
        assert row.after_summary == "The next promotion started 4 days later, so the weeks after can't be judged."
