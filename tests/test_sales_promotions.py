"""The sales tools with promotions (0.1.43): left out of the verdict and the
seasonality as the app does, the weeks after each one checked for a dip,
calendar rows for "all" products, and money written as text.

Simulated weekly sell-out with a Black Friday spike every year and a deal
that pulls sales forward, so each behaviour shows plainly.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from stats_compass_core.sales import wording  # noqa: E402
from stats_compass_core.sales.schema import SalesColumns, build_series, to_numbers  # noqa: E402
from stats_compass_core.state import DataFrameState  # noqa: E402
from stats_compass_core.workflows.sales import (  # noqa: E402
    PromotionLiftInput,
    SalesSeasonalityInput,
    SalesVerdictInput,
    promotion_lift,
    sales_seasonality,
    sales_verdict,
)

WEEKS = pd.date_range(end="2026-10-24", periods=156, freq="W-SAT")
BLACK_FRIDAY = {2023: ("2023-11-17", "2023-11-27"), 2024: ("2024-11-21", "2024-12-02"), 2025: ("2025-11-20", "2025-12-01")}
DEAL = ("2026-04-05", "2026-04-18")  # a deal on the cloths, then a dip


def amazon(seed: int = 5) -> pd.DataFrame:
    """Two ASINs: a pan that Black Friday lifts threefold, cloths with a pull-forward deal."""
    rng = np.random.default_rng(seed)
    years = (WEEKS - WEEKS[0]).days.to_numpy() / 365.25
    rows = []
    for asin, base in (("B0SIM00001", 900.0), ("B0SIM00039", 340.0)):
        units = base * (1 + 0.15 * np.sin(2 * np.pi * (years - 0.6))) * (1 + 0.05 * rng.standard_normal(len(WEEKS)))
        for start, end in BLACK_FRIDAY.values():
            hit = (WEEKS >= start) & (WEEKS - pd.Timedelta(days=6) <= end)
            units[hit] *= 3.0 if asin == "B0SIM00001" else 1.2
        if asin == "B0SIM00039":
            for w, f in (("2026-04-11", 1.9), ("2026-04-18", 1.9), ("2026-04-25", 0.55), ("2026-05-02", 0.7), ("2026-05-09", 0.85)):
                units[WEEKS.get_loc(pd.Timestamp(w))] *= f
        for w, u in zip(WEEKS, units):
            rows.append({"Week ending": w.date().isoformat(), "ASIN": asin, "Shipped COGS": f"£{u * 9.5:,.2f}"})
    return pd.DataFrame(rows)


def calendar() -> pd.DataFrame:
    rows = [{"ASIN": "all", "Event": f"Black Friday {y}", "Start": s, "End": e} for y, (s, e) in BLACK_FRIDAY.items()]
    rows.append({"ASIN": "B0SIM00039", "Event": "Lightning deal", "Start": DEAL[0], "End": DEAL[1]})
    return pd.DataFrame(rows)


def state_with(**frames) -> DataFrameState:
    state = DataFrameState()
    for name, frame in frames.items():
        state.set_dataframe(frame, name=name, operation="test")
    return state


CAL = {"calendar_dataframe": "cal", "calendar_start_column": "Start", "calendar_end_column": "End", "calendar_name_column": "Event"}


class TestVerdict:
    def test_promotions_left_out_of_whats_expected(self):
        state = state_with(sales=amazon(), cal=calendar())
        common = {"dataframe_name": "sales", "date_column": "Week ending", "value_column": "Shipped COGS", "currency": "£"}
        with_promos = sales_verdict(state, SalesVerdictInput(**common))
        without = sales_verdict(state, SalesVerdictInput(**common, **CAL))
        a, b = with_promos.rows[0], without.rows[0]
        # Black Friday in the history widens the range; left out, it tightens.
        assert (b.expected_high - b.expected_low) < 0.6 * (a.expected_high - a.expected_low)
        assert abs(b.expected - b.observed) < abs(a.expected - a.observed)
        assert wording.PROMOTIONS_LEFT_OUT in without.notes and wording.PROMOTIONS_LEFT_OUT not in with_promos.notes

    def test_a_period_with_a_promotion_says_so(self):
        state = state_with(sales=amazon(), cal=calendar())
        result = sales_verdict(state, SalesVerdictInput(
            dataframe_name="sales", date_column="Week ending", value_column="Shipped COGS", group_by=["ASIN"],
            period_start=date(2026, 4, 12), period_end=date(2026, 4, 18), **CAL,
        ))
        cloths = next(r for r in result.rows if r.group == "B0SIM00039")
        assert cloths.promotions_in_period == ["Lightning deal"]
        assert cloths.detail.endswith("This week had a promotion: Lightning deal.")

    def test_one_source_at_most(self):
        with pytest.raises(ValueError, match="one way, or not at all"):
            SalesVerdictInput(date_column="d", value_column="v", promo_column="p", calendar_dataframe="c")


class TestSeasonality:
    def test_black_friday_left_out_of_november(self):
        state = state_with(sales=amazon(), cal=calendar())
        common = {"dataframe_name": "sales", "date_column": "Week ending", "value_column": "Shipped COGS", "group_by": ["ASIN"]}
        nov = lambda r: next(m.effect_pct for m in r.rows[0].months if m.month == 11)  # noqa: E731
        with_promos = sales_seasonality(state, SalesSeasonalityInput(**common))
        without = sales_seasonality(state, SalesSeasonalityInput(**common, **CAL))
        assert with_promos.rows[0].group == "B0SIM00001"
        assert nov(with_promos) > nov(without) + 20
        assert wording.PROMOTIONS_LEFT_OUT in without.rows[0].basis


class TestLift:
    def test_the_dip_after_a_pull_forward_is_reported(self):
        state = state_with(sales=amazon(), cal=calendar())
        result = promotion_lift(state, PromotionLiftInput(
            dataframe_name="sales", date_column="Week ending", value_column="Shipped COGS", group_by=["ASIN"],
            currency="£", **CAL,
        ))
        deal = next(r for r in result.rows if r.promotion == "Lightning deal")
        assert deal.verdict == "real change"
        assert deal.after_pct == pytest.approx(-30, abs=12)
        assert "In the 3 weeks after, sales ran " in deal.after_summary
        assert deal.after_summary.endswith("customers may have stocked up.")
        assert deal.summary.endswith(deal.after_summary)
        assert deal.series is None  # not asked for

    def test_all_rows_in_a_calendar_reach_every_group(self):
        state = state_with(sales=amazon(), cal=calendar())
        result = promotion_lift(state, PromotionLiftInput(
            dataframe_name="sales", date_column="Week ending", value_column="Shipped COGS", group_by=["ASIN"], **CAL,
        ))
        black_friday_2025 = [r.group for r in result.rows if r.promotion == "Black Friday 2025"]
        assert sorted(black_friday_2025) == ["B0SIM00001", "B0SIM00039"]

    @pytest.mark.parametrize("wild", ["all", "ALL", "*", "", None])
    def test_wildcards_in_a_window_list(self, wild):
        state = state_with(sales=amazon())
        result = promotion_lift(state, PromotionLiftInput(
            dataframe_name="sales", date_column="Week ending", value_column="Shipped COGS", group_by=["ASIN"],
            windows=[{"name": "BF", "start": "2025-11-20", "end": "2025-12-01", "group": {"ASIN": wild}}],
        ))
        assert len(result.rows) == 2

    def test_a_first_year_promotion_says_it_needs_a_year_before_it(self):
        state = state_with(sales=amazon(), cal=calendar())
        result = promotion_lift(state, PromotionLiftInput(
            dataframe_name="sales", date_column="Week ending", value_column="Shipped COGS", group_by=["ASIN"], **CAL,
        ))
        first = next(r for r in result.rows if r.promotion == "Black Friday 2024" and r.group == "B0SIM00001")
        assert first.insufficient_reason == "NO_SEASONAL_BEFORE"
        assert "The promotion needs a year of data before it" in first.summary


class TestMoneyAsText:
    def test_parsed_and_said(self):
        state = state_with(sales=amazon(), cal=calendar())
        result = sales_verdict(state, SalesVerdictInput(dataframe_name="sales", date_column="Week ending", value_column="Shipped COGS"))
        assert any(n.startswith("Money written as text in 'Shipped COGS' (for example '£") for n in result.notes)

    def test_the_forms_read(self):
        values, example = to_numbers(pd.Series(["£1,234.56", "1,234.56", "(£12.00)", "-", "-£3.50", "42", None]))
        assert values.tolist()[:3] == [1234.56, 1234.56, -12.0]
        assert np.isnan(values.iloc[3]) and values.iloc[4] == -3.5 and values.iloc[5] == 42.0 and np.isnan(values.iloc[6])
        assert example == "£1,234.56"

    def test_a_dash_is_missing_not_zero_and_text_is_still_refused(self):
        df = pd.DataFrame({"d": pd.date_range("2025-01-04", periods=4, freq="W-SAT"), "v": ["£10.00", "-", "£12.00", "£11.00"]})
        series = build_series(df, SalesColumns(date_column="d", value_column="v"))
        assert np.isnan(series.groups[0].values.iloc[1])
        df.loc[2, "v"] = "n/a"
        with pytest.raises(ValueError, match="1 value in 'v' isn't a number: 'n/a'"):
            build_series(df, SalesColumns(date_column="d", value_column="v"))

    def test_plain_numbers_say_nothing(self):
        df = pd.DataFrame({"d": pd.date_range("2025-01-04", periods=4, freq="W-SAT"), "v": [10.0, 11.0, 12.0, 13.0]})
        assert build_series(df, SalesColumns(date_column="d", value_column="v")).parsed_text is None


class TestFilledWeeks:
    def test_the_warning_is_passed_through(self):
        df = amazon()
        df = df[~((df.ASIN == "B0SIM00001") & df["Week ending"].isin(["2025-06-14", "2025-06-21"]))]
        result = sales_verdict(state_with(sales=df), SalesVerdictInput(
            dataframe_name="sales", date_column="Week ending", value_column="Shipped COGS", group_by=["ASIN"],
        ))
        pan = next(r for r in result.rows if r.group == "B0SIM00001")
        assert "2 weeks had no data and were filled in from the weeks either side (gaps of up to 2 weeks are filled)." in pan.warnings


class TestChance:
    def test_many_products_a_note_about_chance(self):
        assert wording.chance_note(43, 2) == (
            "With 43 groups judged, about 1 could show a real change by chance alone: look for a cause before acting on one."
        )
        assert wording.chance_note(43, 0) is None and wording.chance_note(5, 1) is None


class TestOneBasis:
    """The amount range is the percentage range applied to the expected total (0.1.44).

    On TATASTEEL's turnover, which grew many times over its 21 years, the
    series' own amount range ranked raw errors from years when turnover was
    far smaller: it read +15% to +65% while the percentage range, the verdict's
    basis, read −19% to +99% (Expected, p 0.20).
    """

    def test_the_two_ranges_agree(self):
        from pathlib import Path

        df = pd.read_csv(Path(__file__).parents[1] / "stats_compass_core" / "datasets" / "TATASTEEL.csv")
        result = promotion_lift(state_with(t=df), PromotionLiftInput(
            dataframe_name="t", date_column="Date", value_column="Turnover",
            windows=[{"name": "W", "start": "2020-11-02", "end": "2020-11-13"}],
        ))
        row = result.rows[0]
        lo, hi = row.interval_pct
        assert row.interval == pytest.approx((row.expected_total * lo / 100, row.expected_total * hi / 100))
        assert (row.interval[0] > 0) == (lo > 0)  # both include zero, as the verdict says
        assert row.verdict == "expected"

    def test_weekends_with_no_rows_are_named(self):
        from pathlib import Path

        df = pd.read_csv(Path(__file__).parents[1] / "stats_compass_core" / "datasets" / "TATASTEEL.csv")
        result = promotion_lift(state_with(t=df), PromotionLiftInput(
            dataframe_name="t", date_column="Date", value_column="Turnover",
            windows=[{"name": "W", "start": "2020-11-02", "end": "2020-11-13"}],
        ))
        filled = next(w for w in result.rows[0].warnings if "filled in" in w)
        assert filled.startswith("2,474 days had no data and were filled in from the days either side (gaps of up to 7 days are filled).")
        assert filled.endswith("Most are Saturdays and Sundays: if nothing sells on those days, give them as 0, since filling them in adds sales that didn't happen.")
        assert not any("(s)" in w for w in result.rows[0].warnings)
