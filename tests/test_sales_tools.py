"""The sales tools: the series statistics on any sales DataFrame (0.1.42).

``sales_verdict``, ``sales_seasonality`` and ``promotion_lift`` run the
Shopify app's statistics, with its settings, on daily or weekly rows, per
group. On the demo store's daily revenue they must give the app's own lift,
and every sentence follows the app's copy rules.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from stats_compass_core.sales import wording  # noqa: E402
from stats_compass_core.sales.schema import (  # noqa: E402
    MAX_GROUPS,
    SalesColumns,
    build_series,
    detect_freq,
)
from stats_compass_core.series import its_lift  # noqa: E402
from stats_compass_core.state import DataFrameState  # noqa: E402
from stats_compass_core.transforms.melt import MeltInput, melt  # noqa: E402
from stats_compass_core.workflows.sales import (  # noqa: E402
    PromotionLiftInput,
    SalesSeasonalityInput,
    SalesVerdictInput,
    promotion_lift,
    sales_seasonality,
    sales_verdict,
)

DEMO = Path(__file__).parent / "fixtures" / "series-demo-daily.csv"
WEEKS = pd.date_range(end="2026-10-24", periods=130, freq="W-SAT")  # weeks ending Saturday


def sell_out(seed: int = 1, jump: str | None = None, promo: bool = True) -> pd.DataFrame:
    """Weekly sell-out by retailer and product, as a retailer export holds it."""
    rng = np.random.default_rng(seed)
    years = (WEEKS - WEEKS[0]).days.to_numpy() / 365.25
    rows = []
    for r_i, retailer in enumerate(["Tesco", "Sainsbury's"]):
        for p_i, product in enumerate(["Choc 100g", "Mint 100g"]):
            base = 5000 * (1 + r_i) * (1 + 0.5 * p_i)
            units = base * (1 + 0.2 * np.sin(2 * np.pi * (years - 0.7))) * (1 + 0.06 * rng.standard_normal(len(WEEKS)))
            on = promo & (WEEKS >= "2026-06-06") & (WEEKS <= "2026-06-27") & (retailer == "Tesco") & (product == "Choc 100g")
            units = np.where(on, units * 1.35, units)
            if jump == f"{retailer} · {product}":
                units[-1] *= 1.8
            for w, u, flag in zip(WEEKS, units, on):
                rows.append({"week_ending": w.date().isoformat(), "retailer": retailer, "product": product,
                             "units": round(u), "value": round(u * 1.5, 2), "on_promo": "Y" if flag else ""})
    return pd.DataFrame(rows)


def state_with(**frames: pd.DataFrame) -> DataFrameState:
    state = DataFrameState()
    for name, frame in frames.items():
        state.set_dataframe(frame, name=name, operation="test")
    return state


# =============================================================================
# The data's shape
# =============================================================================


class TestShape:
    def test_rows_are_summed_per_date_within_each_group(self):
        df = sell_out()
        series = build_series(df, SalesColumns(date_column="week_ending", value_column="units", group_by=["retailer"]))
        assert series.freq == "W" and series.n_groups_found == 2
        tesco = next(g for g in series.groups if g.label == "Tesco")
        first = WEEKS[0].date().isoformat()
        assert tesco.values.iloc[0] == df[(df.retailer == "Tesco") & (df.week_ending == first)].units.sum()

    def test_one_series_without_group_by(self):
        series = build_series(sell_out(), SalesColumns(date_column="week_ending", value_column="units"))
        assert [g.label for g in series.groups] == ["All rows"] and len(series.groups[0].values) == 130

    @pytest.mark.parametrize(
        "dates, freq",
        [
            (pd.date_range("2025-01-01", periods=60, freq="D"), "D"),
            (pd.date_range("2025-01-04", periods=60, freq="W-SAT"), "W"),
        ],
    )
    def test_daily_or_weekly_from_the_dates(self, dates, freq):
        assert detect_freq(dates) == freq

    def test_monthly_is_refused_in_plain_words(self):
        with pytest.raises(ValueError, match="neither daily nor weekly"):
            detect_freq(pd.date_range("2022-01-01", periods=40, freq="MS"))

    def test_bad_columns_dates_and_values_are_named(self):
        df = sell_out()
        with pytest.raises(ValueError, match="not found"):
            build_series(df, SalesColumns(date_column="week", value_column="units"))
        bad = df.copy()
        bad.loc[3, "week_ending"] = "last Tuesday"
        with pytest.raises(ValueError, match="could not be read.*last Tuesday"):
            build_series(bad, SalesColumns(date_column="week_ending", value_column="units"))
        bad = df.copy().astype({"units": object})
        bad.loc[3, "units"] = "n/a"
        with pytest.raises(ValueError, match="not numbers.*n/a"):
            build_series(bad, SalesColumns(date_column="week_ending", value_column="units"))

    def test_at_most_fifty_groups_the_largest(self):
        rows = [{"d": d, "sku": f"S{i:03d}", "v": float(i + 1)} for i in range(MAX_GROUPS + 10)
                for d in pd.date_range("2025-01-01", periods=3, freq="D")]
        series = build_series(pd.DataFrame(rows), SalesColumns(date_column="d", value_column="v", group_by=["sku"]))
        assert series.n_groups_found == MAX_GROUPS + 10 and len(series.groups) == MAX_GROUPS
        assert series.groups[0].label == f"S{MAX_GROUPS + 9:03d}"  # the largest first


# =============================================================================
# sales_verdict
# =============================================================================


class TestVerdict:
    def test_a_jump_in_one_group_comes_first(self):
        state = state_with(sales=sell_out(jump="Tesco · Mint 100g"))
        result = sales_verdict(state, SalesVerdictInput(
            dataframe_name="sales", date_column="week_ending", value_column="value",
            group_by=["retailer", "product"], currency="£",
        ))
        assert result.frequency == "weekly" and result.period == "Week ending 24 October 2026"
        top = result.rows[0]
        assert top.group == "Tesco · Mint 100g" and top.verdict == "real change"
        assert top.summary.startswith("Sales of £") and "outside the expected range" in top.summary
        assert result.summary.startswith("1 of 4 groups had a real change")
        assert wording.WEEKLY_HISTORY_NOTE in result.notes

    def test_daily_rows_are_judged_as_the_app_judges_them(self):
        state = state_with(demo=pd.read_csv(DEMO))
        result = sales_verdict(state, SalesVerdictInput(dataframe_name="demo", date_column="date", value_column="net_revenue"))
        row = result.rows[0]
        assert result.frequency == "daily" and result.period == "21 to 27 September 2026"
        assert row.verdict == "expected"
        assert row.summary == "Nothing here looks unusual: sales of 592,706 are within the expected range for this time of year."
        assert row.detail.endswith(f"{wording.MINUS}18% on the previous week.")

    def test_too_little_history_says_what_it_needs(self):
        short = pd.DataFrame({"w": pd.date_range(end="2026-10-24", periods=60, freq="W-SAT"), "v": 100.0})
        result = sales_verdict(state_with(short=short), SalesVerdictInput(dataframe_name="short", date_column="w", value_column="v"))
        row = result.rows[0]
        assert row.verdict == "insufficient" and row.label == "Not enough data"
        assert "needs at least 76 weeks of history; there are 59 weeks" in row.summary

    def test_a_period_given(self):
        state = state_with(sales=sell_out())
        result = sales_verdict(state, SalesVerdictInput(
            dataframe_name="sales", date_column="week_ending", value_column="units", kind="units",
            period_start=date(2026, 10, 4), period_end=date(2026, 10, 24),
        ))
        assert result.period == "Weeks ending 10 to 24 October 2026"
        assert " units are " in result.rows[0].summary

    def test_period_bounds_must_come_together(self):
        with pytest.raises(ValueError, match="both period_start and period_end"):
            SalesVerdictInput(date_column="d", value_column="v", period_start=date(2026, 1, 1))


# =============================================================================
# sales_seasonality
# =============================================================================


class TestSeasonality:
    def test_twelve_months_with_ranges(self):
        state = state_with(sales=sell_out())
        result = sales_seasonality(state, SalesSeasonalityInput(dataframe_name="sales", date_column="week_ending", value_column="units"))
        row = result.rows[0]
        assert row.found and len(row.months) == 12
        assert result.summary.startswith("Sales usually run highest in ")
        assert row.basis.startswith("Based on about 2 years of data.")


# =============================================================================
# promotion_lift
# =============================================================================


class TestLift:
    def test_the_demo_store_gives_the_apps_own_lift(self):
        demo = pd.read_csv(DEMO)
        result = promotion_lift(state_with(demo=demo), PromotionLiftInput(
            dataframe_name="demo", date_column="date", value_column="net_revenue",
            windows=[{"name": "SUMMER40", "start": "2026-05-14", "end": "2026-06-03"}],
        ))
        row = result.rows[0]
        values = demo.assign(date=pd.to_datetime(demo.date)).set_index("date")["net_revenue"].astype(float)
        core = its_lift(
            values, (date(2026, 5, 14), date(2026, 6, 3)), kind="money", method="fourier", periods=(7, 365),
            noise_band="none", post_days=21, baseline_lookback_days=84, level=0.9,
            seasonal_band_z=2.0, noise_band_z=2.5, real_change_alpha=0.05,
        )
        assert row.estimate_pct == core.estimate_pct and row.interval_pct == core.interval_pct
        assert row.summary == "SUMMER40, 14 May to 3 June 2026: +32%, 914,436 more than expected; likely between +13% and +51%. Real change."

    def test_from_a_promotion_calendar_by_group(self):
        calendar = pd.DataFrame([{"retailer": "Tesco", "product": "Choc 100g", "start": "2026-05-31", "end": "2026-06-27", "mechanic": "2 for £3"}])
        result = promotion_lift(state_with(sales=sell_out(), promos=calendar), PromotionLiftInput(
            dataframe_name="sales", date_column="week_ending", value_column="value", group_by=["retailer", "product"],
            currency="£", calendar_dataframe="promos", calendar_name_column="mechanic",
        ))
        assert len(result.rows) == 1
        row = result.rows[0]
        assert row.group == "Tesco · Choc 100g" and row.verdict == "real change"
        assert row.periods_in_window == ["2026-06-06", "2026-06-13", "2026-06-20", "2026-06-27"]
        assert row.estimate_pct == pytest.approx(35, abs=8)
        assert row.summary.startswith("2 for £3, 31 May to 27 June 2026: +")

    def test_a_calendar_without_a_group_column_covers_all_of_it(self):
        calendar = pd.DataFrame([{"retailer": "Tesco", "start": "2026-05-31", "end": "2026-06-27"}])
        result = promotion_lift(state_with(sales=sell_out(), promos=calendar), PromotionLiftInput(
            dataframe_name="sales", date_column="week_ending", value_column="units", kind="units",
            group_by=["retailer", "product"], calendar_dataframe="promos",
        ))
        assert sorted(r.group for r in result.rows) == ["Tesco · Choc 100g", "Tesco · Mint 100g"]

    def test_from_a_flag_column(self):
        result = promotion_lift(state_with(sales=sell_out()), PromotionLiftInput(
            dataframe_name="sales", date_column="week_ending", value_column="units", kind="units",
            group_by=["retailer", "product"], promo_column="on_promo", include_series=True,
        ))
        row = result.rows[0]
        # Flagged weeks ending 6 to 27 June: the promotion's days run from the
        # first flagged week's Sunday.
        assert (row.start, row.end) == ("2026-05-31", "2026-06-27")
        assert row.verdict == "real change" and " more units than expected" in row.summary
        assert row.series and sum(p.in_promotion for p in row.series) == 4

    def test_one_source_of_promotions(self):
        with pytest.raises(ValueError, match="one way"):
            PromotionLiftInput(date_column="d", value_column="v")
        with pytest.raises(ValueError, match="one way"):
            PromotionLiftInput(date_column="d", value_column="v", promo_column="p", calendar_dataframe="c")

    def test_results_are_plain_json(self):
        result = promotion_lift(state_with(sales=sell_out()), PromotionLiftInput(
            dataframe_name="sales", date_column="week_ending", value_column="units",
            group_by=["retailer"], promo_column="on_promo",
        ))
        assert isinstance(result.model_dump(mode="json")["rows"][0]["summary"], str)


# =============================================================================
# Wide exports: melt, then the tools
# =============================================================================


class TestWideExports:
    def test_melt_then_judge(self):
        long = sell_out()
        wide = long.pivot_table(index=["retailer", "product"], columns="week_ending", values="units").reset_index()
        wide.columns = [str(c) for c in wide.columns]
        state = state_with(wide=wide)
        melted = melt(state, MeltInput(dataframe_name="wide", id_columns=["retailer", "product"],
                                       variable_name="week_ending", value_name="units", save_as="long"))
        assert melted.shape == (4 * 130, 4)
        result = sales_verdict(state, SalesVerdictInput(dataframe_name="long", date_column="week_ending", value_column="units", kind="units"))
        assert result.frequency == "weekly" and result.rows[0].verdict != "insufficient"

    def test_melt_refuses_a_name_clash(self):
        state = state_with(wide=pd.DataFrame({"sku": ["a"], "2026-01-03": [1]}))
        with pytest.raises(ValueError, match="already a kept column"):
            melt(state, MeltInput(dataframe_name="wide", id_columns=["sku"], variable_name="sku"))


# =============================================================================
# The words
# =============================================================================


class TestWording:
    def test_true_minus_and_signs(self):
        assert wording.signed_pct(-24.6) == f"{wording.MINUS}25%"
        assert wording.signed_pct(31.6) == "+32%"
        assert wording.signed_pct(0.2) == "0%"
        assert wording.amount(-1377.3, "money", "£") == f"{wording.MINUS}£1,377"
        assert wording.signed_amount(2031.33, "money", "£") == "+£2,031"

    def test_lift_lines(self):
        assert wording.lift_amount(9144.36, "money", "£") == "£9,144 more than expected"
        assert wording.lift_amount(-120, "units", None) == "120 fewer units than expected"
        assert wording.lift_range("expected", (-5, 9), (-1, 1), "money", "£") is None
        assert wording.lift_range("real change", (13.2, 51.4), (1, 2), "money", "£") == "likely between +13% and +51%"

    def test_no_em_dashes_anywhere(self):
        source = Path(wording.__file__).read_text()
        assert "—" not in source
