"""Stats Compass on any sales data: three top-level tools (0.1.42).

- ``sales_verdict``: is a period's figure expected, a real change, or noise?
- ``sales_seasonality``: what each month usually brings.
- ``promotion_lift``: what each promotion added, against what would have sold
  without it.

The same statistics and settings as the Shopify app, on a DataFrame loaded
from a CSV or Excel file, daily or weekly, as one series or one per group
(retailer, product). The data's shape is defined in ``sales.schema``, the
statistics' settings in ``sales.analysis``, every sentence in
``sales.wording``.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, model_validator

from stats_compass_core.registry import registry
from stats_compass_core.sales import analysis, wording
from stats_compass_core.sales.schema import (
    MAX_GROUPS,
    PromotionWindow,
    SalesColumns,
    build_series,
    windows_for,
    windows_from_calendar,
    windows_from_flags,
)
from stats_compass_core.state import DataFrameState

FREQUENCY = {"D": "daily", "W": "weekly"}


def _notes(series) -> list[str]:
    notes = []
    if series.n_groups_found > len(series.groups):
        notes.append(
            f"{series.n_groups_found} groups were found; the {MAX_GROUPS} with the most "
            f"sales were judged. Filter the data to judge the others."
        )
    if series.freq == "W":
        notes.append(wording.WEEKLY_HISTORY_NOTE)
    return notes


# =============================================================================
# The verdict on a period
# =============================================================================


class SalesVerdictInput(SalesColumns):
    """Input for sales_verdict."""

    period_start: date | None = Field(
        default=None,
        description="First day of the period to judge. Default: the latest week in the data.",
    )
    period_end: date | None = Field(
        default=None,
        description="Last day of the period. Default: the latest week in the data.",
    )

    @model_validator(mode="after")
    def _both_or_neither(self):
        if (self.period_start is None) != (self.period_end is None):
            raise ValueError("Give both period_start and period_end, or neither.")
        if self.period_start and self.period_end and self.period_end < self.period_start:
            raise ValueError("period_end is before period_start.")
        return self


class SalesVerdictResult(BaseModel):
    summary: str = Field(description="The verdict in a sentence; for several groups, how many had a real change.")
    frequency: str = Field(description="'daily' or 'weekly', as detected from the dates.")
    period: str
    rows: list[analysis.PeriodRow] = Field(description="One per group, real changes first.")
    groups_found: int
    notes: list[str] = Field(default_factory=list)


@registry.register(
    category="workflows",
    name="sales_verdict",
    input_schema=SalesVerdictInput,
    description=(
        "Judge a period's sales against the history before it, as the Stats Compass "
        "app does: Expected, Real change, Noise, or Not enough data. Works on daily or "
        "weekly rows (detected from the dates), optionally per group such as retailer "
        "and product. Default period: the latest week."
    ),
    tier="workflow",
)
def sales_verdict(state: DataFrameState, params: SalesVerdictInput) -> SalesVerdictResult:
    series = build_series(state.get_dataframe(params.dataframe_name), params)
    period = (
        (params.period_start, params.period_end)
        if params.period_start
        else analysis.default_period([g.values for g in series.groups], series.freq)
    )
    rows = [
        analysis.judge_period(g, series.freq, series.week_label, params.kind, period, params.currency)
        for g in series.groups
    ]
    rows.sort(key=lambda r: (analysis.ORDER[r.verdict], -abs((r.observed or 0) - (r.expected or 0))))
    return SalesVerdictResult(
        summary=analysis.overview(rows, "groups"),
        frequency=FREQUENCY[series.freq],
        period=wording.period_name(
            date.fromisoformat(rows[0].period_start), date.fromisoformat(rows[0].period_end),
            series.freq == "W", series.week_label,
        ),
        rows=rows,
        groups_found=series.n_groups_found,
        notes=_notes(series),
    )


# =============================================================================
# Seasonality
# =============================================================================


class SalesSeasonalityInput(SalesColumns):
    """Input for sales_seasonality."""


class SalesSeasonalityResult(BaseModel):
    summary: str
    frequency: str
    rows: list[analysis.SeasonalityRow] = Field(description="One per group: each month's effect, with a range.")
    groups_found: int
    notes: list[str] = Field(default_factory=list)


@registry.register(
    category="workflows",
    name="sales_seasonality",
    input_schema=SalesSeasonalityInput,
    description=(
        "Find what each month usually brings: how far it runs above or below the "
        "year's average, with a likely range, as the Stats Compass app does. Daily "
        "or weekly rows, optionally per group."
    ),
    tier="workflow",
)
def sales_seasonality(state: DataFrameState, params: SalesSeasonalityInput) -> SalesSeasonalityResult:
    series = build_series(state.get_dataframe(params.dataframe_name), params)
    rows = [analysis.seasonality(g, series.freq, series.week_label) for g in series.groups]
    if len(rows) == 1:
        summary = rows[0].summary
    else:
        found = sum(r.found for r in rows)
        summary = f"A seasonal pattern was measured for {found} of {len(rows)} groups."
    return SalesSeasonalityResult(
        summary=summary,
        frequency=FREQUENCY[series.freq],
        rows=rows,
        groups_found=series.n_groups_found,
        notes=_notes(series),
    )


# =============================================================================
# A promotion's lift
# =============================================================================


class PromotionLiftInput(SalesColumns):
    """Input for promotion_lift. Give the promotions one way: windows, promo_column, or a calendar."""

    windows: list[PromotionWindow] | None = Field(
        default=None,
        description="The promotions, each with a first and last day, and optionally a name and group.",
    )
    promo_column: str | None = Field(
        default=None,
        description=(
            "A column flagging promotion rows (1, true or yes). Each group's run of "
            "consecutive flagged dates is one promotion."
        ),
    )
    calendar_dataframe: str | None = Field(
        default=None,
        description=(
            "A DataFrame with one row per promotion (a promotion calendar). Its group "
            "columns are matched with group_by by name."
        ),
    )
    calendar_start_column: str = Field(default="start", description="The calendar's first-day column.")
    calendar_end_column: str = Field(default="end", description="The calendar's last-day column.")
    calendar_name_column: str | None = Field(default=None, description="The calendar's name column, if any.")
    include_series: bool = Field(
        default=False,
        description="Return each period around each promotion, actual against expected, for a chart.",
    )

    @model_validator(mode="after")
    def _one_source(self):
        given = [x is not None for x in (self.windows, self.promo_column, self.calendar_dataframe)]
        if sum(given) != 1:
            raise ValueError("Give the promotions one way: windows, promo_column, or calendar_dataframe.")
        return self


class PromotionLiftResult(BaseModel):
    summary: str
    frequency: str
    rows: list[analysis.LiftRow] = Field(description="One per promotion per group, real changes first.")
    groups_found: int
    notes: list[str] = Field(default_factory=list)


@registry.register(
    category="workflows",
    name="promotion_lift",
    input_schema=PromotionLiftInput,
    description=(
        "Measure what each promotion added: sales during it against what would have "
        "sold without it, with a likely range and a verdict, as the Stats Compass app "
        "does. Promotions from a list, a flag column, or a promotion calendar. Daily "
        "or weekly rows, optionally per group."
    ),
    tier="workflow",
)
def promotion_lift(state: DataFrameState, params: PromotionLiftInput) -> PromotionLiftResult:
    df = state.get_dataframe(params.dataframe_name)
    series = build_series(df, params)
    if params.windows is not None:
        windows = list(params.windows)
    elif params.promo_column is not None:
        windows = windows_from_flags(df, params, params.promo_column, series.freq)
    else:
        windows = windows_from_calendar(
            state.get_dataframe(params.calendar_dataframe), params,
            params.calendar_start_column, params.calendar_end_column, params.calendar_name_column,
        )
    if not windows:
        raise ValueError("No promotions were found to measure.")
    rows = []
    for g in series.groups:
        mine = windows_for(g, windows, series.group_by)
        for w in mine:
            others = [o for o in mine if o is not w]
            rows.append(analysis.lift(g, w, others, series.freq, series.week_label, params.kind, params.currency, params.include_series))
    if not rows:
        raise ValueError("None of the promotions matched a group in the data.")
    rows.sort(key=lambda r: (analysis.ORDER[r.verdict], -abs(r.estimate_pct or 0)))
    return PromotionLiftResult(
        summary=analysis.overview(rows, "promotions"),
        frequency=FREQUENCY[series.freq],
        rows=rows,
        groups_found=series.n_groups_found,
        notes=_notes(series),
    )
