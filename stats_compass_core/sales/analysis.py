"""The series statistics on one group's sales, with the Shopify app's settings.

The calibration is the app's (hosted's commerce engine): the Fourier
decomposition, an expected band at 2 standard deviations, a noise band at 2.5,
a 5% test, 90% intervals, a four-week baseline before a promotion, three
weeks after it left out of the fit. Weekly input uses the same settings in
whole weeks.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from stats_compass_core.sales import wording
from stats_compass_core.sales.schema import (
    SERIES_KIND,
    Freq,
    GroupSeries,
    PromotionWindow,
)
from stats_compass_core.series import (
    Insufficient,
    decompose,
    its_lift,
    month_effects,
    verdict,
)
from stats_compass_core.series._common import WeekLabel, weekly_freq, weeks_overlapping

METHOD = "fourier"
DAILY_PERIODS = (7, 365)
CAL = {"seasonal_band_z": 2.0, "noise_band_z": 2.5, "real_change_alpha": 0.05}
LEVEL = 0.9
# The longest run of missing rows filled in: a week of days, or two weeks.
MAX_GAP_DAYS = {"D": 7, "W": 14}
BASELINE_DAYS = 28
BASELINE_LOOKBACK_DAYS = 84
POST_DAYS = 21
SERIES_DAYS_BEFORE = 28
SERIES_DAYS_AFTER = 21

Outcome = Literal["expected", "noise", "real change", "insufficient"]
ORDER = {"real change": 0, "noise": 1, "expected": 2, "insufficient": 3}


def _decompose_options(freq: Freq, week_label: WeekLabel) -> dict[str, Any]:
    if freq == "W":
        return {"method": METHOD, "freq": "W", "week_label": week_label, "max_gap_days": MAX_GAP_DAYS["W"]}
    return {"method": METHOD, "periods": DAILY_PERIODS, "max_gap_days": MAX_GAP_DAYS["D"]}


def _unit(freq: Freq) -> str:
    return "weeks" if freq == "W" else "days"


def _history_length(values: pd.Series, freq: Freq) -> int:
    present = values.dropna()
    if present.empty:
        return 0
    step = 7 if freq == "W" else 1
    return int((present.index[-1] - present.index[0]).days // step) + 1


# =============================================================================
# A period's verdict
# =============================================================================


class PeriodRow(BaseModel):
    """One group's verdict on the period."""

    group: str
    verdict: Outcome
    label: str = Field(description="Expected, Real change, Noise or Not enough data.")
    summary: str = Field(description="The verdict in a sentence, in the app's words.")
    detail: str | None = Field(default=None, description="The expected figure and range, and the change vs the previous period.")
    period_start: str
    period_end: str
    observed: float | None = None
    expected: float | None = None
    expected_low: float | None = None
    expected_high: float | None = None
    change_pct: float | None = Field(default=None, description="Percent change vs the previous period of the same length.")
    p_value: float | None = None
    insufficient_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


def default_period(values: list[pd.Series], freq: Freq) -> tuple[date, date]:
    """The latest week: the last weekly row, or the last 7 days of daily rows."""
    last = max(v.dropna().index.max() for v in values if not v.dropna().empty)
    if freq == "W":
        return last.date(), last.date()
    return (last - pd.Timedelta(days=6)).date(), last.date()


def judge_period(
    group: GroupSeries,
    freq: Freq,
    week_label: WeekLabel,
    kind: str,
    period: tuple[date, date],
    currency: str | None,
) -> PeriodRow:
    values = group.values
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    if freq == "W":
        last_row = max(values.index.max(), end + pd.Timedelta(days=7))
        grid = pd.date_range(values.index.min(), last_row, freq=weekly_freq(values.index))
        labels = weeks_overlapping(grid, start, end, week_label)
        if len(labels) == 0:
            raise ValueError(f"No week in the data falls in {wording.span(period[0], period[1])}.")
        period_values = values.reindex(labels)
        # On the full weekly grid up to the week before the period: a group
        # whose rows stop early (a delisted product) has missing weeks there,
        # which are trimmed, not read as data left out.
        history = values.reindex(grid[grid < labels[0]])
        first, last = labels[0], labels[-1]
        min_points = 1
        weekly_period = len(labels) == 1
        prior_name = "week" if len(labels) == 1 else f"{len(labels)} weeks"
    else:
        days = pd.date_range(start, end)
        period_values = values.reindex(days)
        history = values.reindex(pd.date_range(values.index.min(), start - pd.Timedelta(days=1)))
        first, last = days[0], days[-1]
        min_points = min(5, len(days))
        weekly_period = len(days) == 7
        prior_name = "week" if len(days) == 7 else f"{len(days)} days"
    base = PeriodRow(
        group=group.label, verdict="insufficient", label=wording.VERDICT_LABEL["insufficient"], summary="",
        period_start=first.date().isoformat(), period_end=last.date().isoformat(),
    )
    dec = decompose(history, **_decompose_options(freq, week_label))
    facts = dec if isinstance(dec, Insufficient) else verdict(
        dec, period_values, kind=SERIES_KIND[kind], min_points=min_points, **CAL
    )
    if isinstance(facts, Insufficient):
        return base.model_copy(update={
            "summary": wording.insufficient(facts.reason, facts.needs, facts.has, facts.unit, context="period"),
            "insufficient_reason": facts.reason,
        })
    low, high = facts.expected_range
    change_abs, change_pct = facts.change_vs_prior
    return base.model_copy(update={
        "verdict": facts.verdict,
        "label": wording.VERDICT_LABEL[facts.verdict],
        "summary": wording.period_sentence(facts.verdict, facts.observed, facts.expected, kind, currency, weekly_period),
        "detail": wording.period_detail(facts.expected, low, high, change_pct, kind, currency, prior_name),
        "observed": facts.observed,
        "expected": facts.expected,
        "expected_low": low,
        "expected_high": high,
        "change_pct": change_pct,
        "p_value": facts.p_value,
        "warnings": [w.message for w in facts.warnings],
    })


# =============================================================================
# Seasonality
# =============================================================================


class MonthRow(BaseModel):
    month: int
    name: str
    effect_pct: float = Field(description="How far the month runs above (+) or below (−) the year's average, in percent.")
    lower: float
    upper: float
    line: str


class SeasonalityRow(BaseModel):
    """One group's seasonal pattern."""

    group: str
    found: bool
    summary: str
    months: list[MonthRow] = Field(default_factory=list)
    basis: str | None = None
    insufficient_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


def seasonality(group: GroupSeries, freq: Freq, week_label: WeekLabel) -> SeasonalityRow:
    unit = _unit(freq)
    dec = decompose(group.values, **_decompose_options(freq, week_label))
    effects = dec if isinstance(dec, Insufficient) else month_effects(dec, level=LEVEL)
    if isinstance(effects, Insufficient):
        return SeasonalityRow(
            group=group.label, found=False,
            summary=wording.insufficient(effects.reason, effects.needs, effects.has, effects.unit, context="seasonality"),
            insufficient_reason=effects.reason,
        )
    months = [
        MonthRow(month=r.month, name=wording.MONTHS[r.month - 1], effect_pct=r.effect_pct, lower=r.lower, upper=r.upper,
                 line=wording.month_line(r.month, r.effect_pct, r.lower, r.upper))
        for r in effects.rows
    ]
    length = _history_length(group.values, freq)
    return SeasonalityRow(
        group=group.label, found=True,
        summary=wording.seasonality_sentence(effects.rows),
        months=months,
        basis=f"Based on {wording.history(length, unit)} of data. Each range is where the month's effect likely lies (90%).",
        warnings=[w.message for w in effects.warnings],
    )


# =============================================================================
# A promotion's lift
# =============================================================================


class LiftPoint(BaseModel):
    date: str
    actual: float | None
    expected: float
    low: float
    high: float
    in_promotion: bool


class LiftRow(BaseModel):
    """One promotion in one group."""

    group: str
    promotion: str
    start: str
    end: str
    verdict: Outcome
    label: str
    summary: str
    value: str | None = Field(default=None, description="'+32%', the app's headline figure.")
    amount: str | None = Field(default=None, description="'£9,144 more than expected'.")
    range: str | None = Field(default=None, description="'likely between +13% and +51%'; none beside an expected verdict.")
    estimate_pct: float | None = None
    interval_pct: tuple[float, float] | None = None
    estimate: float | None = None
    interval: tuple[float, float] | None = None
    observed_total: float | None = None
    expected_total: float | None = None
    p_value: float | None = None
    periods_in_window: list[str] = Field(default_factory=list, description="The weeks (or days) counted as the promotion.")
    series: list[LiftPoint] | None = Field(default=None, description="Each period around it, actual against expected, for a chart.")
    insufficient_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


def lift(
    group: GroupSeries,
    window: PromotionWindow,
    others: list[PromotionWindow],
    freq: Freq,
    week_label: WeekLabel,
    kind: str,
    currency: str | None,
    include_series: bool,
) -> LiftRow:
    name = window.name or "The promotion"
    base = LiftRow(
        group=group.label, promotion=name, start=window.start.isoformat(), end=window.end.isoformat(),
        verdict="insufficient", label=wording.VERDICT_LABEL["insufficient"], summary="",
    )
    options = _decompose_options(freq, week_label)
    method = options.pop("method")
    facts = its_lift(
        group.values,
        (window.start, window.end),
        kind=SERIES_KIND[kind],
        method=method,
        other_windows=[(w.start, w.end) for w in others],
        noise_band="none",
        post_days=POST_DAYS,
        baseline_days=BASELINE_DAYS,
        baseline_lookback_days=BASELINE_LOOKBACK_DAYS,
        level=LEVEL,
        series_days_before=SERIES_DAYS_BEFORE if include_series else None,
        series_days_after=SERIES_DAYS_AFTER if include_series else None,
        **CAL,
        **options,
    )
    if isinstance(facts, Insufficient):
        reason = wording.insufficient(
            facts.reason, facts.needs, facts.has, facts.unit, context="promotion",
            weeks_of_history=_history_length(group.values, freq),
        )
        return base.model_copy(update={
            "summary": f"{name}, {wording.span(window.start, window.end)}: {reason}",
            "insufficient_reason": facts.reason,
        })
    value = wording.lift_value(facts.estimate_pct, facts.estimate_abs, kind, currency)
    amount_text = wording.lift_amount(facts.estimate_abs, kind, currency)
    range_text = wording.lift_range(facts.verdict, facts.interval_pct, facts.interval_abs, kind, currency)
    periods = facts.params.get("window_weeks") or [
        d.date().isoformat() for d in pd.date_range(window.start, window.end)
    ]
    series = None
    if facts.series is not None:
        series = [
            LiftPoint(date=p.date.isoformat(), actual=p.actual, expected=p.expected, low=p.low, high=p.high, in_promotion=p.in_window)
            for p in facts.series
        ]
    return base.model_copy(update={
        "verdict": facts.verdict,
        "label": wording.VERDICT_LABEL[facts.verdict],
        "summary": wording.lift_sentence(name, window.start, window.end, value, amount_text, range_text, facts.verdict),
        "value": value,
        "amount": amount_text,
        "range": range_text,
        "estimate_pct": facts.estimate_pct,
        "interval_pct": facts.interval_pct,
        "estimate": facts.estimate_abs,
        "interval": facts.interval_abs,
        "observed_total": facts.observed_total,
        "expected_total": facts.counterfactual_total,
        "p_value": facts.p_value,
        "periods_in_window": periods,
        "series": series,
        "warnings": [w.message for w in facts.warnings],
    })


def overview(rows: list[Any], what: str) -> str:
    """"2 of 12 groups had a real change." for many groups; the row's own sentence for one."""
    if len(rows) == 1:
        return rows[0].summary
    counts = {k: sum(1 for r in rows if r.verdict == k) for k in ORDER}
    parts = [f"{counts['real change']} of {len(rows)} {what} had a real change"]
    if counts["insufficient"]:
        parts.append(f"{counts['insufficient']} had too little data to judge")
    return "; ".join(parts) + "."
