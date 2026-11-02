"""The series statistics on one group's sales, with the Shopify app's settings.

The calibration is the app's (hosted's commerce engine): the Fourier
decomposition, an expected band at 2 standard deviations, a noise band at 2.5,
a 5% test, 90% intervals, a four-week baseline before a promotion, three
weeks after it left out of the fit. Weekly input uses the same settings in
whole weeks.
"""

from __future__ import annotations

import math
import re
from datetime import date, timedelta
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
from stats_compass_core.series._common import WeekLabel, week_days, weekly_freq, weeks_overlapping

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


def _exclusions(windows: list[PromotionWindow] | None) -> list[tuple[date, date]]:
    """Promotions to leave out of a fit: each, and the 3 weeks after it (as the app does)."""
    return [(w.start, w.end + timedelta(days=POST_DAYS)) for w in windows or []]


def _warnings(values: pd.Series, freq: Freq, *sources) -> list[str]:
    """Every warning, once, in order (the decomposition's, then the statistic's), said plainly."""
    seen: list[str] = []
    gaps = None
    for source in sources:
        for w in getattr(source, "warnings", None) or []:
            leading = re.match(r"\s*(\d+)", w.message)
            if gaps is None:
                gaps = _weekend_share(values, freq)
            text = wording.plain_warning(
                w.code, w.message, int(leading.group(1)) if leading else None,
                MAX_GAP_DAYS[freq] // (7 if freq == "W" else 1), gaps,
            )
            if text not in seen:
                seen.append(text)
    return seen


def _weekend_share(values: pd.Series, freq: Freq) -> float:
    """Of the days with no row between the first and the last, the share on a Saturday or Sunday."""
    if freq != "D":
        return 0.0
    present = values.dropna().index
    if len(present) < 2:
        return 0.0
    missing = pd.date_range(present.min(), present.max()).difference(present)
    return float((missing.dayofweek >= 5).mean()) if len(missing) else 0.0


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
    promotions_in_period: list[str] = Field(default_factory=list, description="Promotions that ran during the period, if any were given.")
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
    promotions: list[PromotionWindow] | None = None,
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
        period_days = (week_days(first, week_label)[0], week_days(last, week_label)[1])
    else:
        days = pd.date_range(start, end)
        period_values = values.reindex(days)
        history = values.reindex(pd.date_range(values.index.min(), start - pd.Timedelta(days=1)))
        first, last = days[0], days[-1]
        min_points = min(5, len(days))
        weekly_period = len(days) == 7
        prior_name = "week" if len(days) == 7 else f"{len(days)} days"
        period_days = (first, last)
    base = PeriodRow(
        group=group.label, verdict="insufficient", label=wording.VERDICT_LABEL["insufficient"], summary="",
        period_start=first.date().isoformat(), period_end=last.date().isoformat(),
    )
    running = [
        w.name or "a promotion" for w in promotions or []
        if pd.Timestamp(w.start) <= period_days[1] and pd.Timestamp(w.end) >= period_days[0]
    ]
    base = base.model_copy(update={"promotions_in_period": running})
    dec = decompose(history, exclude=_exclusions(promotions), **_decompose_options(freq, week_label))
    facts = dec if isinstance(dec, Insufficient) else verdict(
        dec, period_values, kind=SERIES_KIND[kind], min_points=min_points, **CAL
    )
    if isinstance(facts, Insufficient):
        rows_before = values.dropna()
        rows_before = rows_before[rows_before.index < first]
        if facts.reason == "TOO_FEW_POINTS" and facts.has == 0 and len(rows_before):
            # A product whose rows stopped before the period: delisted, say.
            summary = wording.no_rows_since(rows_before.index[-1].date(), freq == "W", week_label)
        else:
            summary = wording.insufficient(
                facts.reason, facts.needs, facts.has, facts.unit, context="period",
                weeks_of_history=_history_length(values[values.index <= last], freq) if freq == "W" else None,
            )
        return base.model_copy(update={
            "summary": summary,
            "insufficient_reason": facts.reason,
            # A product with no rows since says so; its trimmed weeks needn't be listed too.
            "warnings": [] if summary.startswith("There are no sales") else _warnings(values, freq, dec),
        })
    low, high = facts.expected_range
    change_abs, change_pct = facts.change_vs_prior
    detail = wording.period_detail(facts.expected, low, high, change_pct, kind, currency, prior_name)
    if running:
        detail += " " + wording.promotions_in_period(running, weekly_period)
    return base.model_copy(update={
        "verdict": facts.verdict,
        "label": wording.VERDICT_LABEL[facts.verdict],
        "summary": wording.period_sentence(facts.verdict, facts.observed, facts.expected, kind, currency, weekly_period),
        "detail": detail,
        "observed": facts.observed,
        "expected": facts.expected,
        "expected_low": low,
        "expected_high": high,
        "change_pct": change_pct,
        "p_value": facts.p_value,
        "warnings": _warnings(values, freq, dec, facts),
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


def seasonality(
    group: GroupSeries, freq: Freq, week_label: WeekLabel, promotions: list[PromotionWindow] | None = None
) -> SeasonalityRow:
    """Each month's effect. With promotions given, their weeks (and the 3 after) are
    left out, so the months show the season, not this year's Black Friday."""
    unit = _unit(freq)
    dec = decompose(group.values, exclude=_exclusions(promotions), **_decompose_options(freq, week_label))
    effects = dec if isinstance(dec, Insufficient) else month_effects(dec, level=LEVEL)
    if isinstance(effects, Insufficient):
        return SeasonalityRow(
            group=group.label, found=False,
            summary=wording.insufficient(effects.reason, effects.needs, effects.has, effects.unit, context="seasonality"),
            insufficient_reason=effects.reason,
            warnings=_warnings(group.values, freq, dec),
        )
    months = [
        MonthRow(month=r.month, name=wording.MONTHS[r.month - 1], effect_pct=r.effect_pct, lower=r.lower, upper=r.upper,
                 line=wording.month_line(r.month, r.effect_pct, r.lower, r.upper))
        for r in effects.rows
    ]
    length = _history_length(group.values, freq)
    basis = f"Based on {wording.history(length, unit)} of data. Each range is where the month's effect likely lies (90%)."
    if dec.excluded:
        basis += " " + (wording.PROMOTIONS_LEFT_OUT if freq == "W" else wording.PROMOTION_DAYS_LEFT_OUT)
    return SeasonalityRow(
        group=group.label, found=True,
        summary=wording.seasonality_sentence(effects.rows),
        months=months,
        basis=basis,
        warnings=_warnings(group.values, freq, dec, effects),
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
    interval: tuple[float, float] | None = Field(
        default=None,
        description=(
            "The likely range as an amount. When there's a percentage range it's that range "
            "applied to the expected total, so the two always agree with each other and the verdict."
        ),
    )
    observed_total: float | None = None
    expected_total: float | None = None
    p_value: float | None = None
    periods_in_window: list[str] = Field(default_factory=list, description="The weeks (or days) counted as the promotion.")
    series: list[LiftPoint] | None = Field(default=None, description="Each period around it, actual against expected, for a chart.")
    after_summary: str | None = Field(
        default=None, description="The 3 weeks (or 21 days) just after it, against expected: a dip if customers stocked up."
    )
    after_pct: float | None = Field(default=None, description="Sales in the weeks after, in percent vs expected.")
    after_difference: float | None = Field(default=None, description="Sales in the weeks after, minus expected.")
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
        # The weeks after are always read, for the pull-forward check.
        series_days_before=SERIES_DAYS_BEFORE if include_series else 0,
        series_days_after=SERIES_DAYS_AFTER,
        **CAL,
        **options,
    )
    if isinstance(facts, Insufficient):
        reason_code = facts.reason
        present = group.values.dropna()
        first_year = len(present) and pd.Timestamp(window.start) < present.index[0] + pd.Timedelta(days=365 + BASELINE_DAYS)
        if reason_code == "BASELINE_TOO_SHORT" and first_year:
            # The baseline is short because nothing a year back gives its season.
            reason_code = "NO_SEASONAL_BEFORE"
        reason = wording.insufficient(
            reason_code, facts.needs, facts.has, facts.unit, context="promotion",
            weeks_of_history=_history_length(group.values, freq),
        )
        return base.model_copy(update={
            "summary": f"{name}, {wording.span(window.start, window.end)}: {reason}",
            "insufficient_reason": reason_code,
        })
    # One basis for both ranges. The verdict and the percentage range rank the
    # past errors as percentages of what was expected then; the series' amount
    # range ranks them as raw amounts, which on a series that has grown (or
    # shrunk) understates (or overstates) today's uncertainty. So the amount
    # range shown is the percentage range applied to today's expected total.
    interval = facts.interval_abs
    if facts.interval_pct is not None and facts.counterfactual_total > 0:
        interval = (
            facts.counterfactual_total * facts.interval_pct[0] / 100,
            facts.counterfactual_total * facts.interval_pct[1] / 100,
        )
    value = wording.lift_value(facts.estimate_pct, facts.estimate_abs, kind, currency)
    amount_text = wording.lift_amount(facts.estimate_abs, kind, currency)
    range_text = wording.lift_range(facts.verdict, facts.interval_pct, interval, kind, currency)
    periods = facts.params.get("window_weeks") or [
        d.date().isoformat() for d in pd.date_range(window.start, window.end)
    ]
    points = [
        LiftPoint(date=p.date.isoformat(), actual=p.actual, expected=p.expected, low=p.low, high=p.high, in_promotion=p.in_window)
        for p in facts.series or []
    ]
    after = _after(points, periods[-1], facts.params.get("series_band_half_width"), freq, kind, currency)
    summary = wording.lift_sentence(name, window.start, window.end, value, amount_text, range_text, facts.verdict)
    if after and not after["summary"].endswith("about as expected."):
        summary += " " + after["summary"]
    return base.model_copy(update={
        "verdict": facts.verdict,
        "label": wording.VERDICT_LABEL[facts.verdict],
        "summary": summary,
        "after_summary": after["summary"] if after else None,
        "after_pct": after["pct"] if after else None,
        "after_difference": after["difference"] if after else None,
        "value": value,
        "amount": amount_text,
        "range": range_text,
        "estimate_pct": facts.estimate_pct,
        "interval_pct": facts.interval_pct,
        "estimate": facts.estimate_abs,
        "interval": interval,
        "observed_total": facts.observed_total,
        "expected_total": facts.counterfactual_total,
        "p_value": facts.p_value,
        "periods_in_window": periods,
        "series": points if include_series else None,
        "warnings": _warnings(group.values, freq, facts),
    })


def _after(points: list[LiftPoint], last_in_window: str, half: float | None, freq: Freq, kind: str, currency: str | None) -> dict | None:
    """The weeks just after a promotion against expected, from the lift's own series.

    The expected line is the lift's counterfactual, carried on. A difference
    counts as a dip (or a rise) only beyond the range those weeks usually
    vary by: one step's 90% half-width times the square root of the steps.
    """
    # The 3 weeks after, by the calendar: days without a row (a closed
    # weekend) are left out of the sums, not counted as part of the span.
    end = (pd.Timestamp(last_in_window) + pd.Timedelta(days=POST_DAYS)).date().isoformat()
    after = [p for p in points if last_in_window < p.date <= end and not p.in_promotion and p.actual is not None]
    if not after:
        return None
    actual = sum(p.actual for p in after)
    expected = sum(p.expected for p in after)
    difference = actual - expected
    pct = 100 * difference / expected if expected > 0 else None
    beyond = half is not None and abs(difference) > half * math.sqrt(len(after))
    if freq == "W":
        steps, unit = len(after), "weeks"
    else:
        days = (pd.Timestamp(after[-1].date) - pd.Timestamp(last_in_window)).days
        steps, unit = (POST_DAYS // 7, "weeks") if days > 14 else (days, "days")
    return {
        "summary": wording.after_promotion(pct, difference, steps, unit, beyond, kind, currency),
        "pct": pct,
        "difference": difference,
    }


def overview(rows: list[Any], what: str) -> str:
    """"2 of 12 groups had a real change." for many groups; the row's own sentence for one."""
    if len(rows) == 1:
        return rows[0].summary
    counts = {k: sum(1 for r in rows if r.verdict == k) for k in ORDER}
    parts = [f"{counts['real change']} of {len(rows)} {what} had a real change"]
    if counts["insufficient"]:
        parts.append(f"{counts['insufficient']} had too little data to judge")
    return "; ".join(parts) + "."
