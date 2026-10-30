"""Every sentence the sales tools write, in the Shopify app's words.

The app's wording is the model (stats-compass-commerce): the verdict labels
from ``packages/ui/src/badge.ts``, the period sentences from
``packages/ui/src/headline.ts``, the lift lines from
``apps/web/src/promotions.ts``, date ranges from ``packages/ui/src/dates.ts``,
and the seasonality chart's "the year's average". The copy rules: "expected"
is the baseline word, contractions, a true minus (−), numerals ("2 years"),
no em dashes. Figures come from the
statistics; nothing here computes one.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

MINUS = "−"

VERDICT_LABEL = {
    "expected": "Expected",
    "real change": "Real change",
    "noise": "Noise",
    "insufficient": "Not enough data",
}

# Weekly input, from the calibration (scripts/calibrate_weekly.py): a week is
# first judged at 77 weeks of history, four weeks or a promotion at 83 to 93,
# and 2 years and 3 found about as much.
# A week first gets a verdict with this many weeks of data (the history and
# the week itself): the weekly calibration's floor.
WEEKLY_VERDICT_FLOOR = 77

WEEKLY_HISTORY_NOTE = (
    "Weekly data needs about 18 months of history before a week can be judged, "
    "and works best with 2 years or more."
)

MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


# -- numbers ------------------------------------------------------------------


def number(x: float) -> str:
    """12,345 for large figures, 12.34 for small ones; a true minus."""
    text = f"{abs(x):,.0f}" if abs(x) >= 100 else f"{abs(x):,.2f}"
    return f"{MINUS}{text}" if x < 0 and text.strip("0.,") else text


def amount(x: float, kind: str, currency: str | None) -> str:
    """A figure in its units: "£12,345", "12,345", or "1,234 units"."""
    if kind == "units":
        return f"{number(x)} unit{'' if abs(x) == 1 else 's'}"
    text = number(abs(x))
    sign = MINUS if x < 0 and text.strip("0.,") else ""
    return f"{sign}{currency or ''}{text}"


def signed_pct(x: float) -> str:
    """"+32%", "−25%", "0%"."""
    r = round(x)
    if r == 0:
        return "0%"
    return f"+{r}%" if r > 0 else f"{MINUS}{abs(r)}%"


def signed_amount(x: float, kind: str, currency: str | None) -> str:
    """"+£2,031", "−£1,377", "£0"."""
    text = amount(abs(x), kind, currency)
    if not number(abs(x)).strip("0.,"):
        return text
    return f"+{text}" if x > 0 else f"{MINUS}{text}"


def day(d: date) -> str:
    """"14 May 2026"."""
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def span(first: date, last: date) -> str:
    """As the app writes them: "21–27 September 2026", "14 May – 3 June 2026",
    "28 December 2025 – 3 January 2026": an en dash, spaced across months."""
    if first == last:
        return day(first)
    if first.year == last.year and first.month == last.month:
        return f"{first.day}–{day(last)}"
    if first.year == last.year:
        head = f"{first.day} {MONTHS[first.month - 1]}"
        return f"{head} – {day(last)}"
    return f"{day(first)} – {day(last)}"


def period_name(first: date, last: date, weekly: bool, week_label: str) -> str:
    """"Week ending 24 October 2026", "Weeks ending 3–24 October 2026", "18–24 October 2026"."""
    if not weekly:
        return span(first, last)
    which = "ending" if week_label == "end" else "starting"
    if first == last:
        return f"Week {which} {day(first)}"
    return f"Weeks {which} {span(first, last)}"


def history(n: int, unit: str) -> str:
    """"61 weeks", "about 2 years"."""
    if unit == "weeks" and n >= 104:
        return f"about {round(n / 52)} years"
    if unit == "days" and n >= 730:
        return f"about {round(n / 365)} years"
    return f"{n} {unit[:-1] if n == 1 else unit}"


# -- a period's verdict ---------------------------------------------------------


def period_sentence(
    verdict: str,
    observed: float,
    expected: float,
    kind: str,
    currency: str | None,
    weekly_period: bool,
) -> str:
    """The app's headline sentence, for any sales figure.

    A real change says which way and how far, as the app's finding titles do.
    """
    what = f"sales of {amount(observed, kind, currency)}"
    movement = "week-to-week movement" if weekly_period else "movement from one period to the next"
    if verdict == "expected":
        return f"Nothing here looks unusual: {what} are within the expected range for this time of year."
    if verdict == "noise":
        return f"Nothing here looks unusual: {what} are within the expected {movement}."
    side = "above" if observed >= expected else "below"
    if expected > 0:
        size = f"{abs(round(100 * (observed - expected) / expected))}%"
    else:
        size = amount(abs(observed - expected), kind, currency)
    return (
        f"{what[0].upper()}{what[1:]} are {size} {side} what this time of year usually brings: "
        f"more than the usual {movement}, and unlikely to be chance."
    )


def period_detail(
    expected: float,
    low: float,
    high: float,
    change_pct: float | None,
    kind: str,
    currency: str | None,
    prior_name: str,
) -> str:
    """"Expected about £11,900, in a range of £10,800 to £13,000. +4% vs the previous week."."""
    text = (
        f"Expected about {amount(expected, kind, currency)}, in a range of "
        f"{amount(max(low, 0.0), kind, currency)} to {amount(high, kind, currency)}."
    )
    if change_pct is not None:
        text += f" {signed_pct(change_pct)} vs the previous {prior_name}."
    return text


# -- a promotion's lift ----------------------------------------------------------


def lift_value(pct: float | None, estimate: float, kind: str, currency: str | None) -> str:
    """"+32%", or the change itself when there's no percentage."""
    return signed_pct(pct) if pct is not None else signed_amount(estimate, kind, currency)


def lift_amount(estimate: float, kind: str, currency: str | None) -> str:
    """"£9,144 more than expected", "1,234 fewer units than expected"."""
    if kind == "units":
        direction = "more" if estimate >= 0 else "fewer"
        n = number(abs(estimate))
        return f"{n} {direction} unit{'' if abs(estimate) == 1 else 's'} than expected"
    direction = "more" if estimate >= 0 else "less"
    return f"{amount(abs(estimate), kind, currency)} {direction} than expected"


def lift_range(
    verdict: str,
    interval_pct: tuple[float, float] | None,
    interval_abs: tuple[float, float],
    kind: str,
    currency: str | None,
) -> str | None:
    """"likely between +13% and +51%"; None beside an expected verdict, as in the app."""
    if verdict == "expected":
        return None
    if interval_pct is not None:
        return f"likely between {signed_pct(interval_pct[0])} and {signed_pct(interval_pct[1])}"
    return (
        f"likely between {signed_amount(interval_abs[0], kind, currency)} and "
        f"{signed_amount(interval_abs[1], kind, currency)}"
    )


def lift_sentence(name: str, first: date, last: date, value: str, amount_text: str, range_text: str | None, verdict: str) -> str:
    """"SUMMER40, 14 May – 3 June 2026: +32%, £9,144 more than expected; likely between +13% and +51%. Real change."."""
    middle = f"{value}, {amount_text}" + (f"; {range_text}" if range_text else "")
    return f"{name}, {span(first, last)}: {middle}. {VERDICT_LABEL[verdict]}."


# -- seasonality -------------------------------------------------------------------


def month_line(month: int, effect: float, lower: float, upper: float) -> str:
    """"December: about 33% above the year's average, likely between +25% and +41%."."""
    name = MONTHS[month - 1]
    if lower <= 0 <= upper:
        return f"{name}: about the same as the year's average ({signed_pct(effect)}, likely between {signed_pct(lower)} and {signed_pct(upper)})."
    side = "above" if effect > 0 else "below"
    return (
        f"{name}: about {abs(round(effect))}% {side} the year's average, likely between "
        f"{signed_pct(lower)} and {signed_pct(upper)}."
    )


def seasonality_sentence(rows: list[Any]) -> str:
    """The highest and lowest months, where the range clears zero."""
    clear = [r for r in rows if not (r.lower <= 0 <= r.upper)]
    if not clear:
        return "No month runs clearly above or below the year's average: there's no seasonal pattern to speak of."
    high = max(clear, key=lambda r: r.effect_pct)
    low = min(clear, key=lambda r: r.effect_pct)
    parts = []
    if high.effect_pct > 0:
        parts.append(f"highest in {MONTHS[high.month - 1]} (about {round(high.effect_pct)}% above the year's average)")
    if low.effect_pct < 0:
        parts.append(f"lowest in {MONTHS[low.month - 1]} (about {abs(round(low.effect_pct))}% below)")
    return "Sales usually run " + " and ".join(parts) + "."


# -- not enough data ----------------------------------------------------------------


def insufficient(reason: str, needs: Any, has: Any, unit: str, *, context: str, weeks_of_history: int | None = None) -> str:
    """Why there's no verdict, and what it needs, in plain words.

    ``context`` is "period", "promotion" or "seasonality". For a weekly
    period, ``weeks_of_history`` is the weeks of data including the period:
    any shortage of history is then said as the calibrated floor, when a week
    is first judged, rather than the fit's own 52 weeks.
    """
    u = unit
    if reason == "TOO_SHORT" and context == "period" and u == "weeks" and weeks_of_history is not None:
        return (
            f"Too early to say what to expect: a week is first judged once there are about "
            f"{WEEKLY_VERDICT_FLOOR} weeks of data; there are {history(weeks_of_history, u)}."
        )
    if reason == "TOO_SHORT" and context == "promotion" and weeks_of_history is not None and needs is not None and has is not None:
        # The lift's TOO_SHORT counts past positions to compare with; say it as history.
        more = max(int(needs) - int(has), 1)
        text = (
            f"Not enough history yet to measure this promotion against: it needs about "
            f"{history(weeks_of_history + more, u)} of data; there are {history(weeks_of_history, u)}."
        )
    elif reason == "TOO_SHORT":
        what = {"period": "say what to expect", "promotion": "measure this promotion", "seasonality": "find the seasonal pattern"}[context]
        need = history(int(needs), u)
        floor = need if need.startswith("about") else f"at least {need}"
        text = f"Too early to {what}: it needs {floor} of history; there are {history(int(has), u)}."
    elif reason == "GAP_TOO_LONG":
        text = f"There's a gap of {history(int(has), u)} with no data. Gaps of up to {history(int(needs), u)} are filled in; longer ones aren't."
    elif reason == "TOO_FEW_POINTS":
        whose = "The promotion" if context == "promotion" else "The period"
        text = f"{whose} has data for {history(int(has), u)}; it needs at least {history(int(needs), u)}."
    elif reason == "BASELINE_TOO_SHORT":
        text = (
            f"There isn't enough data just before the promotion to compare it with: it needs "
            f"{history(int(needs), u)} without another promotion; there are {history(int(has), u)}."
        )
    elif reason == "NO_SEASONAL_BEFORE":
        text = "The promotion needs a year of data before it, to know what that time of year usually brings."
    elif reason == "MONTH_TOO_SHORT":
        text = f"Every month needs at least {needs} days of data to compare; the shortest has {has}."
    elif reason == "NONPOSITIVE_LEVEL":
        text = "The underlying level is zero or below in some month, so a percentage of it means nothing."
    elif reason == "NO_ANNUAL":
        text = f"The seasonal pattern needs about 2 years of daily data; there are {history(int(has), u)}."
    elif reason == "RECENT_DAYS_FILLED":
        text = f"Too few recent {u} have real data to read the current level from: it needs {needs}; there are {has}."
    elif reason == "NO_DATA":
        text = "There's no data."
    elif reason == "HISTORY_ENDS_EARLY":
        text = "The history stops before the period starts, leaving out data in between."
    else:
        text = f"Not enough data ({reason})."
    if u == "weeks" and context != "seasonality" and reason in ("TOO_SHORT", "BASELINE_TOO_SHORT", "NO_SEASONAL_BEFORE"):
        text += " " + WEEKLY_HISTORY_NOTE
    return text


def no_rows_since(last: date, weekly: bool, week_label: str) -> str:
    """For a group whose rows stopped before the period: a delisted product."""
    when = f"the week {'ending' if week_label == 'end' else 'starting'} {day(last)}" if weekly else day(last)
    return f"There are no sales in this period: the last row is for {when}."


def promotions_in_period(names: list[str], weekly_period: bool) -> str:
    """"This week had a promotion: Prime Day 2026."."""
    what = "This week" if weekly_period else "The period"
    if len(names) == 1:
        return f"{what} had a promotion: {names[0]}."
    return f"{what} had promotions: {', '.join(names[:-1])} and {names[-1]}."


def after_promotion(pct: float | None, difference: float, steps: int, unit: str, beyond: bool, kind: str, currency: str | None) -> str:
    """What happened in the weeks just after a promotion: a dip, if customers stocked up.

    ``beyond`` says whether the difference is larger than those weeks usually
    vary; only then is a reason offered.
    """
    span_text = f"the {steps} {unit[:-1] if steps == 1 else unit} after"
    if not beyond:
        return f"In {span_text}, sales were about as expected."
    size = f"{abs(round(pct))}% " if pct is not None else ""
    if difference < 0:
        return (
            f"In {span_text}, sales ran {size}below expected ({amount(abs(difference), kind, currency)} less): "
            f"customers may have stocked up."
        )
    return f"In {span_text}, sales stayed {size}above expected ({amount(difference, kind, currency)} more)."


PROMOTIONS_LEFT_OUT = "Promotion weeks, and the 3 weeks after each, were left out of what's expected."
PROMOTION_DAYS_LEFT_OUT = "Promotion days, and the 21 days after each, were left out of what's expected."


# The weekly calibration's highest false-alarm rate for a week's verdict
# (docs/calibration-weekly.md): what chance alone gives across many products.
WEEKLY_FALSE_ALARM_RATE = 0.03


def chance_note(judged: int, real_changes: int) -> str | None:
    """With many products judged, a real change or two is expected by chance alone."""
    expected = round(judged * WEEKLY_FALSE_ALARM_RATE)
    if judged < 10 or expected < 1 or real_changes == 0:
        return None
    return (
        f"With {judged} groups judged, about {expected} could show a real change by chance alone: "
        f"look for a cause before acting on one."
    )


def plain_warning(code: str, message: str, n: int | None, max_gap: int, weekend_share: float) -> str:
    """A series warning, said plainly. The series' own wording is pinned (the app
    reads it), so it's reworded here, by its code, rather than changed there."""
    def count(k: int, unit: str) -> str:
        return f"{k:,} {unit[:-1] if k == 1 else unit}"

    if code in ("IMPUTED_DAYS", "IMPUTED_WEEKS") and n is not None:
        unit = "days" if code == "IMPUTED_DAYS" else "weeks"
        verb = "had no data and was" if n == 1 else "had no data and were"
        text = (
            f"{count(n, unit)} {verb} filled in from the {unit} either side "
            f"(gaps of up to {count(max_gap, unit)} are filled)."
        )
        if unit == "days" and weekend_share >= 0.6:
            text += (
                " Most are Saturdays and Sundays: if nothing sells on those days, give them as 0, "
                "since filling them in adds sales that didn't happen."
            )
        return text
    if code in ("TRIMMED_DAYS", "TRIMMED_WEEKS") and n is not None:
        unit = "days" if code == "TRIMMED_DAYS" else "weeks"
        return f"{count(n, unit)} with no data before the first row or after the last {'was' if n == 1 else 'were'} left out."
    if code == "ANNUAL_DROPPED":
        return (
            "There isn't enough history to allow for the time of year: what's expected follows "
            "the recent level only, so its range is wider."
        )
    if code == "RECENT_DAYS_FILLED":
        return "Few recent days have real data to read the current level from, so what's expected is less certain."
    return message


def money_parsed(column: str, example: str) -> str:
    """"Money written as text in 'Shipped COGS' (for example '£1,234.56') was read as numbers."."""
    return f"Money written as text in '{column}' (for example '{example}') was read as numbers."


def is_finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
