"""What a sales dataset looks like to the sales tools: the one place it is defined.

Everything about the data's shape lives here, so that a change to it (which
columns mean what, how a promotion is given, how a week is labelled) is one
edit:

- **Column roles.** A date, a value (sales in money, or units), and optionally
  the columns that split it into separate series, such as retailer and
  product. Rows are summed to one value per date within each group: sell-out
  data has many rows per date.
- **Frequency.** Daily, or weekly (every date on the same weekday), detected
  from the dates. Monthly or irregular data is refused with a plain reason.
- **Week labelling.** A weekly row's date is the week ending (usual for
  retailer EPOS data) or the week starting.
- **Promotions.** A list of windows (name, first day, last day, and which
  group), a flag column whose consecutive flagged dates become windows, or a
  promotion calendar (a second table with start and end columns, and the
  group columns by the same names). A group value of "all", "*" or blank
  means every group. All three tools take promotions the same way: the lift
  measures them, the verdict and seasonality leave them out of the fit.
- **Money as text.** A value column written as text ("£1,234.56",
  "1,234.56", "(£12.00)" for a negative, "-" for none) is read as numbers.

The expected shape for consumer-goods brands, still to be confirmed: weekly
rows by week ending × retailer × product, with units and sales value, and a
promotion calendar (retailer, product, start, end, mechanic, trade spend).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.series._common import WeekLabel, weekly_freq

Kind = Literal["money", "units"]
Freq = Literal["D", "W"]

# The most separate series one call judges. Retailer × product data can hold
# hundreds; the largest by total value are judged, and the rest named.
MAX_GROUPS = 50

# What the series statistics call each kind.
SERIES_KIND = {"money": "money", "units": "count"}


class SalesColumns(StrictToolInput):
    """The column roles every sales tool shares."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of the DataFrame holding the sales rows. Uses the active one if not given.",
    )
    date_column: str = Field(
        description=(
            "The date of each row: a day, or for weekly data the week's date "
            "(see week_label). Rows may repeat a date; they are summed."
        )
    )
    value_column: str = Field(
        description="The figure to judge: sales value (kind='money') or units sold (kind='units')."
    )
    kind: Kind = Field(
        default="money",
        description="'money' for sales value, 'units' for units sold. Chooses the test and the wording.",
    )
    group_by: list[str] | None = Field(
        default=None,
        description=(
            "Columns to judge separately, for example ['retailer', 'product']. "
            f"Rows are summed within each group. At most {MAX_GROUPS} groups are "
            "judged, the largest by total value; filter first to judge others."
        ),
    )
    week_label: WeekLabel = Field(
        default="end",
        description=(
            "For weekly data, which day each date names: 'end' (the week ending, "
            "usual for retailer sell-out data) or 'start'. Ignored for daily data."
        ),
    )
    currency: str | None = Field(
        default=None,
        description=(
            "A currency symbol for the wording, such as '£'. A CSV carries none, so "
            "money is written as plain numbers unless this is given."
        ),
    )


# A group value meaning "every group": a calendar's Prime Day row with ASIN "all".
WILDCARDS = {"", "all", "*", "nan", "none"}


def is_wildcard(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip().lower() in WILDCARDS


class PromotionWindow(BaseModel):
    """One promotion: its days, and which group it ran in."""

    name: str | None = Field(default=None, description="What to call it, such as 'Summer 2 for 1'.")
    start: date = Field(description="Its first day.")
    end: date = Field(description="Its last day.")
    group: dict[str, str | None] | None = Field(
        default=None,
        description=(
            "Which group it ran in, by the group_by columns, for example "
            "{'retailer': 'Tesco'}. Omitted, or 'all': every group."
        ),
    )


class PromotionColumns(StrictToolInput):
    """How the promotions are given: the same three ways for every sales tool."""

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
            "columns are matched with group_by by name; 'all', '*' or blank means every group."
        ),
    )
    calendar_start_column: str = Field(default="start", description="The calendar's first-day column.")
    calendar_end_column: str = Field(default="end", description="The calendar's last-day column.")
    calendar_name_column: str | None = Field(default=None, description="The calendar's name column, if any.")

    def promotion_sources(self) -> int:
        return sum(x is not None for x in (self.windows, self.promo_column, self.calendar_dataframe))


@dataclass
class GroupSeries:
    """One group's values, summed to one per date."""

    key: tuple
    label: str
    values: pd.Series
    total: float


@dataclass
class SalesSeries:
    """The dataset as separate series, ready for the statistics."""

    freq: Freq
    groups: list[GroupSeries]
    n_groups_found: int
    group_by: list[str] = field(default_factory=list)
    week_label: WeekLabel = "end"
    # An example of money written as text, when the value column was parsed.
    parsed_text: str | None = None


def build_series(df: pd.DataFrame, cols: SalesColumns) -> SalesSeries:
    """Split the rows into groups, sum each to one value per date, and detect the frequency.

    Raises ValueError, in plain words, for a missing column, dates or values
    that do not parse, or dates that are neither daily nor weekly.
    """
    group_by = list(cols.group_by or [])
    needed = [cols.date_column, cols.value_column, *group_by]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(missing_columns(missing, df.columns))
    dates = pd.to_datetime(df[cols.date_column], errors="coerce")
    bad = df.loc[dates.isna() & df[cols.date_column].notna(), cols.date_column]
    if len(bad):
        if len(bad) == 1:
            raise ValueError(f"1 date in '{cols.date_column}' couldn't be read: '{bad.iloc[0]}'.")
        raise ValueError(
            f"{len(bad)} dates in '{cols.date_column}' couldn't be read, for example '{bad.iloc[0]}'."
        )
    values, parsed_text = to_numbers(df[cols.value_column])
    bad = df.loc[values.isna() & ~is_blank(df[cols.value_column]), cols.value_column]
    if len(bad):
        if len(bad) == 1:
            raise ValueError(f"1 value in '{cols.value_column}' isn't a number: '{bad.iloc[0]}'.")
        raise ValueError(
            f"{len(bad)} values in '{cols.value_column}' aren't numbers, for example '{bad.iloc[0]}'."
        )
    frame = pd.DataFrame({"date": dates.dt.normalize(), "value": values})
    for c in group_by:
        frame[c] = df[c].astype(str)
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        raise ValueError(f"'{cols.date_column}' holds no dates.")
    freq = detect_freq(pd.DatetimeIndex(frame["date"].unique()))

    if group_by:
        grouped = frame.groupby(group_by, sort=True)
        items = [(k if isinstance(k, tuple) else (k,), g) for k, g in grouped]
    else:
        items = [((), frame)]
    groups = []
    for key, rows in items:
        # min_count=1: a date whose values are all missing stays missing, never zero.
        summed = rows.groupby("date")["value"].sum(min_count=1).sort_index()
        summed.name = cols.value_column
        label = " · ".join(str(k) for k in key) if key else "All rows"
        groups.append(GroupSeries(key=key, label=label, values=summed, total=float(summed.sum())))
    found = len(groups)
    groups.sort(key=lambda g: -g.total)
    return SalesSeries(
        freq=freq,
        groups=groups[:MAX_GROUPS],
        n_groups_found=found,
        group_by=group_by,
        week_label=cols.week_label,
        parsed_text=parsed_text,
    )


# Money as text: an optional sign or brackets, a currency symbol, digits with
# thousands commas, and pence.
_MONEY = re.compile(r"^(?P<neg>-)?\s*[£$€]?\s*(?P<num>\d{1,3}(,\d{3})+(\.\d+)?|\d+(\.\d+)?|\.\d+)$")
_NONE = {"-", "\u2013", "\u2014", ""}


def is_blank(column: pd.Series) -> pd.Series:
    """Missing, or a bare dash: no figure for that row."""
    return column.isna() | column.map(lambda v: isinstance(v, str) and v.strip() in _NONE)


def to_numbers(column: pd.Series) -> tuple[pd.Series, str | None]:
    """The column as numbers, and an example of the text read, if any was.

    Numbers pass through. Text such as "£1,234.56", "1,234.56", "-£3.50" and
    "(£12.00)" (a negative, as accounts write it) is read; "-" alone, or a
    blank, is missing, never zero. Anything else stays missing here and is
    reported by the caller as not a number.
    """
    if pd.api.types.is_numeric_dtype(column):
        return column.astype(float), None
    example = None

    def parse(v):
        nonlocal example
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        text = str(v).strip()
        if text in _NONE:
            return np.nan
        try:
            return float(text)  # a plain number stored as text
        except ValueError:
            pass
        negative = text.startswith("(") and text.endswith(")")
        match = _MONEY.match((text[1:-1] if negative else text).replace(" ", ""))
        if not match:
            return np.nan
        example = example or text
        number = float(match["num"].replace(",", ""))
        return -number if (negative or match["neg"]) else number

    return column.map(parse).astype(float), example


def detect_freq(dates: pd.DatetimeIndex) -> Freq:
    """Daily, or weekly when every date falls on one weekday.

    Anything else is refused: the statistics read daily or weekly rows, and
    monthly totals hold too few values a year to tell a change from noise.
    """
    dates = dates.sort_values()
    if len(dates) < 2:
        raise ValueError("There's only one date, so there's no history to compare with.")
    gaps = np.diff(dates.to_numpy()).astype("timedelta64[D]").astype(int)
    if len(set(dates.dayofweek)) == 1 and (gaps % 7 == 0).all():
        weekly_freq(dates)  # one weekday: never raises here
        return "W"
    if np.median(gaps) == 1:
        return "D"
    raise ValueError(
        f"The dates are neither daily nor weekly: they are typically {int(np.median(gaps))} "
        f"days apart. The statistics need one row per day or per week (rows with "
        f"the same date are summed)."
    )


def windows_from_flags(
    df: pd.DataFrame, cols: SalesColumns, flag_column: str, freq: Freq
) -> list[PromotionWindow]:
    """Each group's consecutive flagged dates, as promotion windows.

    A flag is true for 1, true, yes, y or t (any case); anything else, or
    missing, is false. Consecutive means the next day for daily data, the next
    week for weekly.
    """
    if flag_column not in df.columns:
        raise ValueError(missing_columns([flag_column], df.columns))
    truthy = {"1", "1.0", "true", "yes", "y", "t"}
    flags = df[flag_column].map(lambda v: str(v).strip().lower() in truthy)
    group_by = list(cols.group_by or [])
    frame = pd.DataFrame({"date": pd.to_datetime(df[cols.date_column], errors="coerce").dt.normalize(), "flag": flags})
    for c in group_by:
        frame[c] = df[c].astype(str)
    frame = frame[frame["flag"] & frame["date"].notna()]
    step = pd.Timedelta(days=7 if freq == "W" else 1)
    windows: list[PromotionWindow] = []
    items = frame.groupby(group_by, sort=True) if group_by else [((), frame)]
    for key, rows in items:
        key = key if isinstance(key, tuple) else (key,)
        days = pd.DatetimeIndex(sorted(rows["date"].unique()))
        if len(days) == 0:
            continue
        run_start = prev = days[0]
        for day in [*days[1:], None]:
            if day is not None and day - prev == step:
                prev = day
                continue
            first, last = run_start, prev
            if freq == "W":
                # A flagged week is the whole week: give the window its days.
                first, last = _week_span(first, last, cols.week_label)
            windows.append(
                PromotionWindow(
                    name=None,
                    start=first.date(),
                    end=last.date(),
                    group=dict(zip(group_by, key)) if group_by else None,
                )
            )
            if day is not None:
                run_start = prev = day
    return windows


def windows_from_calendar(
    calendar: pd.DataFrame,
    cols: SalesColumns,
    start_column: str,
    end_column: str,
    name_column: str | None,
) -> list[PromotionWindow]:
    """Promotion windows from a calendar table: one row per promotion.

    The calendar's group columns are matched by name with ``group_by``; a
    group column it lacks means the promotion ran in every value of it.
    """
    wanted = [start_column, end_column, *([name_column] if name_column else [])]
    missing = [c for c in wanted if c not in calendar.columns]
    if missing:
        raise ValueError(missing_columns(missing, calendar.columns, where=" in the promotion calendar"))
    group_by = [c for c in (cols.group_by or []) if c in calendar.columns]
    windows = []
    # A group column whose value is "all", "*" or blank leaves that column
    # out of the match: the promotion ran in every value of it.
    for _, row in calendar.iterrows():
        start, end = pd.to_datetime(row[start_column], errors="coerce"), pd.to_datetime(row[end_column], errors="coerce")
        if pd.isna(start) or pd.isna(end):
            raise ValueError(
                f"A promotion's dates couldn't be read: '{row[start_column]}' to '{row[end_column]}'."
            )
        windows.append(
            PromotionWindow(
                name=None if not name_column or pd.isna(row[name_column]) else str(row[name_column]),
                start=start.date(),
                end=end.date(),
                group={c: str(row[c]) for c in group_by if not is_wildcard(row[c])} or None,
            )
        )
    return windows


def windows_for(group: GroupSeries, windows: list[PromotionWindow], group_by: list[str]) -> list[PromotionWindow]:
    """The windows that ran in this group."""
    labels = dict(zip(group_by, group.key))
    return [
        w for w in windows
        if not w.group or all(is_wildcard(v) or labels.get(c) == str(v) for c, v in w.group.items())
    ]


def promotions(state, df: pd.DataFrame, params, series: SalesSeries) -> list[PromotionWindow] | None:
    """The promotions given to a tool, however given; None when none were."""
    if params.windows is not None:
        return list(params.windows)
    if params.promo_column is not None:
        return windows_from_flags(df, params, params.promo_column, series.freq)
    if params.calendar_dataframe is not None:
        return windows_from_calendar(
            state.get_dataframe(params.calendar_dataframe), params,
            params.calendar_start_column, params.calendar_end_column, params.calendar_name_column,
        )
    return None


def missing_columns(missing: list[str], columns, where: str = "") -> str:
    """"Couldn't find the column 'week'. The columns are: date, retailer, units."."""
    names = [f"'{c}'" for c in missing]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"
    noun = "column" if len(names) == 1 else "columns"
    whose = "Its" if where else "The"
    return f"Couldn't find the {noun} {listed}{where}. {whose} columns are: {', '.join(map(str, columns))}."


def _week_span(first: pd.Timestamp, last: pd.Timestamp, week_label: WeekLabel) -> tuple[pd.Timestamp, pd.Timestamp]:
    if week_label == "end":
        return first - pd.Timedelta(days=6), last
    return first, last + pd.Timedelta(days=6)
