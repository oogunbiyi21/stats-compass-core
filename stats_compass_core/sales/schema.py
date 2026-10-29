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
  group columns by the same names).

The expected shape for consumer-goods brands, still to be confirmed: weekly
rows by week ending × retailer × product, with units and sales value, and a
promotion calendar (retailer, product, start, end, mechanic, trade spend).
"""

from __future__ import annotations

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


class PromotionWindow(BaseModel):
    """One promotion: its days, and which group it ran in."""

    name: str | None = Field(default=None, description="What to call it, such as 'Summer 2 for 1'.")
    start: date = Field(description="Its first day.")
    end: date = Field(description="Its last day.")
    group: dict[str, str] | None = Field(
        default=None,
        description=(
            "Which group it ran in, by the group_by columns, for example "
            "{'retailer': 'Tesco'}. Omitted: every group."
        ),
    )


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


def build_series(df: pd.DataFrame, cols: SalesColumns) -> SalesSeries:
    """Split the rows into groups, sum each to one value per date, and detect the frequency.

    Raises ValueError, in plain words, for a missing column, dates or values
    that do not parse, or dates that are neither daily nor weekly.
    """
    group_by = list(cols.group_by or [])
    needed = [cols.date_column, cols.value_column, *group_by]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found: {missing}. The columns are: {list(df.columns)}"
        )
    dates = pd.to_datetime(df[cols.date_column], errors="coerce")
    bad = df.loc[dates.isna() & df[cols.date_column].notna(), cols.date_column]
    if len(bad):
        raise ValueError(
            f"{len(bad)} date(s) in '{cols.date_column}' could not be read, for "
            f"example {bad.iloc[0]!r}."
        )
    values = pd.to_numeric(df[cols.value_column], errors="coerce")
    bad = df.loc[values.isna() & df[cols.value_column].notna(), cols.value_column]
    if len(bad):
        raise ValueError(
            f"{len(bad)} value(s) in '{cols.value_column}' are not numbers, for "
            f"example {bad.iloc[0]!r}."
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
    )


def detect_freq(dates: pd.DatetimeIndex) -> Freq:
    """Daily, or weekly when every date falls on one weekday.

    Anything else is refused: the statistics read daily or weekly rows, and
    monthly totals hold too few values a year to tell a change from noise.
    """
    dates = dates.sort_values()
    if len(dates) < 2:
        raise ValueError("There is only one date, so there is no history to compare with.")
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
        raise ValueError(f"Column '{flag_column}' not found.")
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
    for c in [start_column, end_column, *([name_column] if name_column else [])]:
        if c not in calendar.columns:
            raise ValueError(
                f"Column '{c}' not found in the promotion calendar. Its columns are: "
                f"{list(calendar.columns)}"
            )
    group_by = [c for c in (cols.group_by or []) if c in calendar.columns]
    windows = []
    for _, row in calendar.iterrows():
        start, end = pd.to_datetime(row[start_column], errors="coerce"), pd.to_datetime(row[end_column], errors="coerce")
        if pd.isna(start) or pd.isna(end):
            raise ValueError(
                f"A promotion's dates could not be read: {row[start_column]!r} to {row[end_column]!r}."
            )
        windows.append(
            PromotionWindow(
                name=None if not name_column or pd.isna(row[name_column]) else str(row[name_column]),
                start=start.date(),
                end=end.date(),
                group={c: str(row[c]) for c in group_by} or None,
            )
        )
    return windows


def windows_for(group: GroupSeries, windows: list[PromotionWindow], group_by: list[str]) -> list[PromotionWindow]:
    """The windows that ran in this group."""
    labels = dict(zip(group_by, group.key))
    return [
        w for w in windows
        if not w.group or all(labels.get(c) == str(v) for c, v in w.group.items())
    ]


def _week_span(first: pd.Timestamp, last: pd.Timestamp, week_label: WeekLabel) -> tuple[pd.Timestamp, pd.Timestamp]:
    if week_label == "end":
        return first - pd.Timedelta(days=6), last
    return first, last + pd.Timedelta(days=6)
