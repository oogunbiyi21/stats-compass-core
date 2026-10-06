"""Pieces shared by the series statistics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Sequence

import pandas as pd

Unit = Literal["days", "months", "orders", "customers"]


@dataclass(frozen=True)
class Insufficient:
    """Too little data for the statistic: what it needs, and what it was given.

    Returned, never raised. A caller deciding what to show cannot tell a
    shortage of data from a bug if both arrive as exceptions, and the shortage
    is the expected case for a young store. ``reason`` is a code, not prose;
    the caller words it.

    ``needs`` and ``has`` read as "needs X; you have Y" for the shortage
    reasons (``TOO_SHORT``, ``TOO_FEW_POINTS``, ``MONTH_TOO_SHORT``,
    ``NO_DATA``, ``NO_DENOMINATOR``). For ``GAP_TOO_LONG`` they are the longest
    gap allowed and the longest found. Where no number applies
    (``NONPOSITIVE_LEVEL``) both are None and the reason alone explains it.
    """

    needs: int | None
    has: int | None
    unit: Unit
    reason: str


@dataclass(frozen=True)
class Prepared:
    """A daily series ready to decompose, and an account of what was done to it."""

    values: pd.Series
    imputed: list[date]  # missing days filled by interpolation
    excluded: list[date]  # days left out of the fit on request, also filled for it
    trimmed: list[date]  # missing days before the first value or after the last


def prepare_daily(
    values: pd.Series,
    max_gap_days: int,
    exclude: Sequence[tuple[date, date]] = (),
) -> Prepared | Insufficient:
    """One value per calendar day, short gaps filled, and every change named.

    A missing day means "no data", never zero, so it is never treated as zero.
    Interior gaps up to ``max_gap_days`` are filled by linear interpolation and
    returned as imputed; a longer gap makes the series insufficient rather than
    invented. Missing days before the first value or after the last are
    trimmed, since there is nothing on one side to interpolate from, and
    returned as trimmed.

    Days inside an ``exclude`` window are left out of the fit on purpose (a
    promotion, say): their values are replaced by interpolation like a gap,
    but they are not a gap, so they never make the series insufficient.
    """
    if not isinstance(values, pd.Series) or not isinstance(
        values.index, pd.DatetimeIndex
    ):
        raise TypeError("values must be a pandas Series with a DatetimeIndex")
    if values.index.has_duplicates:
        raise ValueError("values has repeated dates; pass one value per day")

    series = values.sort_index().astype(float)
    series.index = series.index.normalize()
    if series.notna().sum() == 0:
        return Insufficient(needs=1, has=0, unit="days", reason="NO_DATA")

    full = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full)
    first, last = series.first_valid_index(), series.last_valid_index()
    trimmed = [day.date() for day in full if day < first or day > last]
    series = series.loc[first:last]

    excluded_mask = pd.Series(False, index=series.index)
    for start, end in exclude:
        excluded_mask.loc[pd.Timestamp(start) : pd.Timestamp(end)] = True

    missing = series.isna() & ~excluded_mask
    longest = _longest_run(missing)
    if longest > max_gap_days:
        return Insufficient(
            needs=max_gap_days, has=longest, unit="days", reason="GAP_TOO_LONG"
        )

    imputed = [day.date() for day in series.index[missing]]
    excluded = [day.date() for day in series.index[excluded_mask]]
    series = series.mask(excluded_mask)
    if imputed or excluded:
        series = series.interpolate(method="time", limit_direction="both")
    series.index.freq = "D"
    return Prepared(values=series, imputed=imputed, excluded=excluded, trimmed=trimmed)


def _longest_run(flags: pd.Series) -> int:
    longest = current = 0
    for flag in flags.to_numpy():
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def week_start(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The Monday of each date's week."""
    return index - pd.to_timedelta(index.dayofweek, unit="D")
