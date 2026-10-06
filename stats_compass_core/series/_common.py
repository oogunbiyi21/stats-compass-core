"""Pieces shared by the series statistics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

Unit = Literal["days", "months", "orders", "customers"]


@dataclass(frozen=True)
class Insufficient:
    """Too little data for the statistic: what it needs, and what it was given.

    Returned, never raised. A caller deciding what to show cannot tell a
    shortage of data from a bug if both arrive as exceptions, and the shortage
    is the expected case for a young store. ``reason`` is a code, not prose;
    the caller words it.
    """

    needs: int
    has: int
    unit: Unit
    reason: str


def prepare_daily(
    values: pd.Series, max_gap_days: int
) -> tuple[pd.Series, list[date]] | Insufficient:
    """One value per calendar day, short gaps filled, and the filled days named.

    A missing day means "no data", never zero, so it is never treated as zero.
    Interior gaps up to ``max_gap_days`` are filled by linear interpolation and
    returned as imputed; a longer gap makes the series insufficient rather than
    invented. Missing days before the first value or after the last are
    trimmed: there is nothing on one side to interpolate from.
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
    series = series.loc[series.first_valid_index() : series.last_valid_index()]

    missing = series.isna()
    longest = _longest_run(missing)
    if longest > max_gap_days:
        return Insufficient(
            needs=max_gap_days, has=longest, unit="days", reason="GAP_TOO_LONG"
        )

    imputed = [day.date() for day in series.index[missing]]
    if imputed:
        series = series.interpolate(method="time", limit_area="inside")
    series.index.freq = "D"
    return series, imputed


def _longest_run(flags: pd.Series) -> int:
    longest = current = 0
    for flag in flags.to_numpy():
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def week_start(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The Monday of each date's week."""
    return index - pd.to_timedelta(index.dayofweek, unit="D")
