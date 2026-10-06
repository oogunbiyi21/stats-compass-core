"""Runs: four or more weekly moves the same way, ending at the latest week.

Architecture §7.6 asks for runs "whose joint probability under the noise model
is < realChangeAlpha". Under independent noise any run of four monotone moves
has probability 2/5! (about 0.017), whatever its size, so that gate would pass
every run; and real weekly series drift and are autocorrelated, which makes
runs commoner than independent noise suggests. So the noise model is the
store's own history: the run is ranked among every earlier stretch of the same
number of weeks, counting those that were also monotone (either way) and moved
at least as far.

Weekly values are complete Monday-start totals of the daily values minus the
seasonal pattern read one period earlier. Last year's pattern adjusts this
year's weeks, so the run cannot leak into its own seasonal through the fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from stats_compass_core.results import ToolWarning
from stats_compass_core.series._common import Insufficient, week_start
from stats_compass_core.series.decompose import Decomposition


@dataclass(frozen=True)
class RunFacts:
    direction: Literal["up", "down"]
    moves: int
    weeks: list[
        tuple[pd.Timestamp, float]
    ]  # (Monday, adjusted weekly total), oldest first
    change_abs: float
    change_pct: float | None  # None when the run's first week is not positive
    p_value: float
    n_reference: int
    significant: bool
    method: str
    params: dict[str, Any]
    warnings: list[ToolWarning]


def detect_run(
    dec: Decomposition,
    *,
    min_moves: int = 4,
    real_change_alpha: float = 0.05,
    min_reference: int = 20,
) -> RunFacts | None | Insufficient:
    """The run of same-direction weekly moves ending at the latest complete week.

    Args:
        dec: A decomposition of the full history, the latest week included.
        min_moves: Fewest consecutive same-direction moves that make a run.
        real_change_alpha: The significance level.
        min_reference: Fewest earlier stretches to rank against.

    Returns:
        ``RunFacts``; None when no run of ``min_moves`` ends at the latest
        week; ``Insufficient(reason="TOO_SHORT")`` when there are too few
        earlier stretches to compare with.
    """
    if min_moves < 1:
        raise ValueError("min_moves must be at least 1")
    weekly = _adjusted_weeks(dec)
    if len(weekly) < 2:
        return None
    values = weekly.to_numpy()
    moves = np.sign(np.diff(values))
    last = moves[-1]
    if last == 0:
        return None
    count = 0
    for move in moves[::-1]:
        if move != last:
            break
        count += 1
    if count < min_moves:
        return None

    span = count + 1
    change = float(values[-1] - values[-span])
    earlier = values[: len(values) - span]
    windows = (
        np.lib.stride_tricks.sliding_window_view(earlier, span)
        if len(earlier) >= span
        else np.empty((0, span))
    )
    if len(windows) < min_reference:
        return Insufficient(
            needs=len(dec.daily) + 7 * (min_reference - len(windows)),
            has=len(dec.daily),
            unit="days",
            reason="TOO_SHORT",
        )
    steps = np.diff(windows, axis=1)
    monotone = (steps > 0).all(axis=1) | (steps < 0).all(axis=1)
    moved = np.abs(windows[:, -1] - windows[:, 0]) >= abs(change)
    p_value = (int((monotone & moved).sum()) + 1) / (len(windows) + 1)

    start_value = float(values[-span])
    return RunFacts(
        direction="up" if last > 0 else "down",
        moves=count,
        weeks=[(week, float(value)) for week, value in weekly.iloc[-span:].items()],
        change_abs=change,
        change_pct=100 * change / start_value if start_value > 0 else None,
        p_value=float(p_value),
        n_reference=int(len(windows)),
        significant=bool(p_value < real_change_alpha),
        method="lagged_seasonal+empirical_runs",
        params={
            "min_moves": min_moves,
            "real_change_alpha": real_change_alpha,
            "min_reference": min_reference,
            "decomposition_method": dec.method,
            "n_weeks": len(weekly),
        },
        warnings=[],
    )


def _adjusted_weeks(dec: Decomposition) -> pd.Series:
    """Complete Monday-start weekly totals of daily values minus last period's seasonal."""
    adjusted = dec.daily - dec._lagged_seasonal_daily()
    if dec.excluded:
        adjusted = adjusted.copy()
        adjusted.loc[pd.DatetimeIndex(dec.excluded)] = np.nan
    weeks = week_start(adjusted.index)
    grouped = adjusted.groupby(weeks)
    complete = (grouped.count() == 7) & (grouped.size() == 7)
    totals = grouped.sum()[complete]
    # A run is consecutive weeks: keep only the unbroken stretch that reaches
    # the latest week, so a gap in the data does not join two stretches.
    expected = (
        pd.date_range(totals.index[0], totals.index[-1], freq="7D")
        if len(totals)
        else totals.index
    )
    totals = totals.reindex(expected)
    breaks = np.flatnonzero(totals.isna().to_numpy())
    if len(breaks):
        totals = totals.iloc[breaks[-1] + 1 :]
    return totals
