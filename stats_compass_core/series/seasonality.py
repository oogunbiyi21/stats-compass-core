"""How far each calendar month runs above or below trend, with an interval.

The effect is defined the way it is measured on data: the month's mean
deviation from trend (with the weekly pattern removed), as a share of the
month's mean trend. It is computed from the observed values, not from the
annual component, so it does not depend on how that component was smoothed,
and one-day spikes such as Black Friday count towards their month.

**Why not a bootstrap.** Resampling the remainder was the first plan. With two
years of history the annual component follows the noise, the remainder is far
smaller than the noise, and intervals built from it excluded zero for 94% of
months on stores with no seasonality at all. The interval here uses the
spread of the deviations themselves: a Newey-West standard error with a t
critical value whose degrees of freedom count weeks, not days, because days
within a month are not independent. On 40 patternless stores at level 0.9 it
excluded zero for 9.6% of months with ``mstl``, 9.2% with ``fourier`` and
12.5% with ``stl``.

At weekly grain a month holds four or five values a year, and a Newey-West
correction on so few made the interval narrower, not wider (16.9% excluding
zero at one lag), so a weekly decomposition uses the plain standard error.

**What the interval does not cover.** With two years of history it describes
the noise within those years. It says nothing about whether the next November
will look like these two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from stats_compass_core.results import ToolWarning
from stats_compass_core.series._common import Insufficient
from stats_compass_core.series.decompose import Decomposition

# A month needs at least this many days of data to be estimated at all.
MIN_DAYS_PER_MONTH = 14
# Deviations smaller than this share of the level are floating-point residue.
NUMERICAL_ZERO = 1e-9


@dataclass(frozen=True)
class MonthEffect:
    month: int
    effect_pct: float
    lower: float
    upper: float


@dataclass(frozen=True)
class MonthEffects:
    rows: list[MonthEffect]
    interval_basis: str
    hac_lags: int
    years_of_history: float
    params: dict[str, Any]
    warnings: list[ToolWarning]


def month_effects(
    dec: Decomposition, *, level: float = 0.9, hac_lags: int = 7
) -> MonthEffects | Insufficient:
    """Each calendar month's effect in percent of trend, with a ``level`` interval.

    Args:
        dec: A decomposition from ``decompose``.
        level: Interval level, e.g. 0.9.
        hac_lags: Newey-West lags, in days. A weekly-grain decomposition
            ignores it and uses the plain standard error (lags 0).

    Returns:
        Twelve ``MonthEffect`` rows, or ``Insufficient`` when a month has fewer
        than 14 days of data, or a month's trend is not positive (an effect in
        percent of a level at or below zero means nothing).
    """
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")

    daily = dec.grain == "day"
    if daily and not any(p >= 300 for p in dec.seasonal_by_period):
        # Without an annual component the trend follows the seasons, and every
        # month's deviation from it reads as zero.
        return Insufficient(
            needs=730, has=len(dec.observed), unit="days", reason="NO_ANNUAL"
        )
    deviation = (
        dec.observed
        - dec.trend
        - (dec.seasonal_by_period.get(7, 0.0) if daily else 0.0)
    )
    months = (
        dec.observed.index.month
        if daily
        else (dec.observed.index + pd.Timedelta(days=3)).month
    )
    lags = hac_lags if daily else 0
    days_per_unit = 1 if daily else 7

    counts = pd.Series(months).value_counts()
    have = min(int(counts.get(m, 0)) * days_per_unit for m in range(1, 13))
    if have < MIN_DAYS_PER_MONTH:
        return Insufficient(
            needs=MIN_DAYS_PER_MONTH, has=have, unit="days", reason="MONTH_TOO_SHORT"
        )

    rows: list[MonthEffect] = []
    df_by_month: dict[int, int] = {}
    for month in range(1, 13):
        in_month = months == month
        level_mean = float(dec.trend[in_month].mean())
        if level_mean <= 0:
            return Insufficient(
                needs=None, has=None, unit="days", reason="NONPOSITIVE_LEVEL"
            )
        values = deviation[in_month].to_numpy(dtype=float)
        # Rounding dust from the fit is not a deviation: a flat series must give
        # intervals of exactly zero, not ones that miss zero by 1e-14.
        values[np.abs(values) <= NUMERICAL_ZERO * level_mean] = 0.0
        df = max(len(values) * days_per_unit // 7 - 1, 1)
        df_by_month[month] = df
        half_width = stats.t.ppf(0.5 + level / 2, df) * _newey_west_se(values, lags)
        estimate = values.mean()
        rows.append(
            MonthEffect(
                month=month,
                effect_pct=float(100 * estimate / level_mean),
                lower=float(100 * (estimate - half_width) / level_mean),
                upper=float(100 * (estimate + half_width) / level_mean),
            )
        )

    n_days = len(dec.observed) * days_per_unit
    return MonthEffects(
        rows=rows,
        interval_basis="newey_west_t",
        hac_lags=lags,
        years_of_history=n_days / 365.25,
        params={
            "level": level,
            "hac_lags": lags,
            "grain": dec.grain,
            "df_by_month": df_by_month,
            "decomposition_method": dec.method,
        },
        warnings=[],
    )


def _newey_west_se(values: np.ndarray, lags: int) -> float:
    """Standard error of the mean with a Bartlett-kernel long-run variance."""
    n = len(values)
    centred = values - values.mean()
    variance = centred @ centred / n
    for lag in range(1, min(lags, n - 1) + 1):
        variance += 2 * (1 - lag / (lags + 1)) * (centred[lag:] @ centred[:-lag]) / n
    return float(np.sqrt(max(variance, 0.0) / n))
