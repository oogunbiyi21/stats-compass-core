"""A seasonally adjusted daily forecast whose intervals are measured, not modelled.

- **Point.** The same expectation the verdict reads ahead of the data (the
  seasonally adjusted level over the last ``trend_window_days``, plus the
  seasonal pattern repeated from its last full period), plus the median of
  that forecast's past errors at each horizon. A flat level under-forecasts a
  growing store; the median error corrects it by as much as it has been wrong
  before.
- **Intervals.** At every past origin where the forecast can be tried, its
  error at each horizon, with the seasonal pattern read as it would have been
  known at that origin (the latest same-phase value before it). Daily
  intervals are the per-horizon quantiles; monthly intervals are the
  quantiles of each origin's errors summed over the same days. Exact over the
  origins: nothing is simulated and no seed is needed.
- **Never narrowing.** Each side's distance from the point is the running
  maximum over the horizon, so ``lower <= mean <= upper`` always holds and no
  later day is more certain than an earlier one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from stats_compass_core.results import ToolWarning
from stats_compass_core.series._common import Insufficient
from stats_compass_core.series.decompose import Decomposition


@dataclass(frozen=True)
class ForecastPoint:
    day: date
    mean: float
    lower: float
    upper: float


@dataclass(frozen=True)
class ForecastMonth:
    month: str  # YYYY-MM
    mean: float
    lower: float
    upper: float
    wide: bool  # (upper - lower) / mean above wide_ratio, or a mean at or below zero
    n_days: int  # forecast days falling in this month


@dataclass(frozen=True)
class ForecastFacts:
    points: list[ForecastPoint]
    months: list[ForecastMonth]
    n_origins: int
    method: str
    params: dict[str, Any]
    warnings: list[ToolWarning]


def forecast(
    dec: Decomposition,
    *,
    horizon_days: int = 90,
    level: float = 0.9,
    wide_ratio: float = 0.5,
    min_origins: int = 30,
) -> ForecastFacts | Insufficient:
    """Forecast the ``horizon_days`` after the decomposed history.

    Args:
        dec: A decomposition of the history up to the day before the forecast.
        horizon_days: Days to forecast.
        level: Interval level.
        wide_ratio: A month is flagged ``wide`` when its interval's width
            exceeds this share of its mean.
        min_origins: Fewest past origins to measure the error from.

    Returns:
        ``ForecastFacts``, or ``Insufficient(reason="TOO_SHORT")`` when the
        history holds fewer than ``min_origins`` origins.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")

    errors = _horizon_errors(dec, horizon_days)
    annual_fallback = False
    warnings: list[ToolWarning] = []
    if len(errors) < min_origins:
        # The annual pattern can use up the history the origins need; the
        # weekly pattern alone may still leave enough.
        fallback = dec.without_annual()
        if fallback is not None:
            fallback_errors = _horizon_errors(fallback, horizon_days)
            if len(fallback_errors) >= min_origins:
                dec, errors, annual_fallback = fallback, fallback_errors, True
                warnings = [w for w in dec.warnings if w.code == "ANNUAL_DROPPED"]
    if len(errors) < min_origins:
        return Insufficient(
            needs=len(dec.daily) + min_origins - len(errors),
            has=len(dec.daily),
            unit="days",
            reason="TOO_SHORT",
        )

    first = pd.Timestamp(dec.params["end"]) + pd.Timedelta(days=1)
    days = pd.date_range(first, periods=horizon_days, freq="D")
    base = dec.expected_daily(days[0].date(), days[-1].date()).to_numpy()

    tail = (1 - level) / 2
    low_q, mid_q, high_q = np.quantile(errors, [tail, 0.5, 1 - tail], axis=0)
    mean = base + mid_q
    below = np.maximum.accumulate(mid_q - low_q)
    above = np.maximum.accumulate(high_q - mid_q)
    points = [
        ForecastPoint(
            day=d.date(), mean=float(m), lower=float(m - b), upper=float(m + a)
        )
        for d, m, b, a in zip(days, mean, below, above)
    ]

    months: list[ForecastMonth] = []
    labels = days.strftime("%Y-%m")
    for label in dict.fromkeys(labels):
        mask = labels == label
        totals = errors[:, mask].sum(axis=1)
        # The origins overlap: a month's totals from n origins hold only about
        # n / n_days independent months, too few for tail quantiles. Use a
        # prediction interval on that effective sample instead: t critical
        # value x spread x sqrt(1 + 1/n_eff).
        m_mid = float(np.median(totals))
        n_eff = max(len(totals) / max(int(mask.sum()), 1), 2.0)
        t_crit = float(stats.t.ppf(1 - tail, df=n_eff - 1))
        m_half = t_crit * float(np.std(totals, ddof=1)) * float(np.sqrt(1 + 1 / n_eff))
        m_low, m_high = m_mid - m_half, m_mid + m_half
        total = float(base[mask].sum())
        mid = total + m_mid
        width = float(m_high - m_low)
        months.append(
            ForecastMonth(
                month=label,
                mean=float(sum(p.mean for p, keep in zip(points, mask) if keep)),
                lower=float(total + m_low),
                upper=float(total + m_high),
                wide=bool(mid <= 0 or width / mid > wide_ratio),
                n_days=int(mask.sum()),
            )
        )
        # The month's mean is the sum of its daily means, so it always equals
        # what the daily points add up to; keep its interval around it.
        last = months[-1]
        if not last.lower <= last.mean <= last.upper:
            months[-1] = ForecastMonth(
                month=last.month,
                mean=last.mean,
                lower=min(last.lower, last.mean),
                upper=max(last.upper, last.mean),
                wide=last.wide,
                n_days=last.n_days,
            )

    params = {
        "annual_fallback": annual_fallback,
        "horizon_days": horizon_days,
        "level": level,
        "wide_ratio": wide_ratio,
        "min_origins": min_origins,
        "bias_corrected": True,
        "decomposition_method": dec.method,
        "trend_ahead": dec.params["trend_ahead"],
        "reference": "out_of_sample_exact" if dec.linear is not None else "in_sample",
        # Origins overlap: monthly quantiles rest on about this many
        # independent months, not on n_origins.
        "effective_independent_months": round(len(errors) / 30, 1),
        "first_day": days[0].date().isoformat(),
    }
    return ForecastFacts(
        points=points,
        months=months,
        n_origins=int(len(errors)),
        method=f"{dec.method}+empirical_horizon_errors",
        params=params,
        warnings=warnings,
    )


def _horizon_errors(dec: Decomposition, horizon: int) -> np.ndarray:
    """Errors of the forecast from every usable past origin: origins × horizon."""
    y = dec.daily.to_numpy(dtype=float)
    n = len(y)
    baseline = int(dec.params["trend_window_days"])
    seasonal_by_day = _seasonal_by_day(dec)
    adjusted_lagged = y - _one_period_back(dec, seasonal_by_day)
    level = pd.Series(adjusted_lagged).rolling(baseline).mean().shift(1).to_numpy()

    excluded = np.zeros(n, dtype=bool)
    if dec.excluded:
        excluded = dec.daily.index.isin(pd.DatetimeIndex(dec.excluded))

    rows = []
    for origin in range(baseline, n - horizon + 1):
        if not np.isfinite(level[origin]):
            continue
        if excluded[origin - baseline : origin + horizon].any():
            continue
        seasonal = np.zeros(horizon)
        ok = True
        for period, component in seasonal_by_day.items():
            steps = np.arange(1, horizon + 1)
            back = period * np.ceil(steps / period).astype(int)
            idx = origin + steps - 1 - back
            if (idx < 0).any():
                ok = False
                break
            values = component[idx]
            if not np.isfinite(values).all():
                ok = False
                break
            seasonal += values
        if not ok:
            continue
        error = y[origin : origin + horizon] - level[origin] - seasonal
        if dec.linear is not None:
            # Out of sample: delete the horizon's rows from the Fourier fit
            # and move the error by what that does to the coefficients.
            Z = dec.linear["Z"]
            delta = dec._deletion_delta(np.arange(origin, origin + horizon))
            gradient = (
                Z[origin - baseline : origin].mean(axis=0)
                - Z[origin : origin + horizon]
            )
            error = error - gradient @ delta
        rows.append(error)
    return np.asarray(rows).reshape(-1, horizon)


def _seasonal_by_day(dec: Decomposition) -> dict[int, np.ndarray]:
    """Each seasonal component as a daily array, with its period in days."""
    if dec.grain == "day":
        return {
            period: component.reindex(dec.daily.index).to_numpy(dtype=float)
            for period, component in dec.seasonal_by_period.items()
        }
    days = dec.daily.index
    shares = np.asarray(dec.day_of_week_share)[days.dayofweek]
    weeks = days - pd.to_timedelta(days.dayofweek, unit="D")
    return {
        period * 7: component.reindex(weeks).to_numpy(dtype=float) * shares
        for period, component in dec.seasonal_by_period.items()
    }


def _one_period_back(
    dec: Decomposition, seasonal_by_day: dict[int, np.ndarray]
) -> np.ndarray:
    total = np.zeros(len(dec.daily))
    for period, values in seasonal_by_day.items():
        shifted = np.full(len(values), np.nan)
        shifted[period:] = values[:-period]
        total = total + shifted
    return total
