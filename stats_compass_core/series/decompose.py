"""Seasonal decomposition of a daily series: trend + seasonal + remainder.

Three methods behind one signature, so they can be compared on the same data:

- ``mstl``: statsmodels MSTL on the daily series, one seasonal component per
  period (by default weekly and annual).
- ``stl``: STL with period 52 on Monday-start weekly sums. The annual pattern
  only; a weekly grain has no day-of-week pattern to find.
- ``fourier``: least squares on a linear trend, day-of-week effects and
  ``fourier_terms`` annual harmonics.

All are additive: a store can have days of zero or negative net revenue, which
a multiplicative model cannot represent.

**Why the annual component is smoothed.** With two years of history, a daily
annual component is estimated from two values per day of the year, and it
follows the noise in them. On patternless data MSTL's remainder had a tenth of
the true noise's spread, so bands and tests built on the remainder called
ordinary weeks a real change. ``annual_smoothing_days`` replaces the annual
component with a centred moving average of itself and returns what it removed
to the remainder. A one-day spike such as Black Friday therefore lands in the
remainder, not in the seasonal pattern. Set it to 0 to see the raw component.
The Fourier method is smooth by construction and is not smoothed again.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from stats_compass_core.results import ToolWarning
from stats_compass_core.series._common import Insufficient, prepare_daily, week_start

Method = Literal["mstl", "stl", "fourier"]
TrendAhead = Literal["flat", "linear"]

WEEKLY_PERIOD = 52
# Periods at least this long are treated as the annual pattern, and smoothed.
ANNUAL_MIN_PERIOD_DAYS = 300


@dataclass(frozen=True)
class WeeklyComponents:
    """The decomposition as Monday-start weekly sums, complete weeks only."""

    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    remainder: pd.Series


@dataclass(frozen=True, eq=False)
class Decomposition:
    """An additive decomposition: ``observed = trend + seasonal + remainder``.

    Series are indexed by day (``grain == "day"``) or by the Monday that starts
    each week (``grain == "week"``). ``observed`` is the series decomposed, with
    imputed days filled in; ``daily`` is the same input at daily grain, whatever
    the method.

    On ``excluded`` days the remainder is NaN: their values were left out of
    the fit, so nothing observed is left over. Everything that measures noise
    skips them.
    """

    method: str
    grain: Literal["day", "week"]
    params: dict[str, Any]
    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    remainder: pd.Series
    seasonal_by_period: dict[int, pd.Series]
    imputed: list[date]
    warnings: list[ToolWarning]
    daily: pd.Series = field(repr=False, default_factory=lambda: pd.Series(dtype=float))
    day_of_week_share: tuple[float, ...] | None = field(default=None, repr=False)
    excluded: list[date] = field(default_factory=list)
    trimmed: list[date] = field(default_factory=list)

    # -- expectation ---------------------------------------------------------

    def expected_daily(self, start: date, end: date) -> pd.Series:
        """Trend + seasonal for each day in ``[start, end]``, in-sample or ahead.

        To judge a period, decompose the history that ends the day before it
        and read the expectation ahead. An in-sample expectation has already
        absorbed the period: the trend bends towards it, and a real change
        reads as expected. ``verdict`` refuses a decomposition that overlaps
        the period for that reason.

        Ahead of the data, each seasonal component repeats its value from whole
        periods earlier and the trend follows ``params["trend_ahead"]``. A
        weekly-grain decomposition apportions each week's expectation over its
        days by the store's day-of-week shares.
        """
        days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
        if len(days) == 0:
            return pd.Series(dtype=float)
        if self.grain == "day":
            return pd.Series([self._expected_at(day) for day in days], index=days)
        weeks = week_start(days)
        shares = np.asarray(self.day_of_week_share)
        values = [
            self._expected_at(week) * shares[day.dayofweek]
            for day, week in zip(days, weeks)
        ]
        return pd.Series(values, index=days)

    def expected_total(self, start: date, end: date) -> float:
        """The sum of ``expected_daily`` over ``[start, end]``."""
        return float(self.expected_daily(start, end).sum())

    def _expected_at(self, when: pd.Timestamp) -> float:
        first, last = self.trend.index[0], self.trend.index[-1]
        if when < first:
            raise ValueError(
                f"{when.date()} is before the decomposed series starts ({first.date()})"
            )
        if when <= last:
            return float(self.trend.loc[when] + self.seasonal.loc[when])
        return self._trend_ahead(when) + sum(
            self._seasonal_ahead(component, period, when)
            for period, component in self.seasonal_by_period.items()
        )

    def _step(self) -> pd.Timedelta:
        return pd.Timedelta(days=1 if self.grain == "day" else 7)

    def _seasonal_ahead(
        self, component: pd.Series, period: int, when: pd.Timestamp
    ) -> float:
        span = period * self._step()
        back = when - span
        while back > component.index[-1]:
            back -= span
        return float(component.loc[back])

    def _trend_ahead(self, when: pd.Timestamp) -> float:
        window = max(
            1,
            round(self.params["trend_window_days"] / (1 if self.grain == "day" else 7)),
        )
        recent = self.trend.iloc[-window:]
        if self.params["trend_ahead"] == "flat":
            return float(recent.mean())
        slope = (recent.iloc[-1] - recent.iloc[0]) / max(len(recent) - 1, 1)
        steps = (when - self.trend.index[-1]) / self._step()
        return float(recent.iloc[-1] + slope * steps)

    # -- noise ---------------------------------------------------------------

    def residual_sigma(self, ndays: int, robust: bool = False) -> float:
        """Spread of the remainder summed over ``ndays`` consecutive days.

        Measured, not modelled: the spread of every ``ndays``-day rolling sum,
        so autocorrelation in the remainder is reflected without being
        specified. A weekly grain gives whole weeks exactly and scales anything
        else by the square root of its length.

        ``robust=False`` is the standard deviation. ``robust=True`` is
        1.4826 × the median absolute deviation, which matches it on plain noise
        but is not widened by the store's own past spikes (each Black Friday
        sits in the remainder).
        """
        if ndays < 1:
            raise ValueError("ndays must be at least 1")
        if self.grain == "day":
            return _spread(self.remainder.rolling(ndays).sum().dropna(), robust)
        if ndays % 7 == 0:
            return _spread(self.remainder.rolling(ndays // 7).sum().dropna(), robust)
        return _spread(self.remainder.dropna(), robust) * float(np.sqrt(ndays / 7))

    # -- display -------------------------------------------------------------

    def weekly(self) -> WeeklyComponents:
        """Monday-start weekly sums, complete weeks only, whatever the method."""
        parts = {
            "observed": self.observed,
            "trend": self.trend,
            "seasonal": self.seasonal,
            "remainder": self.remainder,
        }
        if self.grain == "week":
            return WeeklyComponents(**parts)
        weeks = week_start(self.observed.index)
        complete = (
            pd.Series(1, index=self.observed.index).groupby(weeks).transform("size")
            == 7
        )
        summed = {
            name: series[complete.to_numpy()].groupby(weeks[complete.to_numpy()]).sum()
            for name, series in parts.items()
        }
        return WeeklyComponents(**summed)


def decompose(
    values: pd.Series,
    *,
    method: Method = "mstl",
    periods: Sequence[int] = (7, 365),
    max_gap_days: int = 7,
    annual_smoothing_days: int = 31,
    fourier_terms: int = 4,
    trend_ahead: TrendAhead = "flat",
    trend_window_days: int = 28,
    robust: bool = False,
    exclude: Sequence[tuple[date, date]] = (),
) -> Decomposition | Insufficient:
    """Decompose a daily series into trend, seasonal components and remainder.

    Args:
        values: One value per day, on a DatetimeIndex. NaN means no data, never
            zero; missing dates are treated the same way.
        method: ``"mstl"``, ``"stl"`` (weekly sums, period 52) or ``"fourier"``.
        periods: Seasonal periods in days, for ``mstl`` and ``fourier``.
        max_gap_days: The longest run of missing days filled by interpolation.
            A longer gap returns ``Insufficient(reason="GAP_TOO_LONG")``.
        annual_smoothing_days: Width of the centred moving average applied to
            the annual component (``mstl`` and ``stl``); 0 disables it.
        fourier_terms: Annual harmonics, for ``fourier``.
        trend_ahead: How ``expected_*`` continues the trend past the data:
            ``"flat"`` holds its mean over the last ``trend_window_days``,
            ``"linear"`` extends its slope over that window.
        trend_window_days: The window ``trend_ahead`` reads.
        robust: Robust LOESS fitting for ``mstl`` and ``stl`` alike, so that a
            comparison of the methods is not a comparison of robustness.
        exclude: ``(start, end)`` windows to leave out of the fit, such as a
            promotion being measured. Their values are replaced by
            interpolation for fitting; unlike a gap, they never make the series
            insufficient. They are returned in ``excluded``.

    Returns:
        A ``Decomposition``, or ``Insufficient`` when there is too little
        history for the method (``TOO_SHORT``) or a gap too long to fill.
    """
    if method not in ("mstl", "stl", "fourier"):
        raise ValueError(f"unknown method {method!r}; use 'mstl', 'stl' or 'fourier'")
    periods = sorted({int(p) for p in periods})

    prepared = prepare_daily(values, max_gap_days, exclude)
    if isinstance(prepared, Insufficient):
        return prepared
    y, imputed = prepared.values, prepared.imputed

    needs = {
        "mstl": 2 * max(periods),
        "stl": 2 * WEEKLY_PERIOD * 7,
        "fourier": 365,
    }[method]
    if len(y) < needs:
        return Insufficient(needs=needs, has=len(y), unit="days", reason="TOO_SHORT")

    params: dict[str, Any] = {
        "method": method,
        "periods": periods,
        "max_gap_days": max_gap_days,
        "annual_smoothing_days": annual_smoothing_days,
        "fourier_terms": fourier_terms,
        "trend_ahead": trend_ahead,
        "trend_window_days": trend_window_days,
        "robust": robust,
        "exclude": [[str(start), str(end)] for start, end in exclude],
        "start": y.index[0].date().isoformat(),
        "end": y.index[-1].date().isoformat(),
        "n_days": len(y),
        "n_imputed": len(imputed),
        "n_excluded": len(prepared.excluded),
        "n_trimmed": len(prepared.trimmed),
    }
    columns = [str(values.name)] if values.name is not None else []
    warnings: list[ToolWarning] = []
    if prepared.trimmed:
        warnings.append(
            ToolWarning(
                code="TRIMMED_DAYS",
                columns=columns,
                message=(
                    f"{len(prepared.trimmed)} day(s) with no data before the first "
                    f"value or after the last were left out: there is nothing on one "
                    f"side to fill them from. The series runs {y.index[0].date()} to "
                    f"{y.index[-1].date()}."
                ),
            )
        )
    if imputed:
        warnings.append(
            ToolWarning(
                code="IMPUTED_DAYS",
                columns=columns,
                message=(
                    f"{len(imputed)} day(s) with no data were filled by linear "
                    f"interpolation before decomposing; the longest gap allowed is "
                    f"{max_gap_days} days."
                ),
            )
        )

    if method == "mstl":
        dec = _mstl(
            y, periods, imputed, params, warnings, annual_smoothing_days, robust
        )
    elif method == "stl":
        dec = _stl_weekly(y, imputed, params, warnings, annual_smoothing_days, robust)
    else:
        dec = _fourier(y, periods, imputed, params, warnings, fourier_terms)
    return _account_for(dec, prepared.excluded, prepared.trimmed)


def _account_for(
    dec: Decomposition, excluded: list[date], trimmed: list[date]
) -> Decomposition:
    """Attach what was excluded and trimmed; blank the remainder where excluded."""
    remainder = dec.remainder
    if excluded:
        days = pd.DatetimeIndex(excluded)
        hit = days if dec.grain == "day" else week_start(days).unique()
        remainder = remainder.copy()
        remainder.loc[remainder.index.isin(hit)] = np.nan
    return replace(dec, remainder=remainder, excluded=excluded, trimmed=trimmed)


def _spread(values: pd.Series, robust: bool) -> float:
    if robust:
        centred = values - values.median()
        return float(1.4826 * centred.abs().median())
    return float(values.std(ddof=1))


def _require_statsmodels():
    try:
        from statsmodels.tsa.seasonal import MSTL, STL
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "decompose needs statsmodels. Install with: pip install stats-compass-core[timeseries]"
        ) from exc
    return MSTL, STL


def _statsmodels_version() -> str:
    import statsmodels

    return statsmodels.__version__


def _smooth(component: pd.Series, width: int) -> pd.Series:
    if width <= 1:
        return component
    return component.rolling(width, center=True, min_periods=1).mean()


def _mstl(
    y, periods, imputed, params, warnings, annual_smoothing_days, robust
) -> Decomposition:
    MSTL, _ = _require_statsmodels()
    fit = MSTL(y, periods=periods, stl_kwargs={"robust": robust}).fit()
    seasonal_frame = (
        fit.seasonal.to_frame(f"seasonal_{periods[0]}")
        if isinstance(fit.seasonal, pd.Series)
        else fit.seasonal
    )
    params["statsmodels_version"] = _statsmodels_version()
    remainder = fit.resid.copy()
    components: dict[int, pd.Series] = {}
    for column in seasonal_frame.columns:
        # statsmodels names each component "seasonal_<period>"; match by name,
        # never by position.
        period = int(str(column).rsplit("_", 1)[1])
        component = seasonal_frame[column].rename(None)
        if period >= ANNUAL_MIN_PERIOD_DAYS and annual_smoothing_days > 1:
            smooth = _smooth(component, annual_smoothing_days)
            remainder = remainder + (component - smooth)
            component = smooth
        components[period] = component
    seasonal = sum(components.values())
    return Decomposition(
        method="mstl",
        grain="day",
        params=params,
        observed=y.rename(None),
        daily=y.rename(None),
        trend=fit.trend.rename(None),
        seasonal=seasonal,
        remainder=remainder.rename(None),
        seasonal_by_period=components,
        imputed=imputed,
        warnings=warnings,
    )


def _stl_weekly(
    y, imputed, params, warnings, annual_smoothing_days, robust
) -> Decomposition:
    _, STL = _require_statsmodels()
    weeks = week_start(y.index)
    complete = (
        pd.Series(1, index=y.index).groupby(weeks).transform("size").to_numpy() == 7
    )
    weekly = y[complete].groupby(weeks[complete]).sum()
    weekly.index = pd.DatetimeIndex(weekly.index, freq="W-MON")

    fit = STL(weekly, period=WEEKLY_PERIOD, robust=robust).fit()
    smoothing_weeks = (
        max(1, round(annual_smoothing_days / 7)) if annual_smoothing_days > 1 else 1
    )
    if smoothing_weeks > 1 and smoothing_weeks % 2 == 0:
        smoothing_weeks += 1
    annual = _smooth(fit.seasonal, smoothing_weeks)
    remainder = fit.resid + (fit.seasonal - annual)

    by_day = y.groupby(y.index.dayofweek).mean()
    shares = tuple(
        float(v) for v in (by_day / by_day.sum()).reindex(range(7), fill_value=1 / 7)
    )
    params.update(
        weekly_period=WEEKLY_PERIOD,
        annual_smoothing_weeks=smoothing_weeks,
        n_weeks=len(weekly),
        statsmodels_version=_statsmodels_version(),
    )
    return Decomposition(
        method="stl",
        grain="week",
        params=params,
        observed=weekly.rename(None),
        daily=y.rename(None),
        trend=fit.trend.rename(None),
        seasonal=annual.rename(None),
        remainder=remainder.rename(None),
        seasonal_by_period={WEEKLY_PERIOD: annual.rename(None)},
        imputed=imputed,
        warnings=warnings,
        day_of_week_share=shares,
    )


def _fourier(y, periods, imputed, params, warnings, fourier_terms) -> Decomposition:
    t = (y.index - y.index[0]).days.to_numpy(dtype=float)
    columns: dict[str, np.ndarray] = {"const": np.ones(len(y)), "t": t / 365.25}
    weekly = 7 in periods
    if weekly:
        for dow in range(1, 7):
            columns[f"dow{dow}"] = (y.index.dayofweek == dow).astype(float)
    annual_periods = [p for p in periods if p != 7]
    for period in annual_periods:
        for k in range(1, fourier_terms + 1):
            angle = 2 * np.pi * k * t / (365.25 if period == 365 else period)
            columns[f"sin{period}_{k}"] = np.sin(angle)
            columns[f"cos{period}_{k}"] = np.cos(angle)
    X = pd.DataFrame(columns, index=y.index)
    beta, *_ = np.linalg.lstsq(X.to_numpy(), y.to_numpy(), rcond=None)
    coef = pd.Series(beta, index=X.columns)

    components: dict[int, pd.Series] = {}
    level_shift = 0.0
    if weekly:
        effects = np.array([0.0] + [coef[f"dow{d}"] for d in range(1, 7)])
        level_shift = effects.mean()
        components[7] = pd.Series(
            effects[y.index.dayofweek] - level_shift, index=y.index
        )
    for period in annual_periods:
        names = [
            c for c in X.columns if c.startswith((f"sin{period}_", f"cos{period}_"))
        ]
        components[period] = X[names] @ coef[names]
    trend = coef["const"] + coef["t"] * X["t"] + level_shift
    seasonal = sum(components.values()) if components else pd.Series(0.0, index=y.index)
    remainder = y - trend - seasonal
    return Decomposition(
        method="fourier",
        grain="day",
        params=params,
        observed=y.rename(None),
        daily=y.rename(None),
        trend=trend.rename(None),
        seasonal=seasonal.rename(None),
        remainder=remainder.rename(None),
        seasonal_by_period=components,
        imputed=imputed,
        warnings=warnings,
    )
