"""Interrupted time series: did a window move a series, against what would have happened?

Architecture §7.5. For one window (a promotion) and one daily series:

- **Counterfactual.** The mean of the seasonally adjusted values over the
  ``baseline_days`` real days before the window, plus the seasonal pattern on
  each day of the window. Both adjustments read the seasonal one period
  earlier, the way a forecast made before the window would have read it. The
  decomposition never sees the window, the ``post_days`` after it (where
  pull-forward leaves a dip), or any ``other_windows``: those days are
  excluded from the fit.
- **Lift.** Observed total over the window minus the counterfactual total, in
  the series' units and in percent of the counterfactual.
- **Interval and test.** The same estimator applied at every past position
  where it can be tried (``Decomposition.expectation_errors``): the counterfactual
  and its reference are one function. The interval subtracts the errors'
  quantiles from the lift; the p-value ranks the lift among them. Exact, so
  nothing is drawn. The first version read the window's seasonal in-sample
  while its reference read it a period earlier; its 90% intervals missed zero
  for 20% of inert promotions.
- **Verdict.** §7.3 applied to the lift. The expected band is the central
  ``band_level`` of the errors, where ``band_level`` is the normal coverage of
  ``±seasonal_band_z`` (0.954 at 2): the same empirical basis as the interval,
  at the band's level. With ``noise_band="prior_window"`` the noise band is the
  lift estimated at ``prior_window`` ± ``noise_band_z`` × √2 × the errors'
  spread: two estimates by the same function, each with the reference's
  variance. It needs a seasonal period before the prior window, so with two
  years of history it is unavailable. ``"none"`` (the default until the
  calibration is decided) skips it.

The interval is at ``level`` and the band at ``band_level``; a 90% interval can
exclude zero while the lift is still ``expected``. Both levels are in
``params`` so a caller does not print one beside the other as a contradiction.

Not measured: pull-forward. Customers who would have bought after the window
but bought inside it inflate the lift, and the dip they leave is excluded from
the fit rather than netted off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from stats_compass_core.results import ToolWarning
from stats_compass_core.series._common import Insufficient
from stats_compass_core.series.decompose import Decomposition, Method, decompose

Kind = Literal["count", "money"]
NoiseBand = Literal["prior_window", "none"]
Window = tuple[date, date]

# A baseline needs this share of its days to hold real data.
MIN_BASELINE_SHARE = 0.8
MIN_REFERENCE_WINDOWS = 20


@dataclass(frozen=True)
class LiftFacts:
    """A window's lift over its counterfactual, with an interval and a verdict.

    ``estimate_pct`` and ``interval_pct`` are None when the counterfactual
    total is not positive: a percentage of zero or less means nothing.
    """

    estimate_abs: float
    interval_abs: tuple[float, float]
    estimate_pct: float | None
    interval_pct: tuple[float, float] | None
    observed_total: float
    counterfactual_total: float
    verdict: Literal["expected", "noise", "real change"]
    decided_by: Literal["expected_band", "noise_band", "test"]
    p_value: float
    n_reference: int
    prior_lift_pct: float | None
    expected_range_pct: tuple[float, float] | None
    noise_range_pct: tuple[float, float] | None
    method: str
    params: dict[str, Any]
    warnings: list[ToolWarning]


def its_lift(
    values: pd.Series,
    window: Window,
    *,
    kind: Kind,
    method: Method = "mstl",
    periods: Sequence[int] = (7, 365),
    baseline_days: int = 28,
    baseline_lookback_days: int = 84,
    post_days: int = 21,
    prior_window: Window | None = None,
    other_windows: Sequence[Window] = (),
    noise_band: NoiseBand = "none",
    level: float = 0.9,
    seasonal_band_z: float = 2.0,
    noise_band_z: float = 3.0,
    real_change_alpha: float = 0.05,
    **decompose_options: Any,
) -> LiftFacts | Insufficient:
    """Estimate one window's lift on one daily series.

    Args:
        values: The full daily history, the window included. NaN = no data.
        window: First and last day of the window.
        kind: ``"money"`` or ``"count"``. Both use the empirical test; the kind
            is recorded so a caller can see what was assumed.
        method, periods: The decomposition (see ``decompose``).
        baseline_days: Real days before the window whose seasonally adjusted
            mean is the counterfactual level.
        baseline_lookback_days: How far back the baseline may reach past
            excluded or missing days to find them. Promotions a month apart
            would otherwise leave each other no baseline.
        post_days: Days after the window excluded from the fit and from the
            reference, where pull-forward leaves a dip.
        prior_window: The comparable window one cycle earlier; its lift
            centres the noise band when ``noise_band="prior_window"``.
        other_windows: Other promotions, excluded the same way.
        noise_band: ``"none"`` or ``"prior_window"``; see the module notes.
        level: Interval level.
        seasonal_band_z, noise_band_z, real_change_alpha: The calibration.
        **decompose_options: Passed to ``decompose`` (``max_gap_days``,
            ``annual_smoothing_days``, ``fourier_terms``, ``robust``).

    Returns:
        ``LiftFacts``, or ``Insufficient``: the decomposition's own reasons,
        ``BASELINE_TOO_SHORT``, ``TOO_FEW_POINTS`` (under half the window has
        data), ``NO_SEASONAL_BEFORE`` (no seasonal period before the window
        to read), ``TOO_SHORT`` (too few past positions to measure the error).
    """
    if kind not in ("count", "money"):
        raise ValueError(f"unknown kind {kind!r}")
    if noise_band not in ("prior_window", "none"):
        raise ValueError(f"unknown noise_band {noise_band!r}")
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    if end < start:
        raise ValueError("window ends before it starts")
    length = (end - start).days + 1

    def tail(w: Window) -> tuple[date, date]:
        return (w[0], w[1] + timedelta(days=post_days))

    exclusions = [tail(window), *[tail(w) for w in other_windows]]
    dec = decompose(
        values,
        method=method,
        periods=periods,
        exclude=exclusions,
        trend_window_days=baseline_days,
        **decompose_options,
    )
    if isinstance(dec, Insufficient):
        return dec

    observed_series = values.copy()
    observed_series.index = pd.DatetimeIndex(observed_series.index).normalize()
    observed = observed_series.reindex(pd.date_range(start, end)).astype(float)
    present = observed.notna()
    if int(present.sum()) * 2 < length:
        return Insufficient(
            needs=(length + 1) // 2,
            has=int(present.sum()),
            unit="days",
            reason="TOO_FEW_POINTS",
        )

    estimate = _estimate(
        dec, observed_series, start, present, baseline_days, baseline_lookback_days
    )
    if isinstance(estimate, Insufficient):
        return estimate
    observed_total, counterfactual_total, baseline_info = estimate
    lift_abs = observed_total - counterfactual_total
    lift_pct = (
        100 * lift_abs / counterfactual_total if counterfactual_total > 0 else None
    )

    errors = dec.expectation_errors(length)
    if len(errors) < MIN_REFERENCE_WINDOWS:
        return Insufficient(
            needs=MIN_REFERENCE_WINDOWS,
            has=len(errors),
            unit="days",
            reason="TOO_SHORT",
        )
    err_abs = errors["error"].to_numpy()
    positive = errors["expected"].to_numpy() > 0
    err_pct = 100 * err_abs[positive] / errors["expected"].to_numpy()[positive]

    tail_q = (1 - level) / 2
    interval_abs = _interval(lift_abs, err_abs, tail_q)
    interval_pct = (
        _interval(lift_pct, err_pct, tail_q) if lift_pct is not None else None
    )

    # Judge on the percent scale when there is one: it does not drift with the
    # store's growth the way absolute errors do.
    judged, reference = (
        (lift_pct, err_pct) if lift_pct is not None else (lift_abs, err_abs)
    )
    p_value = (1 + int(np.sum(np.abs(reference) >= abs(judged)))) / (len(reference) + 1)
    band_level = float(2 * stats.norm.cdf(seasonal_band_z) - 1)
    band_low, band_high = np.quantile(
        reference, [(1 - band_level) / 2, (1 + band_level) / 2]
    )
    expected_range = (float(band_low), float(band_high))
    spread = float(np.std(reference, ddof=1))

    prior_lift_pct: float | None = None
    prior_info: dict[str, Any] = {}
    if prior_window is not None:
        prior = _prior_lift(
            dec, observed_series, prior_window, baseline_days, baseline_lookback_days
        )
        if isinstance(prior, str):
            prior_info = {"prior_unavailable": prior}
        else:
            prior_lift_pct, populated = prior
            prior_info = {"prior_window_populated_share": populated}
    noise_range: tuple[float, float] | None = None
    if (
        noise_band == "prior_window"
        and prior_lift_pct is not None
        and lift_pct is not None
    ):
        half = noise_band_z * math.sqrt(2) * spread
        noise_range = (prior_lift_pct - half, prior_lift_pct + half)

    if expected_range[0] <= judged <= expected_range[1]:
        outcome, decided_by = "expected", "expected_band"
    elif noise_range is not None and noise_range[0] <= judged <= noise_range[1]:
        outcome, decided_by = "noise", "noise_band"
    elif p_value < real_change_alpha:
        outcome, decided_by = "real change", "test"
    else:
        outcome, decided_by = "noise", "test"

    params: dict[str, Any] = {
        "kind": kind,
        "window": [str(window[0]), str(window[1])],
        "baseline_days": baseline_days,
        "baseline_lookback_days": baseline_lookback_days,
        "post_days": post_days,
        "prior_window": None
        if prior_window is None
        else [str(prior_window[0]), str(prior_window[1])],
        "other_windows": [[str(a), str(b)] for a, b in other_windows],
        "noise_band": noise_band,
        "level": level,
        "interval_level": level,
        "band_level": band_level,
        "band_basis": "empirical_quantiles",
        "seasonal_band_z": seasonal_band_z,
        "noise_band_z": noise_band_z,
        "real_change_alpha": real_change_alpha,
        "exclude": [[str(a), str(b)] for a, b in exclusions],
        "n_window_days_with_data": int(present.sum()),
        "judged_on": "percent" if lift_pct is not None else "absolute",
        "decomposition": dec.params,
        **baseline_info,
        **prior_info,
    }
    return LiftFacts(
        estimate_abs=float(lift_abs),
        interval_abs=interval_abs,
        estimate_pct=None if lift_pct is None else float(lift_pct),
        interval_pct=interval_pct,
        observed_total=float(observed_total),
        counterfactual_total=float(counterfactual_total),
        verdict=outcome,
        decided_by=decided_by,
        p_value=float(p_value),
        n_reference=int(len(reference)),
        prior_lift_pct=prior_lift_pct,
        expected_range_pct=expected_range if lift_pct is not None else None,
        noise_range_pct=noise_range,
        method=f"{dec.method}+its_empirical_windows",
        params=params,
        warnings=list(dec.warnings),
    )


def _estimate(
    dec: Decomposition,
    values: pd.Series,
    start: pd.Timestamp,
    present: pd.Series,
    baseline_days: int,
    lookback_days: int,
) -> tuple[float, float, dict[str, Any]] | Insufficient:
    """Observed and counterfactual totals over the window's days with data.

    The estimator ``expectation_errors`` measures: seasonal read one period
    earlier, level from the adjusted baseline. The baseline walks back past
    excluded, imputed and missing days, up to ``lookback_days``, until it has
    ``baseline_days`` real ones.
    """
    lagged = dec._lagged_seasonal_daily()
    adjusted = dec.daily - lagged
    filled = set(dec.imputed) | set(dec.excluded)
    needed = math.ceil(MIN_BASELINE_SHARE * baseline_days)
    chosen: list[pd.Timestamp] = []
    for back in range(1, lookback_days + 1):
        day = start - pd.Timedelta(days=back)
        if day < adjusted.index[0]:
            break
        if day.date() in filled or not np.isfinite(adjusted.get(day, np.nan)):
            continue
        chosen.append(day)
        if len(chosen) == baseline_days:
            break
    if len(chosen) < needed:
        return Insufficient(
            needs=needed, has=len(chosen), unit="days", reason="BASELINE_TOO_SHORT"
        )
    days = present.index[present.to_numpy()]
    seasonal = lagged.reindex(days)
    if not np.isfinite(seasonal.to_numpy()).all():
        return Insufficient(
            needs=None, has=None, unit="days", reason="NO_SEASONAL_BEFORE"
        )
    level = float(adjusted.loc[chosen].mean())
    counterfactual = level * len(days) + float(seasonal.sum())
    observed = float(values.reindex(days).sum())
    info = {
        "baseline_real_days": len(chosen),
        "baseline_first_day": min(chosen).date().isoformat(),
    }
    return observed, counterfactual, info


def _prior_lift(
    dec: Decomposition,
    values: pd.Series,
    prior: Window,
    baseline_days: int,
    lookback_days: int,
) -> tuple[float, float] | str:
    """The same estimate at the comparable prior window, or why there is none."""
    start, end = pd.Timestamp(prior[0]), pd.Timestamp(prior[1])
    observed = values.reindex(pd.date_range(start, end)).astype(float)
    present = observed.notna()
    if not present.any():
        return "NO_DATA_IN_PRIOR"
    estimate = _estimate(dec, values, start, present, baseline_days, lookback_days)
    if isinstance(estimate, Insufficient):
        return (
            "NO_SEASONAL_BEFORE_PRIOR"
            if estimate.reason == "NO_SEASONAL_BEFORE"
            or estimate.reason == "BASELINE_TOO_SHORT"
            and not np.isfinite(dec._lagged_seasonal_daily().get(start, np.nan))
            else estimate.reason
        )
    observed_total, counterfactual_total, _ = estimate
    if counterfactual_total <= 0:
        return "NONPOSITIVE_COUNTERFACTUAL"
    lift = 100 * (observed_total - counterfactual_total) / counterfactual_total
    return float(lift), float(present.mean())


def _interval(estimate: float, errors: np.ndarray, tail: float) -> tuple[float, float]:
    low, high = np.quantile(errors, [tail, 1 - tail])
    return float(estimate - high), float(estimate - low)
