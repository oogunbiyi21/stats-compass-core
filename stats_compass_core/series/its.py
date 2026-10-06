"""Interrupted time series: did a window move a series, against what would have happened?

Architecture §7.5. For one window (a promotion) and one daily series:

- **Counterfactual.** The mean of the seasonally adjusted values over the
  ``baseline_days`` before the window, plus the seasonal pattern on each day
  of the window. The seasonal pattern comes from a decomposition that never
  saw the window, the ``post_days`` after it (where pull-forward leaves a
  dip), or any ``other_windows``: those days are excluded from the fit.
- **Lift.** Observed total over the window minus the counterfactual total, in
  the series' units and in percent of the counterfactual.
- **Interval and test.** The same estimator applied at every past position
  where it can be tried, with the seasonal pattern read one period earlier, as
  a forecast reads it (``Decomposition.expectation_errors``). The interval
  subtracts the errors' quantiles from the lift; the p-value ranks the lift
  among them. Exact, so nothing is drawn. Reading the seasonal in-sample at
  past positions instead, where the fit had already absorbed part of each
  window's noise, gave p-values two to three times too small.
- **Verdict.** §7.3 applied to the lift: the expected band is zero ±
  ``seasonal_band_z`` × the errors' spread; with ``noise_band="prior_window"``
  the noise band is the lift estimated at ``prior_window`` ±
  ``noise_band_z`` × √2 × that spread (the spread of a difference of two
  independent estimates); then the test. ``noise_band="none"`` skips the noise
  band, because a period-to-period band has no clear meaning for a lift. Which
  one applies is a calibration decision for the caller.

Not measured: pull-forward. Customers who would have bought after the window
but bought inside it inflate the lift, and the dip they leave is excluded from
the fit rather than netted off.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

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
    post_days: int = 21,
    prior_window: Window | None = None,
    other_windows: Sequence[Window] = (),
    noise_band: NoiseBand = "prior_window",
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
        baseline_days: Days before the window whose seasonally adjusted mean
            is the counterfactual level.
        post_days: Days after the window excluded from the fit and from the
            reference, where pull-forward leaves a dip.
        prior_window: The comparable window one cycle earlier. Its estimated
            lift centres the noise band.
        other_windows: Other promotions, excluded the same way.
        noise_band: ``"prior_window"`` or ``"none"``; see the module notes.
        level: Interval level.
        seasonal_band_z, noise_band_z, real_change_alpha: The calibration.
        **decompose_options: Passed to ``decompose`` (``max_gap_days``,
            ``annual_smoothing_days``, ``fourier_terms``, ``robust``).

    Returns:
        ``LiftFacts``, or ``Insufficient``: the decomposition's own reasons,
        ``BASELINE_TOO_SHORT``, ``TOO_FEW_POINTS`` (under half the window has
        data), ``TOO_SHORT`` (too few past positions to measure the error).
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
    window_days = pd.date_range(start, end)
    observed = observed_series.reindex(window_days).astype(float)
    present = observed.notna()
    if int(present.sum()) * 2 < length:
        return Insufficient(
            needs=(length + 1) // 2,
            has=int(present.sum()),
            unit="days",
            reason="TOO_FEW_POINTS",
        )

    estimate = _estimate(dec, observed_series, start, end, baseline_days, present)
    if isinstance(estimate, Insufficient):
        return estimate
    observed_total, counterfactual_total = estimate
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
    spread = float(np.std(reference, ddof=1))
    expected_range = (-seasonal_band_z * spread, seasonal_band_z * spread)

    prior_lift_pct: float | None = None
    params_prior: dict[str, Any] = {}
    if prior_window is not None:
        prior = _prior_lift(dec, observed_series, prior_window, baseline_days)
        if prior is not None:
            prior_lift_pct, populated = prior
            params_prior = {"prior_window_populated_share": populated}
    noise_range: tuple[float, float] | None = None
    if (
        noise_band == "prior_window"
        and prior_lift_pct is not None
        and lift_pct is not None
    ):
        half = noise_band_z * np.sqrt(2) * spread
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
        "post_days": post_days,
        "prior_window": None
        if prior_window is None
        else [str(prior_window[0]), str(prior_window[1])],
        "other_windows": [[str(a), str(b)] for a, b in other_windows],
        "noise_band": noise_band,
        "level": level,
        "seasonal_band_z": seasonal_band_z,
        "noise_band_z": noise_band_z,
        "real_change_alpha": real_change_alpha,
        "exclude": [[str(a), str(b)] for a, b in exclusions],
        "n_window_days_with_data": int(present.sum()),
        "judged_on": "percent" if lift_pct is not None else "absolute",
        "decomposition": dec.params,
        **params_prior,
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
    end: pd.Timestamp,
    baseline_days: int,
    present: pd.Series,
) -> tuple[float, float] | Insufficient:
    """Observed and counterfactual totals over the days of the window with data."""
    seasonal = dec._seasonal_daily()
    adjusted = dec.daily - seasonal
    baseline = pd.date_range(
        start - pd.Timedelta(days=baseline_days), start - pd.Timedelta(days=1)
    )
    filled = set(dec.imputed) | set(dec.excluded)
    real = [
        day for day in baseline if day in adjusted.index and day.date() not in filled
    ]
    if len(real) < MIN_BASELINE_SHARE * baseline_days:
        return Insufficient(
            needs=baseline_days, has=len(real), unit="days", reason="BASELINE_TOO_SHORT"
        )
    level = float(adjusted.loc[real].mean())
    days = present.index[present.to_numpy()]
    counterfactual = level * len(days) + float(seasonal.reindex(days).sum())
    observed = float(values.reindex(days).sum())
    return observed, counterfactual


def _prior_lift(
    dec: Decomposition, values: pd.Series, prior: Window, baseline_days: int
) -> tuple[float, float] | None:
    """The same estimate at the comparable prior window, and its populated share."""
    start, end = pd.Timestamp(prior[0]), pd.Timestamp(prior[1])
    days = pd.date_range(start, end)
    observed = values.reindex(days).astype(float)
    present = observed.notna()
    if (
        not present.any()
        or start - pd.Timedelta(days=baseline_days) < dec.daily.index[0]
    ):
        return None
    estimate = _estimate(dec, values, start, end, baseline_days, present)
    if isinstance(estimate, Insufficient) or estimate[1] <= 0:
        return None
    observed_total, counterfactual_total = estimate
    lift = 100 * (observed_total - counterfactual_total) / counterfactual_total
    return float(lift), float(present.mean())


def _interval(estimate: float, errors: np.ndarray, tail: float) -> tuple[float, float]:
    low, high = np.quantile(errors, [tail, 1 - tail])
    return float(estimate - high), float(estimate - low)
