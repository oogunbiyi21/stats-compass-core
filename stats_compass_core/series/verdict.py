"""Is a period's figure expected, noise, or a real change?

Architecture §7.3, with the steps in a fixed order so that a figure can only
be a real change if it is outside both bands *and* survives a test:

1. The expectation comes from a decomposition of the history *before* the
   period. One that included the period would let the trend lean towards the
   very figure being judged, and real changes would read as expected.
2. Expected band: expectation ± ``seasonal_band_z`` × the spread of the
   remainder over a period of this length.
3. Noise band: the prior period's value ± ``noise_band_z`` × the spread of the
   change between consecutive seasonally adjusted periods.
4. Inside the expected band: ``expected``.
5. Else inside the noise band: ``noise``.
6. Else the test that fits the figure's kind, against the expectation:
   ``real change`` if significant at ``real_change_alpha``, otherwise
   ``noise``.
7. Too few days with data in the period: ``Insufficient``.

The test runs on every call, whichever step decides, so its p-value can be
read on its own: a caller can compare the test's false-alarm rate with the
banded procedure's.

The tests, each against the seasonal expectation:

- money (``empirical_windows``): the period's deviation from expectation,
  ranked among the expectation's errors at every past position where the same
  estimator can be tried (``Decomposition.expectation_errors``). Exact, so
  nothing is drawn and no seed is needed. An earlier version drew from
  in-sample remainder windows: the fit had absorbed part of each past
  window's noise, and the test fired at three times its nominal rate.
- count: Poisson, or negative binomial when the history's counts vary more
  than a Poisson would, with the dispersion measured from the same past errors.
- ratio: binomial on the period's successes out of its denominators. It
  assumes the trials are independent; customers who cluster make it
  over-confident, which matters less because it only decides after both bands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from stats_compass_core.results import ToolWarning
from stats_compass_core.series._common import Insufficient
from stats_compass_core.series.decompose import Decomposition

Kind = Literal["count", "money", "ratio"]
Aggregate = Literal["sum", "mean"]
Outcome = Literal["expected", "noise", "real change"]

# The money test needs this many past positions to rank a deviation among.
MIN_REFERENCE_WINDOWS = 20


@dataclass(frozen=True)
class VerdictFacts:
    """What was judged, against what, and which step decided."""

    verdict: Outcome
    observed: float
    expected: float
    expected_range: tuple[float, float]
    noise_range: tuple[float, float]
    change_vs_prior: tuple[
        float, float | None
    ]  # (absolute, percent); None when prior is 0
    p_value: float
    n_reference: int | None  # past positions the money test ranked against
    test: str
    decided_by: Literal["expected_band", "noise_band", "test"]
    method: str
    params: dict[str, Any]
    warnings: list[ToolWarning]


def verdict(
    dec: Decomposition,
    period: pd.Series,
    *,
    kind: Kind,
    weights: pd.Series | None = None,
    aggregate: Aggregate = "sum",
    seasonal_band_z: float = 2.0,
    noise_band_z: float = 3.0,
    real_change_alpha: float = 0.05,
    min_points: int = 5,
    robust_scale: bool = False,
) -> VerdictFacts | Insufficient:
    """Judge one period's figure against the history before it.

    Args:
        dec: A decomposition of the history before ``period``. It should end
            the day before the period starts; days in between are allowed
            only if they were trimmed for having no data.
        period: The period's daily values; NaN means no data.
        kind: ``"count"``, ``"money"`` or ``"ratio"``. Chooses the test.
        weights: Daily denominators (``n``), over at least the prior period
            and the period. Required for ``ratio``; weights a ``mean``.
        aggregate: ``"sum"`` for totals, ``"mean"`` for averages and rates.
        seasonal_band_z, noise_band_z, real_change_alpha: The calibration.
        min_points: Fewest days with data for the period to be judged.
        robust_scale: Measure both bands' spreads with 1.4826 × the median
            absolute deviation instead of the standard deviation, so that the
            store's own past spikes do not widen them. Recorded as
            ``params["scale"]``.

    Returns:
        ``VerdictFacts``, or ``Insufficient``: ``TOO_FEW_POINTS``,
        ``NO_DENOMINATOR``, ``HISTORY_ENDS_EARLY`` (days with data between the
        history and the period), ``TOO_SHORT`` (too little history to measure
        the expectation's error against).

    Raises:
        ValueError: if ``dec`` overlaps ``period``, or the arguments
            contradict each other.
    """
    if kind not in ("count", "money", "ratio"):
        raise ValueError(f"unknown kind {kind!r}")
    if aggregate not in ("sum", "mean"):
        raise ValueError(f"unknown aggregate {aggregate!r}")
    if kind == "ratio" and aggregate != "mean":
        raise ValueError("a ratio is judged as a mean: pass aggregate='mean'")
    if kind == "count" and aggregate != "sum":
        raise ValueError("a count is judged as a total: pass aggregate='sum'")

    period = _daily(period)
    start, end = period.index[0], period.index[-1]
    history_end = pd.Timestamp(dec.params["end"])
    if history_end >= start:
        raise ValueError(
            f"the decomposition must end before the period starts: it ends "
            f"{history_end.date()}, the period starts {start.date()}"
        )
    between = pd.date_range(
        history_end + pd.Timedelta(days=1), start - pd.Timedelta(days=1)
    )
    if len(between) and not set(between.date) <= set(dec.trimmed):
        # A day with data the history left out: the caller cut it short.
        return Insufficient(
            needs=None, has=None, unit="days", reason="HISTORY_ENDS_EARLY"
        )
    length = len(period)

    params: dict[str, Any] = {
        "kind": kind,
        "aggregate": aggregate,
        "seasonal_band_z": seasonal_band_z,
        "noise_band_z": noise_band_z,
        "real_change_alpha": real_change_alpha,
        "min_points": min_points,
        "scale": "mad" if robust_scale else "std",
        "period_start": start.date().isoformat(),
        "period_end": end.date().isoformat(),
        "decomposition_method": dec.method,
        "days_read_across": len(between),
    }
    warnings: list[ToolWarning] = []

    present = period.notna()
    if int(present.sum()) < min_points:
        return Insufficient(
            needs=min_points,
            has=int(present.sum()),
            unit="days",
            reason="TOO_FEW_POINTS",
        )
    if kind == "ratio" and weights is None:
        return Insufficient(needs=length, has=0, unit="days", reason="NO_DENOMINATOR")

    errors = dec.expectation_errors(length)
    params["reference"] = errors.attrs["reference"]
    if len(errors) < MIN_REFERENCE_WINDOWS:
        return Insufficient(
            needs=len(dec.daily) + MIN_REFERENCE_WINDOWS - len(errors),
            has=len(dec.daily),
            unit="days",
            reason="TOO_SHORT",
        )

    expected_daily = dec.expected_daily(start.date(), end.date())
    period_w = _weights_for(weights, period.index)
    if (
        aggregate == "mean"
        and weights is not None
        and float(period_w[present].sum()) == 0
    ):
        warnings.append(
            ToolWarning(
                code="ZERO_WEIGHTS",
                columns=[],
                message="Every weight in the period is zero, so its mean is unweighted.",
            )
        )
    observed = _aggregate(period[present], period_w[present], aggregate)
    expected = _aggregate(expected_daily[present], period_w[present], aggregate)
    params["n_points"] = int(present.sum())

    scale = 1.0 if aggregate == "sum" else 1.0 / length
    sigma_expected = dec.residual_sigma(length, robust=robust_scale) * scale
    expected_range = (
        expected - seasonal_band_z * sigma_expected,
        expected + seasonal_band_z * sigma_expected,
    )

    prior_days = dec.daily.iloc[-length:]
    prior = _aggregate(prior_days, _weights_for(weights, prior_days.index), aggregate)
    sigma_change = _change_sigma(dec, length, robust_scale) * scale
    noise_range = (
        prior - noise_band_z * sigma_change,
        prior + noise_band_z * sigma_change,
    )
    change_abs = observed - prior
    change_pct = None if prior == 0 else 100 * change_abs / prior

    n_reference: int | None = None
    if kind == "money":
        reference = errors["error"].to_numpy() * scale
        exceed = int(np.sum(np.abs(reference) >= abs(observed - expected)))
        test, p_value, n_reference = (
            "empirical_windows",
            (exceed + 1) / (len(reference) + 1),
            len(reference),
        )
    elif kind == "count":
        phi = _dispersion(errors)
        params["dispersion"] = phi
        test, p_value = _count_p(observed, expected, phi)
    else:
        test, p_value = (
            "binomial",
            _binomial_p(period[present], period_w[present], expected),
        )

    if expected_range[0] <= observed <= expected_range[1]:
        outcome, decided_by = "expected", "expected_band"
    elif noise_range[0] <= observed <= noise_range[1]:
        outcome, decided_by = "noise", "noise_band"
    elif p_value < real_change_alpha:
        outcome, decided_by = "real change", "test"
    else:
        outcome, decided_by = "noise", "test"

    return VerdictFacts(
        verdict=outcome,
        observed=float(observed),
        expected=float(expected),
        expected_range=(float(expected_range[0]), float(expected_range[1])),
        noise_range=(float(noise_range[0]), float(noise_range[1])),
        change_vs_prior=(
            float(change_abs),
            None if change_pct is None else float(change_pct),
        ),
        p_value=float(p_value),
        n_reference=n_reference,
        test=test,
        decided_by=decided_by,
        method=f"{dec.method}+{test}",
        params=params,
        warnings=warnings,
    )


# =============================================================================
# Pieces
# =============================================================================


def _daily(values: pd.Series) -> pd.Series:
    if not isinstance(values, pd.Series) or not isinstance(
        values.index, pd.DatetimeIndex
    ):
        raise TypeError("period must be a pandas Series with a DatetimeIndex")
    if len(values) == 0:
        raise ValueError("period is empty")
    values = values.sort_index().astype(float)
    values.index = values.index.normalize()
    full = pd.date_range(values.index[0], values.index[-1], freq="D")
    return values.reindex(full)


def _weights_for(weights: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series:
    if weights is None:
        return pd.Series(1.0, index=index)
    aligned = weights.copy()
    aligned.index = pd.DatetimeIndex(aligned.index).normalize()
    return aligned.reindex(index).fillna(0.0).astype(float)


def _aggregate(values: pd.Series, weights: pd.Series, how: Aggregate) -> float:
    if how == "sum":
        return float(values.sum())
    total = float(weights.sum())
    if total == 0:
        return float(values.mean())
    return float((values * weights).sum() / total)


def _change_sigma(dec: Decomposition, length: int, robust: bool = False) -> float:
    """Spread of the change between consecutive seasonally adjusted periods."""
    adjusted = (dec.observed - dec.seasonal).to_numpy()
    if dec.grain == "day":
        blocks = len(adjusted) // length
        totals = (
            adjusted[len(adjusted) - blocks * length :]
            .reshape(blocks, length)
            .sum(axis=1)
        )
        return _spread(np.diff(totals), robust) if blocks > 2 else float("nan")
    weeks = max(1, round(length / 7))
    blocks = len(adjusted) // weeks
    totals = (
        adjusted[len(adjusted) - blocks * weeks :].reshape(blocks, weeks).sum(axis=1)
    )
    return _spread(np.diff(totals), robust) * length / (7 * weeks)


def _dispersion(errors: pd.DataFrame) -> float:
    """Variance of the expectation's past errors over the expected count."""
    mean = float(errors["expected"].mean())
    return float(errors["error"].var(ddof=1) / mean) if mean > 0 else float("inf")


def _count_p(observed: float, expected: float, phi: float) -> tuple[str, float]:
    mu = max(float(expected), 1e-9)
    count = int(round(observed))
    if phi <= 1:
        dist, name = stats.poisson(mu), "poisson"
    else:
        size = mu / (phi - 1)
        dist, name = stats.nbinom(size, size / (size + mu)), "negative_binomial"
    p = 2 * min(dist.cdf(count), dist.sf(count - 1))
    return name, float(min(p, 1.0))


def _binomial_p(rates: pd.Series, n: pd.Series, expected_rate: float) -> float:
    trials = int(round(float(n.sum())))
    successes = int(round(float((rates * n).sum())))
    if trials == 0:
        return 1.0
    p0 = float(np.clip(expected_rate, 1e-9, 1 - 1e-9))
    return float(stats.binomtest(min(successes, trials), trials, p0).pvalue)


def _spread(values: np.ndarray, robust: bool) -> float:
    if robust:
        return float(1.4826 * np.median(np.abs(values - np.median(values))))
    return float(np.std(values, ddof=1))
