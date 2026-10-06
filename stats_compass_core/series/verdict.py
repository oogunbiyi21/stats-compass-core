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

- money: the period's deviation from expectation against ``n_boot`` seeded
  draws of historical remainder windows of the same length.
- count: Poisson, or negative binomial when the history's counts vary more
  than a Poisson would, with the dispersion measured from that history.
- ratio: binomial on the period's successes out of its denominators.
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
    ]  # (absolute, percent); percent None when prior is 0
    p_value: float
    n_boot: int | None
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
    seed: int,
    weights: pd.Series | None = None,
    aggregate: Aggregate = "sum",
    seasonal_band_z: float = 2.0,
    noise_band_z: float = 3.0,
    real_change_alpha: float = 0.05,
    n_boot: int = 2000,
    min_points: int = 5,
    robust_scale: bool = False,
) -> VerdictFacts | Insufficient:
    """Judge one period's figure against the history before it.

    Args:
        dec: A decomposition of the history that ends the day before
            ``period`` starts.
        period: The period's daily values; NaN means no data.
        kind: ``"count"``, ``"money"`` or ``"ratio"``. Chooses the test.
        seed: Seeds the money bootstrap. Required so that a run can be
            repeated exactly.
        weights: Daily denominators (``n``), over at least the prior period
            and the period. Required for ``ratio``; weights a ``mean``.
        aggregate: ``"sum"`` for totals, ``"mean"`` for averages and rates.
        seasonal_band_z, noise_band_z, real_change_alpha: The calibration.
        n_boot: Bootstrap draws for the money test.
        min_points: Fewest days with data for the period to be judged.
        robust_scale: Measure both bands' spreads with 1.4826 × the median
            absolute deviation instead of the standard deviation, so that the
            store's own past spikes do not widen them. Recorded as
            ``params["scale"]``.

    Returns:
        ``VerdictFacts``, or ``Insufficient`` (``TOO_FEW_POINTS``,
        ``NO_DENOMINATOR``).

    Raises:
        ValueError: if ``dec`` does not end the day before ``period`` starts,
            or the arguments contradict each other.
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
    if start != history_end + pd.Timedelta(days=1):
        raise ValueError(
            f"the period must start the day after the decomposition ends "
            f"({(history_end + pd.Timedelta(days=1)).date()}), not {start.date()}"
        )
    length = len(period)

    params: dict[str, Any] = {
        "kind": kind,
        "seed": seed,
        "aggregate": aggregate,
        "seasonal_band_z": seasonal_band_z,
        "noise_band_z": noise_band_z,
        "real_change_alpha": real_change_alpha,
        "n_boot": n_boot,
        "min_points": min_points,
        "scale": "mad" if robust_scale else "std",
        "period_start": start.date().isoformat(),
        "period_end": end.date().isoformat(),
        "decomposition_method": dec.method,
    }

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

    expected_daily = dec.expected_daily(start.date(), end.date())
    period_w = _weights_for(weights, period.index)
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

    rng = np.random.default_rng(seed)
    if kind == "money":
        test, p_value, draws = (
            "block_bootstrap",
            _bootstrap_p(dec, length, observed - expected, scale, n_boot, rng),
            n_boot,
        )
    elif kind == "count":
        test, p_value = _count_p(dec, length, observed, expected)
        draws = None
        params["dispersion"] = _dispersion(dec, length)
    else:
        test, p_value, draws = (
            "binomial",
            _binomial_p(period[present], period_w[present], expected),
            None,
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
        n_boot=draws,
        test=test,
        decided_by=decided_by,
        method=f"{dec.method}+{test}",
        params=params,
        warnings=[],
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
    if dec.grain == "day":
        adjusted = (dec.observed - dec.seasonal).to_numpy()
        blocks = len(adjusted) // length
        totals = (
            adjusted[len(adjusted) - blocks * length :]
            .reshape(blocks, length)
            .sum(axis=1)
        )
        return _spread(np.diff(totals), robust) if blocks > 2 else float("nan")
    weeks = max(1, round(length / 7))
    adjusted = (dec.observed - dec.seasonal).to_numpy()
    blocks = len(adjusted) // weeks
    totals = (
        adjusted[len(adjusted) - blocks * weeks :].reshape(blocks, weeks).sum(axis=1)
    )
    return _spread(np.diff(totals), robust) * length / (7 * weeks)


def _remainder_windows(dec: Decomposition, length: int) -> np.ndarray:
    """Every historical remainder total over a window of the period's length."""
    if dec.grain == "day":
        return dec.remainder.rolling(length).sum().dropna().to_numpy()
    weeks = max(1, round(length / 7))
    return (
        dec.remainder.rolling(weeks).sum().dropna() * length / (7 * weeks)
    ).to_numpy()


def _bootstrap_p(dec, length, deviation, scale, n_boot, rng) -> float:
    windows = _remainder_windows(dec, length) * scale
    draws = windows[rng.integers(0, len(windows), size=n_boot)]
    exceed = int(np.sum(np.abs(draws) >= abs(deviation)))
    return (exceed + 1) / (n_boot + 1)


def _dispersion(dec: Decomposition, length: int) -> float:
    """Variance of period totals over their mean, from the history."""
    windows = _remainder_windows(dec, length)
    level = (
        (dec.trend + dec.seasonal).rolling(length).sum().dropna()
        if dec.grain == "day"
        else ((dec.trend + dec.seasonal) * length / 7)
    )
    mean = float(level.mean())
    return float(windows.var(ddof=1) / mean) if mean > 0 else float("inf")


def _count_p(dec, length, observed, expected) -> tuple[str, float]:
    mu = max(float(expected), 1e-9)
    count = int(round(observed))
    phi = _dispersion(dec, length)
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
