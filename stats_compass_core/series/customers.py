"""Customer statistics with their uncertainty: repeat rate and the gap between orders.

``repeat_rate`` takes one cohort's order counts within its window, each count
including the first order, so a customer repeated when the count is 2 or more.
Which cohort to pass (the newest with a full window, or another) is the
caller's choice; the function does not depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from scipy import stats

from stats_compass_core.series._common import Insufficient

# Below this many expected counts in a cell, compare two rates exactly.
MIN_CELL_FOR_Z_TEST = 5


@dataclass(frozen=True)
class RepeatRateFacts:
    estimate: float
    interval: tuple[float, float]
    n_customers: int
    n_repeat: int
    verdict: Literal["expected", "noise", "real change"] | None
    p_value: float | None
    decided_by: Literal["expected_band", "test"] | None
    test: str | None
    method: str
    params: dict[str, Any]


@dataclass(frozen=True)
class GapFacts:
    """The median gap and the spread of gaps around it.

    ``iqr`` describes how gaps vary between orders, not how certain the median
    is. Gaps from one customer are not independent, so an interval for the
    median would have to resample customers, not gaps.
    """

    estimate: float  # median days between a customer's consecutive orders
    iqr: tuple[float, float]
    n: int


def repeat_rate(
    order_counts: Sequence[int],
    *,
    level: float = 0.9,
    min_customers: int = 100,
    comparison_rate: float | None = None,
    comparison_n: int | None = None,
    seasonal_band_z: float = 2.0,
    real_change_alpha: float = 0.05,
) -> RepeatRateFacts | Insufficient:
    """The share of a cohort that ordered again within its window.

    Args:
        order_counts: Each cohort member's orders within the window, the first
            order included.
        level: Interval level. The interval is Clopper-Pearson: exact for a
            share of independent customers, so nothing is drawn. (Resampling
            customers gives the same distribution, but drawn, and it claimed
            certainty, [0, 0], for a cohort with no repeats.)
        min_customers: Fewest customers for an estimate (the count floor; the
            time floor is the caller's).
        comparison_rate: The rate to judge against, for example the store's
            other cohorts. It must not include the cohort being judged, or
            the cohort is partly compared with itself. Without it there is no
            verdict.
        comparison_n: Customers behind ``comparison_rate``. With it the test
            compares two rates; without it, one rate against a fixed value.
        seasonal_band_z: Width of the expected band, in standard errors of
            ``comparison_rate`` at this cohort's size.
        real_change_alpha: The significance level.

    Returns:
        ``RepeatRateFacts`` or ``Insufficient(unit="customers",
        reason="TOO_FEW_CUSTOMERS")``.
    """
    counts = np.asarray(order_counts, dtype=int)
    n = len(counts)
    if n < min_customers:
        return Insufficient(
            needs=min_customers, has=n, unit="customers", reason="TOO_FEW_CUSTOMERS"
        )
    repeats = int((counts >= 2).sum())
    estimate = repeats / n

    tail = (1 - level) / 2
    low = 0.0 if repeats == 0 else float(stats.beta.ppf(tail, repeats, n - repeats + 1))
    high = (
        1.0
        if repeats == n
        else float(stats.beta.ppf(1 - tail, repeats + 1, n - repeats))
    )
    interval = (low, high)

    verdict = p_value = decided_by = test = None
    if comparison_rate is not None:
        p0 = float(np.clip(comparison_rate, 1e-9, 1 - 1e-9))
        se = np.sqrt(p0 * (1 - p0) / n)
        test, p_value = _compare(repeats, n, p0, comparison_n)
        if abs(estimate - p0) <= seasonal_band_z * se:
            verdict, decided_by = "expected", "expected_band"
        elif p_value < real_change_alpha:
            verdict, decided_by = "real change", "test"
        else:
            verdict, decided_by = "noise", "test"

    return RepeatRateFacts(
        estimate=float(estimate),
        interval=interval,
        n_customers=n,
        n_repeat=repeats,
        verdict=verdict,
        p_value=None if p_value is None else float(p_value),
        decided_by=decided_by,
        test=test,
        method="clopper_pearson",
        params={
            "level": level,
            "min_customers": min_customers,
            "comparison_rate": comparison_rate,
            "comparison_n": comparison_n,
            "seasonal_band_z": seasonal_band_z,
            "real_change_alpha": real_change_alpha,
        },
    )


def _compare(
    repeats: int, n: int, p0: float, comparison_n: int | None
) -> tuple[str, float]:
    if comparison_n is None:
        return "binomial", float(stats.binomtest(repeats, n, p0).pvalue)
    other_repeats = int(round(p0 * comparison_n))
    table = np.array(
        [[repeats, n - repeats], [other_repeats, comparison_n - other_repeats]]
    )
    expected = (
        table.sum(axis=1, keepdims=True)
        * table.sum(axis=0, keepdims=True)
        / table.sum()
    )
    if expected.min() < MIN_CELL_FOR_Z_TEST:
        return "fisher_exact", float(stats.fisher_exact(table).pvalue)
    pooled = (repeats + other_repeats) / (n + comparison_n)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n + 1 / comparison_n))
    z = (repeats / n - other_repeats / comparison_n) / se
    return "two_proportion_z", float(2 * stats.norm.sf(abs(z)))


def median_gap(
    gaps_days: Sequence[float], *, min_gaps: int = 20
) -> GapFacts | Insufficient:
    """Median days between consecutive orders, with its interquartile range."""
    gaps = np.asarray(gaps_days, dtype=float)
    gaps = gaps[np.isfinite(gaps)]
    if len(gaps) < min_gaps:
        return Insufficient(
            needs=min_gaps, has=len(gaps), unit="orders", reason="TOO_FEW_GAPS"
        )
    q25, median, q75 = np.percentile(gaps, [25, 50, 75])
    return GapFacts(
        estimate=float(median), iqr=(float(q25), float(q75)), n=int(len(gaps))
    )
