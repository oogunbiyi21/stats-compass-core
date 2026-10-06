"""repeat_rate and median_gap: customer statistics with their uncertainty."""

import numpy as np
import pytest

from stats_compass_core.series import (
    GapFacts,
    Insufficient,
    RepeatRateFacts,
    median_gap,
    repeat_rate,
)


def _cohort(n: int, rate: float, seed: int = 0) -> list[int]:
    """Order counts in the window: 1 for a single order, 2+ for a repeat."""
    rng = np.random.default_rng(seed)
    repeats = rng.random(n) < rate
    return [int(1 + rng.integers(1, 4)) if r else 1 for r in repeats]


class TestRepeatRate:
    def test_estimate_is_the_share_with_a_second_order(self):
        facts = repeat_rate([1, 1, 2, 3, 1] * 40)
        assert isinstance(facts, RepeatRateFacts)
        assert facts.estimate == pytest.approx(0.4)
        assert (facts.n_customers, facts.n_repeat) == (200, 80)
        lo, hi = facts.interval
        assert lo <= facts.estimate <= hi

    def test_too_few_customers_is_counted_in_customers(self):
        result = repeat_rate([1, 2] * 20)
        assert isinstance(result, Insufficient)
        assert (result.needs, result.has, result.unit, result.reason) == (
            100,
            40,
            "customers",
            "TOO_FEW_CUSTOMERS",
        )

    def test_exact_interval_and_no_seed(self):
        """Clopper-Pearson: exact, nothing drawn, so nothing to seed."""
        counts = _cohort(500, 0.25)
        assert repeat_rate(counts).interval == repeat_rate(counts).interval
        with pytest.raises(TypeError):
            repeat_rate(counts, seed=1)  # type: ignore[call-arg]

    def test_no_repeats_is_not_certainty(self):
        """A bootstrap of 0 out of 100 gave [0, 0]."""
        facts = repeat_rate([1] * 100, level=0.9)
        assert facts.interval[0] == 0
        assert facts.interval[1] == pytest.approx(1 - 0.05 ** (1 / 100), rel=1e-6)

    def test_interval_covers_the_true_rate_at_about_the_level(self):
        covered = 0
        trials = 200
        for i in range(trials):
            facts = repeat_rate(_cohort(400, 0.25, seed=i), level=0.9)
            covered += facts.interval[0] <= 0.25 <= facts.interval[1]
        assert 0.86 <= covered / trials <= 0.99, (
            covered / trials
        )  # exact: at or above the level

    def test_no_comparison_no_verdict(self):
        facts = repeat_rate(_cohort(500, 0.25))
        assert facts.verdict is None and facts.p_value is None

    def test_the_november_cohort_against_every_other(self):
        """Brief §3.1: 925 customers at 12.86% against 12,436 at 24.66%."""
        november = [2] * 119 + [1] * (925 - 119)
        facts = repeat_rate(november, comparison_rate=0.2466, comparison_n=12436)
        assert facts.estimate == pytest.approx(0.1286, abs=1e-3)
        assert facts.verdict == "real change"
        assert facts.p_value < 0.001

    def test_a_cohort_at_the_usual_rate_is_expected(self):
        same = [2] * 247 + [1] * (1000 - 247)
        facts = repeat_rate(same, comparison_rate=0.2466, comparison_n=12436)
        assert facts.verdict == "expected"


class TestMedianGap:
    def test_median_and_interquartile_range(self):
        facts = median_gap(list(range(1, 101)))
        assert isinstance(facts, GapFacts)
        assert facts.estimate == pytest.approx(50.5)
        assert facts.iqr == pytest.approx((25.75, 75.25))
        assert facts.n == 100

    def test_too_few_gaps(self):
        result = median_gap([10.0, 20.0])
        assert isinstance(result, Insufficient)
        assert (result.reason, result.needs, result.has) == ("TOO_FEW_GAPS", 20, 2)
