"""
Tools for common hypothesis tests (t-test and z-test).
"""

from __future__ import annotations

import math

from pydantic import Field
from scipy import stats

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import HypothesisTestResult, ToolWarning
from stats_compass_core.state import DataFrameState

# Student's t assumes equal variances. Past this ratio of the larger sample
# variance to the smaller its p-value is unreliable; Welch's is not affected.
MAX_VARIANCE_RATIO = 4.0

# Below this many observations a z-test on sample standard deviations
# understates uncertainty: the p-value is too small. A t-test is the right test.
MIN_Z_SAMPLE = 30


class TTestInput(StrictToolInput):
    """Input schema for two-sample t-test."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    column_a: str = Field(description="First sample column")
    column_b: str = Field(description="Second sample column")
    alternative: str = Field(
        default="two-sided",
        pattern="^(two-sided|less|greater)$",
        description="Alternative hypothesis",
    )
    equal_var: bool = Field(
        default=True, description="Assume equal variances (Student) or not (Welch)"
    )


class ZTestInput(StrictToolInput):
    """Input schema for two-sample z-test on means."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    column_a: str = Field(description="First sample column")
    column_b: str = Field(description="Second sample column")
    population_std_a: float | None = Field(
        default=None, gt=0, description="Known population std for sample A (optional)"
    )
    population_std_b: float | None = Field(
        default=None, gt=0, description="Known population std for sample B (optional)"
    )
    alternative: str = Field(
        default="two-sided",
        pattern="^(two-sided|less|greater)$",
        description="Alternative hypothesis",
    )


@registry.register(
    category="eda",
    input_schema=TTestInput,
    description="Two-sample t-test (Student or Welch)",
)
def t_test(state: DataFrameState, params: TTestInput) -> HypothesisTestResult:
    """
    Perform a two-sample t-test between two columns.

    Args:
        state: DataFrameState containing the DataFrame to analyze
        params: Test parameters

    Returns:
        HypothesisTestResult with t statistic and p-value

    Raises:
        ValueError: If columns are missing or contain insufficient data
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    for col in (params.column_a, params.column_b):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    a = df[params.column_a].dropna().to_numpy()
    b = df[params.column_b].dropna().to_numpy()

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Need at least 2 observations in each sample for t-test")

    statistic, p_value = stats.ttest_ind(
        a, b, equal_var=params.equal_var, alternative=params.alternative
    )
    if math.isnan(p_value):
        # Both samples constant: there is no variance to test against. A NaN
        # p-value is never below 0.05, so it used to read as "no difference".
        raise ValueError(
            f"t-test is undefined: '{params.column_a}' and '{params.column_b}' "
            f"have no variation (means {a.mean():g} and {b.mean():g})."
        )

    warnings: list[ToolWarning] = []
    var_a, var_b = float(a.var(ddof=1)), float(b.var(ddof=1))
    ratio = max(var_a, var_b) / min(var_a, var_b) if min(var_a, var_b) > 0 else math.inf
    if params.equal_var and ratio > MAX_VARIANCE_RATIO:
        warnings.append(ToolWarning(
            code="UNEQUAL_VARIANCE",
            columns=[params.column_a, params.column_b],
            message=(
                f"Student's t-test assumes equal variances, but one sample's "
                f"variance is {ratio:.1f}x the other's (limit {MAX_VARIANCE_RATIO:g}), "
                f"so the p-value is unreliable. Use equal_var=False (Welch)."
            ),
        ))
    state.record_warnings(source_name, warnings)

    return HypothesisTestResult(
        test_type="t-test (Student)" if params.equal_var else "t-test (Welch)",
        statistic=float(statistic),
        p_value=float(p_value),
        alternative=params.alternative,
        n_a=len(a),
        n_b=len(b),
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
        dataframe_name=source_name,
        details={
            "column_a": params.column_a,
            "column_b": params.column_b,
            "equal_var": params.equal_var,
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "std_a": float(a.std(ddof=1)),
            "std_b": float(b.std(ddof=1)),
        },
        warnings=warnings,
    )


@registry.register(
    category="eda",
    input_schema=ZTestInput,
    description="Two-sample z-test for difference in means",
)
def z_test(state: DataFrameState, params: ZTestInput) -> HypothesisTestResult:
    """
    Perform a two-sample z-test for difference in means.

    Args:
        state: DataFrameState containing the DataFrame to analyze
        params: Test parameters, including optional known population std values

    Returns:
        HypothesisTestResult with z statistic and p-value

    Raises:
        ValueError: If columns are missing or contain insufficient data
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    for col in (params.column_a, params.column_b):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    a = df[params.column_a].dropna().to_numpy()
    b = df[params.column_b].dropna().to_numpy()

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Need at least 2 observations in each sample for z-test")

    std_a = params.population_std_a or float(a.std(ddof=0))
    std_b = params.population_std_b or float(b.std(ddof=0))

    warnings: list[ToolWarning] = []
    estimated = [
        (col, n) for col, n, known in (
            (params.column_a, len(a), params.population_std_a),
            (params.column_b, len(b), params.population_std_b),
        )
        if known is None and n < MIN_Z_SAMPLE
    ]
    if estimated:
        warnings.append(ToolWarning(
            code="SMALL_SAMPLE",
            columns=[col for col, _ in estimated],
            message=(
                f"A z-test with standard deviations estimated from "
                f"{' and '.join(f'{n} rows' for _, n in estimated)} (fewer than "
                f"{MIN_Z_SAMPLE}) gives p-values that are too small. Use t_test, "
                f"or pass the known population standard deviations."
            ),
        ))
    state.record_warnings(source_name, warnings)

    se = math.sqrt((std_a ** 2) / len(a) + (std_b ** 2) / len(b))
    if se == 0:
        raise ValueError("Pooled standard error is zero; z-test undefined.")

    z_stat = (float(a.mean()) - float(b.mean())) / se

    if params.alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif params.alternative == "greater":
        p_value = 1 - stats.norm.cdf(z_stat)
    else:  # less
        p_value = stats.norm.cdf(z_stat)

    return HypothesisTestResult(
        test_type="z-test",
        statistic=float(z_stat),
        p_value=float(p_value),
        alternative=params.alternative,
        n_a=len(a),
        n_b=len(b),
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
        dataframe_name=source_name,
        details={
            "column_a": params.column_a,
            "column_b": params.column_b,
            "population_std_a": std_a,
            "population_std_b": std_b,
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
        },
        warnings=warnings,
    )
