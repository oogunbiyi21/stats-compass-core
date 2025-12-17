"""
Tools for common hypothesis tests (t-test and z-test).
"""

from __future__ import annotations

import math

from pydantic import Field
from scipy import stats

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import HypothesisTestResult
from stats_compass_core.state import DataFrameState


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
    )
