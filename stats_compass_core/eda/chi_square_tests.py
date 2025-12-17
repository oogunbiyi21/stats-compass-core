"""
Chi-square statistical tests for categorical data analysis.

Provides:
- Chi-square test of independence (two categorical variables)
- Chi-square goodness of fit test (one variable vs expected distribution)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field
from scipy import stats

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import ChiSquareResult
from stats_compass_core.state import DataFrameState


class ChiSquareIndependenceInput(StrictToolInput):
    """Input schema for chi-square test of independence."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    column1: str = Field(description="First categorical column (row variable)")
    column2: str = Field(description="Second categorical column (column variable)")
    alpha: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Significance level for the test",
    )


class ChiSquareGoodnessOfFitInput(StrictToolInput):
    """Input schema for chi-square goodness of fit test."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    column: str = Field(description="Categorical column to test")
    expected_frequencies: list[float] | None = Field(
        default=None,
        description="Expected frequencies (proportions). If None, assumes uniform distribution.",
    )
    alpha: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Significance level for the test",
    )


def _interpret_cramers_v(v: float) -> str:
    """Interpret Cramér's V effect size."""
    if v < 0.1:
        return "negligible"
    elif v < 0.3:
        return "small"
    elif v < 0.5:
        return "medium"
    else:
        return "large"


def _check_expected_frequencies(expected: np.ndarray) -> tuple[int, str | None]:
    """
    Check for cells with low expected frequency.
    
    Returns:
        Tuple of (count of cells < 5, warning message or None)
    """
    low_count = int((expected < 5).sum())
    total_cells = expected.size

    if low_count == 0:
        return 0, None

    pct = (low_count / total_cells) * 100
    warning = (
        f"{low_count} cells ({pct:.1f}%) have expected frequency < 5. "
        "Consider combining categories or using exact tests for small samples."
    )
    return low_count, warning


@registry.register(
    category="eda",
    input_schema=ChiSquareIndependenceInput,
    description="Chi-square test of independence between two categorical variables",
)
def chi_square_independence(
    state: DataFrameState, params: ChiSquareIndependenceInput
) -> ChiSquareResult:
    """
    Perform chi-square test of independence between two categorical variables.
    
    Tests whether there is a statistically significant association between
    two categorical variables.

    Args:
        state: DataFrameState containing the DataFrame to analyze
        params: Test parameters including column names and alpha level

    Returns:
        ChiSquareResult with test statistics, p-value, and contingency table

    Raises:
        ValueError: If columns are missing, not categorical, or have insufficient data
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    # Validate columns exist
    for col in (params.column1, params.column2):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # Get data and drop rows with missing values in either column
    data = df[[params.column1, params.column2]].dropna()

    if len(data) < 2:
        raise ValueError(
            f"Insufficient data: need at least 2 rows, got {len(data)} after removing missing values"
        )

    # Create contingency table
    contingency_table = pd.crosstab(data[params.column1], data[params.column2])

    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 categories in each column. "
            f"Got {contingency_table.shape[0]} x {contingency_table.shape[1]}"
        )

    # Perform chi-square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Calculate Cramér's V effect size
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = float(np.sqrt(chi2_stat / (n * min_dim))) if min_dim > 0 else 0.0

    # Check for low expected frequencies
    low_count, low_warning = _check_expected_frequencies(expected)

    # Convert tables to JSON-safe format
    observed_dict = contingency_table.to_dict()
    expected_df = pd.DataFrame(
        expected,
        index=contingency_table.index,
        columns=contingency_table.columns
    )
    expected_dict = expected_df.round(2).to_dict()

    return ChiSquareResult(
        test_type="independence",
        chi2_statistic=float(chi2_stat),
        p_value=float(p_value),
        degrees_of_freedom=int(dof),
        n_samples=int(n),
        effect_size=cramers_v,
        effect_interpretation=_interpret_cramers_v(cramers_v),
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
        observed_frequencies=observed_dict,
        expected_frequencies=expected_dict,
        dataframe_name=source_name,
        column1=params.column1,
        column2=params.column2,
        low_expected_count=low_count,
        low_expected_warning=low_warning,
    )


@registry.register(
    category="eda",
    input_schema=ChiSquareGoodnessOfFitInput,
    description="Chi-square goodness of fit test for a single categorical variable",
)
def chi_square_goodness_of_fit(
    state: DataFrameState, params: ChiSquareGoodnessOfFitInput
) -> ChiSquareResult:
    """
    Perform chi-square goodness of fit test.
    
    Tests whether the observed frequency distribution of a categorical variable
    matches an expected distribution (uniform by default).

    Args:
        state: DataFrameState containing the DataFrame to analyze
        params: Test parameters including column name and expected frequencies

    Returns:
        ChiSquareResult with test statistics and frequency comparison

    Raises:
        ValueError: If column is missing or expected frequencies don't match categories
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    # Validate column exists
    if params.column not in df.columns:
        raise ValueError(f"Column '{params.column}' not found in DataFrame")

    # Get observed frequencies
    observed_freq = df[params.column].value_counts().sort_index()
    categories = observed_freq.index.tolist()
    observed_values = observed_freq.values
    n_total = int(observed_values.sum())

    if len(observed_values) < 2:
        raise ValueError(
            f"Need at least 2 categories for goodness of fit test, found {len(observed_values)}"
        )

    # Determine expected frequencies
    if params.expected_frequencies is not None:
        expected_input = params.expected_frequencies

        # Validate expected frequencies
        if any(f <= 0 for f in expected_input):
            raise ValueError("Expected frequencies must all be positive (greater than 0)")

        if len(expected_input) != len(observed_values):
            raise ValueError(
                f"Expected frequencies length ({len(expected_input)}) must match "
                f"number of categories ({len(observed_values)})"
            )

        # Normalize expected frequencies to sum to total observations
        expected_prop = np.array(expected_input) / np.sum(expected_input)
        expected_values = expected_prop * n_total
    else:
        # Assume uniform distribution
        expected_values = np.full(len(observed_values), n_total / len(observed_values))

    # Perform chi-square test
    chi2_stat, p_value = stats.chisquare(observed_values, expected_values)
    dof = len(observed_values) - 1

    # Check for low expected frequencies
    low_count, low_warning = _check_expected_frequencies(expected_values)

    # Create frequency tables as dicts
    observed_dict = {str(cat): int(count) for cat, count in zip(categories, observed_values)}
    expected_dict = {str(cat): round(float(exp), 2) for cat, exp in zip(categories, expected_values)}

    return ChiSquareResult(
        test_type="goodness_of_fit",
        chi2_statistic=float(chi2_stat),
        p_value=float(p_value),
        degrees_of_freedom=int(dof),
        n_samples=n_total,
        effect_size=None,  # No standard effect size for goodness of fit
        effect_interpretation=None,
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
        observed_frequencies=observed_dict,
        expected_frequencies=expected_dict,
        dataframe_name=source_name,
        column1=params.column,
        column2=None,
        low_expected_count=low_count,
        low_expected_warning=low_warning,
    )
