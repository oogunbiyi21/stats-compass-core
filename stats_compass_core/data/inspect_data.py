"""
Tool for evaluating read-only pandas expressions to inspect data.
"""

from typing import Any

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState
from stats_compass_core.utils.safe_expr import evaluate


class InspectDataInput(StrictToolInput):
    """Input schema for inspect_data tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to inspect. Uses active if not specified.",
    )
    expression: str = Field(
        description=(
            "Read-only expression on the columns. "
            "Examples: "
            "1. 'df[\"col\"].unique()' "
            "2. 'col.mean()' "
            "3. 'len(df[df[\"col\"] > 5])' "
            "4. 'region.value_counts()' "
            "Columns, constants, arithmetic, comparisons, listed functions and column "
            "summaries (mean, median, sum, min, max, std, count, nunique, unique, "
            "value_counts, describe, head, tail). Use groupby_aggregate for group sums."
        )
    )


@registry.register(
    category="data",
    input_schema=InspectDataInput,
    description="Evaluate read-only pandas expressions to inspect data (e.g., check unique values, calculate specific stats)",
)
def inspect_data(state: DataFrameState, params: InspectDataInput) -> dict[str, Any]:
    """
    Evaluate a read-only pandas expression.

    Args:
        state: DataFrameState containing the DataFrame
        params: Parameters containing the expression

    Returns:
        Dictionary containing the result of the evaluation
    """
    df = state.get_dataframe(params.dataframe_name)

    # Read by utils.safe_expr, never by pd.eval: pandas' own evaluator reached
    # file readers through pd and np (security scan F3, 8 Oct 2026).
    try:
        result = evaluate(params.expression, df)

        # Format result for output
        if isinstance(result, (pd.DataFrame, pd.Series)):
            # For large results, truncate
            if len(result) > 20:
                result_str = result.head(20).to_string() + f"\n\n... ({len(result) - 20} more rows)"
            else:
                result_str = result.to_string()

            return {
                "result_type": type(result).__name__,
                "result_text": result_str,
                "shape": result.shape,
            }
        else:
            # For scalars (int, float, list, etc.)
            return {
                "result_type": type(result).__name__,
                "result": str(result),
            }

    except Exception as e:
        raise ValueError(
            f"Invalid expression '{params.expression}': {str(e)}. "
            f"Available: df, the columns, listed functions and column summaries."
        ) from e
