"""
Tool for evaluating read-only pandas expressions to inspect data.
"""

import re
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState


class InspectDataInput(StrictToolInput):
    """Input schema for inspect_data tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to inspect. Uses active if not specified.",
    )
    expression: str = Field(
        description=(
            "Python/Pandas expression to evaluate. "
            "Examples: "
            "1. 'df[\"col\"].unique()' "
            "2. 'df[\"col\"].mean()' "
            "3. 'len(df[df[\"col\"] > 5])' "
            "4. 'df.groupby(\"cat\")[\"val\"].sum()'"
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
    
    # Security check for dangerous patterns
    dangerous_patterns = [
        r"\bimport\b",
        r"\bexec\b",
        r"\beval\b",
        r"\bopen\b",
        r"\.to_csv",
        r"\.to_excel",
        r"\.to_pickle",
        r"\b__.*?__\b",
        r"\bdel\b",
        r"=",  # Prevent assignment/modification
    ]
    if any(re.search(p, params.expression, re.IGNORECASE) for p in dangerous_patterns):
        raise ValueError("Unsafe operation or assignment detected. This tool is for read-only inspection.")

    # Create namespace with safe globals and column variables
    namespace = {
        "df": df,
        "pd": pd,
        "np": np,
        "__builtins__": {},  # Restrict builtins
    }

    # Add valid column names to namespace for convenience
    for col in df.columns:
        if isinstance(col, str) and col.isidentifier():
            namespace[col] = df[col]

    try:
        # Evaluate expression
        result = eval(params.expression, namespace)
        
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
            f"Available variables: df, pd, np, and columns."
        ) from e
