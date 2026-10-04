"""When a 'categorical' column is really an identifier.

Binning and target-encoding both assume a column's values repeat. An ID does
not: every value is rare, so binning collapses the column to one label and
encoding turns that label into a constant, and the constant goes to the model
as a feature. Nothing fails. The check below is shared so every tool that bins
or encodes refuses the same columns for the same reason.
"""

import pandas as pd

from stats_compass_core.results import ToolWarning

HIGH_CARDINALITY = "HIGH_CARDINALITY"

# A column with more unique values than this is not treated as categorical...
MAX_UNIQUE = 200
# ...nor is one where more than this fraction of its non-null values are distinct.
MAX_UNIQUE_FRACTION = 0.5


def high_cardinality_reason(series: pd.Series) -> str | None:
    """Why this column is too high-cardinality to bin or encode, or None.

    The fraction is of non-null rows: a sparse column whose filled values are
    all distinct is still an identifier.
    """
    non_null = int(series.notna().sum())
    if non_null == 0:
        return None
    n_unique = int(series.nunique(dropna=True))
    if n_unique > MAX_UNIQUE:
        return f"{n_unique} unique values (limit {MAX_UNIQUE})"
    if n_unique > MAX_UNIQUE_FRACTION * non_null:
        return (
            f"{n_unique} unique values in {non_null} non-null rows "
            f"(limit {MAX_UNIQUE_FRACTION:.0%})"
        )
    return None


def high_cardinality_warning(column: str, reason: str, action: str) -> ToolWarning:
    return ToolWarning(
        code=HIGH_CARDINALITY,
        columns=[column],
        message=(
            f"'{column}' looks like an identifier, not a category: {reason}. "
            f"It was {action}. Treating it as a category would collapse it to a "
            f"constant; drop it, or derive a real category from it."
        ),
    )
