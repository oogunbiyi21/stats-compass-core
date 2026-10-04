"""Rows that a grouping silently leaves out.

pandas drops rows whose group key is missing. For "revenue by discount code"
that is every order without a code, and the per-group totals no longer add up
to the whole with nothing to say why.
"""

import pandas as pd

from stats_compass_core.results import ToolWarning


def null_key_warning(df: pd.DataFrame, keys: list[str]) -> ToolWarning | None:
    missing = df[keys].isna().any(axis=1)
    if not missing.any():
        return None
    columns = [k for k in keys if df[k].isna().any()]
    return ToolWarning(
        code="NULL_GROUP_KEYS",
        columns=columns,
        message=(
            f"{int(missing.sum())} of {len(df)} rows have no value in {columns} "
            f"and are in no group, so the groups do not add up to the whole. "
            f"Fill the key (e.g. with 'none') to keep them."
        ),
    )
