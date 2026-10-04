"""Values that are text but mean something else.

Exports write missing values as "nan", "null" or "" and numbers as "10". pandas
keeps both as strings, so a missing-data report counts them as present and a
numeric summary leaves the column out. Shared here so the tool that converts
them (convert_dtype) and the tools that should notice them agree on what they
look like.
"""

import pandas as pd

# Matched after trimming whitespace and lower-casing.
NULL_TOKENS = ["", "nan", "null", "none", "na", "n/a", "nat", "-"]

# A text column is "numbers as text" when at least this share of its
# non-missing values parse as numbers.
NUMERIC_SHARE = 0.9


def _is_text(series: pd.Series) -> bool:
    return series.dtype == "object" or pd.api.types.is_string_dtype(series)


def missing_as_text(series: pd.Series, tokens: list[str] = NULL_TOKENS) -> int:
    """How many values in a text column are a missing marker spelled as text."""
    if not _is_text(series):
        return 0
    lowered = {t.strip().lower() for t in tokens}
    text = series.dropna().map(
        lambda v: v.strip().lower() if isinstance(v, str) else None
    )
    return int(text.isin(lowered).sum())


def numeric_as_text(series: pd.Series) -> bool:
    """Whether a text column is really numbers stored as strings."""
    if not _is_text(series):
        return False
    text = series.dropna().map(lambda v: v.strip() if isinstance(v, str) else v)
    text = text[~text.map(lambda v: isinstance(v, str) and v.lower() in NULL_TOKENS)]
    if text.empty:
        return False
    parsed = pd.to_numeric(text, errors="coerce")
    return bool(parsed.notna().mean() >= NUMERIC_SHARE)
