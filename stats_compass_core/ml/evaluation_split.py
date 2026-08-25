"""Selecting which rows a model evaluation should score.

Training tools write predictions for every row and record which split each row
belonged to in a `<target>_split` column. Scoring that frame wholesale grades the
model largely on rows it memorised, which silently inflates every metric — and
inflates them in the direction nobody questions.

The default here is therefore the holdout, not everything. Callers who genuinely
want the full set have to say so.
"""

from __future__ import annotations

import pandas as pd

SPLIT_SUFFIX = "_split"
TRAIN = "train"
TEST = "test"


def default_split_column(target_column: str) -> str:
    """The column name training tools use to record the split."""
    return f"{target_column}{SPLIT_SUFFIX}"


def resolve_split_column(
    df: pd.DataFrame, target_column: str, split_column: str | None
) -> str | None:
    """Find the split column to filter on, or None if there isn't one.

    An explicitly named column that is missing is an error worth raising: the
    caller asked for a holdout evaluation and would otherwise silently get a
    full-set one, which is the failure this module exists to prevent.
    """
    if split_column:
        if split_column not in df.columns:
            raise ValueError(
                f"Split column '{split_column}' not found in DataFrame. "
                "Without it the evaluation would score training rows as well, "
                "which inflates every metric."
            )
        return split_column

    inferred = default_split_column(target_column)
    return inferred if inferred in df.columns else None


def select_rows(
    df: pd.DataFrame,
    target_column: str,
    split_column: str | None,
    evaluate_on: str,
) -> tuple[pd.DataFrame, str]:
    """Return the rows to score and a label describing them.

    Returns ("all") when no split information exists — a frame of predictions
    made without a holdout, which is a legitimate thing to score as long as the
    result says so.
    """
    resolved = resolve_split_column(df, target_column, split_column)

    if resolved is None or evaluate_on == "all":
        return df, "all"

    rows = df[df[resolved] == evaluate_on]
    if rows.empty:
        # Better to score everything and say so than to fail outright, but the
        # caller must not be told this was a holdout evaluation.
        return df, "all"

    return rows, evaluate_on


def train_rows(
    df: pd.DataFrame, target_column: str, split_column: str | None
) -> pd.DataFrame | None:
    """The training rows, for reporting the train/test gap. None if unavailable."""
    resolved = resolve_split_column(df, target_column, split_column)
    if resolved is None:
        return None
    rows = df[df[resolved] == TRAIN]
    return rows if not rows.empty else None
