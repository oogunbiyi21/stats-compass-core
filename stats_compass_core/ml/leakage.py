"""Leakage detection on the training split.

A feature that on its own predicts the target perfectly is almost never a
discovery. It is usually a column written after the outcome — a cancellation
reason that exists only for customers who left, a refund amount that exists
only for returned orders. Nothing about the model's scores looks wrong: the
holdout is 100% because the holdout has the same column. So the check is on
the feature, not the score.

Two signals, each checked one feature at a time on the training rows only:

- Null pattern: whether the value is missing at all separates the classes.
- Value: one threshold per class boundary separates the classes (classifier),
  or the feature is a monotone function of the target (regressor).

The value check fits a tree capped at one leaf per class. The cap is what keeps
it honest: an uncapped tree fits any column with a distinct value per row
perfectly, so every ID and every continuous measurement would fire.
"""

import numpy as np
import pandas as pd

from stats_compass_core.results import ToolWarning

LEAKAGE_SUSPECTED = "LEAKAGE_SUSPECTED"

# Below this many training rows a perfect split is too easy to get by chance.
MIN_ROWS = 20

# Rank correlation at or above this is treated as the target in another unit.
MONOTONE_RHO = 0.999


def detect_leakage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_column: str,
    is_classifier: bool,
) -> list[ToolWarning]:
    """One LEAKAGE_SUSPECTED warning per feature that alone predicts the target."""
    if len(X_train) < MIN_ROWS or y_train.nunique() < 2:
        return []

    warnings: list[ToolWarning] = []
    for col in X_train.columns:
        reason = _why_it_leaks(X_train[col], y_train, is_classifier)
        if reason:
            warnings.append(ToolWarning(
                code=LEAKAGE_SUSPECTED,
                columns=[col],
                message=(
                    f"'{col}' alone predicts '{target_column}' perfectly on the "
                    f"training split ({reason}). A column like this is usually "
                    f"recorded after the outcome; if so, the scores are not real "
                    f"and '{col}' must not be a feature."
                ),
            ))
    return warnings


def _why_it_leaks(feature: pd.Series, y: pd.Series, is_classifier: bool) -> str | None:
    missing = feature.isna()

    if is_classifier and 0 < missing.sum() < len(feature):
        when_missing = y[missing].unique()
        when_present = y[~missing].unique()
        if (
            len(when_missing) == 1
            and len(when_present) == 1
            and when_missing[0] != when_present[0]
        ):
            return f"it is missing exactly when the target is '{when_missing[0]}'"

    present = feature[~missing]
    y_present = y[~missing]
    if len(present) < MIN_ROWS or present.nunique() < 2 or y_present.nunique() < 2:
        return None

    try:
        values = present.astype(float)
    except (TypeError, ValueError):
        return None

    if is_classifier:
        return _separates_classes(values, y_present)
    return _monotone_in_target(values, y_present)


def _separates_classes(values: pd.Series, y: pd.Series) -> str | None:
    from sklearn.tree import DecisionTreeClassifier

    n_classes = y.nunique()
    tree = DecisionTreeClassifier(max_leaf_nodes=max(2, n_classes), random_state=0)
    X = values.to_numpy().reshape(-1, 1)
    tree.fit(X, y)
    if tree.score(X, y) == 1.0:
        return f"{n_classes - 1} threshold(s) on its value separate every class"
    return None


def _monotone_in_target(values: pd.Series, y: pd.Series) -> str | None:
    rho = values.rank().corr(y.astype(float).rank())
    if np.isfinite(rho) and abs(rho) >= MONOTONE_RHO:
        return f"its rank correlation with the target is {rho:.4f}"
    return None
