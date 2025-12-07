"""
Tool for evaluating classification model predictions.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState
from stats_compass_core.results import ClassificationEvaluationResult


class EvaluateClassificationInput(BaseModel):
    """Input schema for evaluate_classification_model tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    target_column: str = Field(description="Name of the true label column")
    prediction_column: str = Field(description="Name of the predicted label column")
    drop_na: bool = Field(
        default=True, description="Drop rows with missing target or prediction"
    )
    average: str = Field(
        default="weighted",
        pattern="^(micro|macro|weighted|binary)$",
        description="Averaging strategy for precision/recall/f1",
    )


@registry.register(
    category="ml",
    input_schema=EvaluateClassificationInput,
    description="Compute accuracy, precision, recall, f1, and confusion matrix",
)
def evaluate_classification_model(
    state: DataFrameState, params: EvaluateClassificationInput
) -> ClassificationEvaluationResult:
    """
    Evaluate classification predictions using scikit-learn metrics.

    Note: Requires scikit-learn installed (ml extra).

    Args:
        state: DataFrameState containing the DataFrame with predictions
        params: Evaluation parameters

    Returns:
        ClassificationEvaluationResult with metrics and confusion matrix

    Raises:
        ImportError: If scikit-learn is not installed
        ValueError: If columns are missing or no data available
    """
    try:
        from sklearn import metrics
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for classification evaluation. "
            "Install with: pip install stats-compass-core[ml]"
        ) from exc

    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state._active_dataframe

    for col in (params.target_column, params.prediction_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    data = df[[params.target_column, params.prediction_column]]
    if params.drop_na:
        data = data.dropna()

    if data.empty:
        raise ValueError("No rows available after dropping missing values")

    y_true = data[params.target_column]
    y_pred = data[params.prediction_column]

    labels = sorted(y_true.unique().tolist())
    average = params.average
    # For binary targets with non-string labels, sklearn expects 'binary' only when 2 classes
    if average == "binary" and len(labels) != 2:
        raise ValueError("binary average requires exactly 2 unique classes")

    accuracy = float(metrics.accuracy_score(y_true, y_pred))
    precision = float(metrics.precision_score(
        y_true, y_pred, average=average, zero_division=0
    ))
    recall = float(metrics.recall_score(y_true, y_pred, average=average, zero_division=0))
    f1 = float(metrics.f1_score(y_true, y_pred, average=average, zero_division=0))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)

    return ClassificationEvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm.tolist(),
        labels=labels,
        n_samples=len(data),
        average=average,
        dataframe_name=source_name,
        target_column=params.target_column,
        prediction_column=params.prediction_column,
    )
