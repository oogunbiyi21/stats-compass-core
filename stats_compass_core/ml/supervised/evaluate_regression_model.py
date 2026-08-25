"""
Tool for evaluating regression model predictions.
"""

from __future__ import annotations

import math

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import RegressionEvaluationResult
from stats_compass_core.ml.evaluation_split import select_rows, train_rows
from stats_compass_core.state import DataFrameState


class EvaluateRegressionInput(StrictToolInput):
    """Input schema for evaluate_regression_model tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    target_column: str = Field(description="Name of the true target column")
    prediction_column: str = Field(description="Name of the prediction column")
    drop_na: bool = Field(
        default=True, description="Drop rows with missing target or prediction"
    )
    split_column: str | None = Field(
        default=None,
        description=(
            "Column recording which rows were training vs test data. Defaults to "
            "'<target_column>_split', which the training tools create."
        ),
    )
    evaluate_on: str = Field(
        default="test",
        pattern="^(test|train|all)$",
        description=(
            "Which rows to score. Defaults to 'test': scoring 'all' includes rows "
            "the model trained on, which inflates every metric."
        ),
    )


@registry.register(
    category="ml",
    input_schema=EvaluateRegressionInput,
    description="Compute RMSE, MAE, and R^2 for regression predictions",
)
def evaluate_regression_model(
    state: DataFrameState, params: EvaluateRegressionInput
) -> RegressionEvaluationResult:
    """
    Evaluate regression predictions against ground truth columns.

    Args:
        state: DataFrameState containing the DataFrame with predictions
        params: Evaluation parameters

    Returns:
        RegressionEvaluationResult with metrics

    Raises:
        ValueError: If columns are missing or insufficient data
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    for col in (params.target_column, params.prediction_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # Scoring rows the model trained on is what made every metric optimistic;
    # the holdout is the default and the result records which was used.
    scored, evaluated_on = select_rows(
        df, params.target_column, params.split_column, params.evaluate_on
    )

    data = scored[[params.target_column, params.prediction_column]]
    if params.drop_na:
        data = data.dropna()

    if data.empty:
        raise ValueError("No rows available after dropping missing values")

    y_true = data[params.target_column].to_numpy()
    y_pred = data[params.prediction_column].to_numpy()

    errors = y_pred - y_true
    mse = float((errors ** 2).mean())
    mae = float(abs(errors).mean())
    rmse = math.sqrt(mse)

    # R^2 calculation
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # The train/test gap is what makes memorisation visible; reporting only one
    # of the two numbers is how an overfitted model passes for a good one.
    train_metrics = None
    if evaluated_on == "test":
        train = train_rows(df, params.target_column, params.split_column)
        if train is not None:
            train_data = train[[params.target_column, params.prediction_column]].dropna()
            if not train_data.empty:
                train_metrics = _regression_scores(
                    train_data[params.target_column].to_numpy(),
                    train_data[params.prediction_column].to_numpy(),
                )

    return RegressionEvaluationResult(
        rmse=rmse,
        mae=mae,
        r2=r2,
        n_samples=len(data),
        dataframe_name=source_name,
        target_column=params.target_column,
        prediction_column=params.prediction_column,
        evaluated_on=evaluated_on,
        train_metrics=train_metrics,
    )


def _regression_scores(y_true, y_pred) -> dict:
    errors = y_pred - y_true
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "rmse": math.sqrt(float((errors ** 2).mean())),
        "mae": float(abs(errors).mean()),
        "r2": 1 - ss_res / ss_tot if ss_tot != 0 else 0.0,
        "n_samples": len(y_true),
    }
