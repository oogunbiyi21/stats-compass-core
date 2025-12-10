"""Common models and utilities for ML tools."""

from typing import Any

import pandas as pd

from stats_compass_core.results import ModelTrainingResult
from stats_compass_core.state import DataFrameState


def prepare_ml_data(
    state: DataFrameState,
    target_column: str,
    feature_columns: list[str] | None,
    dataframe_name: str | None,
) -> tuple[pd.DataFrame, pd.Series, list[str], str]:
    """
    Common data preparation for ML tools.
    
    Args:
        state: DataFrameState containing the DataFrame
        target_column: Target column name
        feature_columns: Optional list of feature columns
        dataframe_name: Optional DataFrame name
    
    Returns:
        Tuple of (X, y, feature_cols, source_name)
    """
    df = state.get_dataframe(dataframe_name)
    source_name = dataframe_name or state.get_active_dataframe_name()

    # Validate target column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    # Determine feature columns
    if feature_columns:
        feature_cols = feature_columns
        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Feature columns not found: {missing_cols}")
    else:
        # Use all numeric columns except target
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_column]
        if not feature_cols:
            raise ValueError("No numeric feature columns available")

    # Prepare data
    X = df[feature_cols]
    y = df[target_column]

    # Check for sufficient data
    if len(df) < 2:
        raise ValueError("Insufficient data: need at least 2 samples")

    return X, y, feature_cols, source_name


def create_training_result(
    state: DataFrameState,
    model: object,
    model_type: str,
    target_column: str,
    feature_cols: list[str],
    train_score: float,
    test_score: float | None,
    train_size: int,
    test_size: int | None,
    source_name: str,
    hyperparameters: dict[str, Any],
) -> ModelTrainingResult:
    """
    Create a ModelTrainingResult and store the model in state.
    
    Args:
        state: DataFrameState to store model in
        model: Trained model object
        model_type: Type of model (e.g., "linear_regression")
        target_column: Target column name
        feature_cols: List of feature column names
        train_score: Training score
        test_score: Test score (if test split was used)
        train_size: Number of training samples
        test_size: Number of test samples
        source_name: Source DataFrame name
        hyperparameters: Model hyperparameters
    
    Returns:
        ModelTrainingResult with model stored in state
    """
    # Store model in state with descriptive name
    model_id = state.store_model(
        model=model,
        model_type=model_type,
        target_column=target_column,
        feature_columns=feature_cols,
        source_dataframe=source_name,
    )

    # Build metrics dict
    metrics: dict[str, float] = {"train_score": train_score}
    if test_score is not None:
        metrics["test_score"] = test_score

    # Extract feature importances if available
    feature_importances: dict[str, float] | None = None
    if hasattr(model, 'feature_importances_'):
        feature_importances = {
            col: float(imp)
            for col, imp in zip(feature_cols, model.feature_importances_)
        }

    # Extract coefficients if available
    coefficients: dict[str, float] | None = None
    intercept: float | None = None
    if hasattr(model, 'coef_'):
        coef = model.coef_
        if len(coef.shape) == 1:
            coefficients = {col: float(c) for col, c in zip(feature_cols, coef)}
        elif len(coef.shape) == 2 and coef.shape[0] == 1:
            coefficients = {col: float(c) for col, c in zip(feature_cols, coef[0])}
    if hasattr(model, 'intercept_'):
        intercept_val = model.intercept_
        if hasattr(intercept_val, 'item'):
            intercept = intercept_val.item()
        elif isinstance(intercept_val, (int, float)):
            intercept = float(intercept_val)

    return ModelTrainingResult(
        model_id=model_id,
        model_type=model_type,
        target_column=target_column,
        feature_columns=feature_cols,
        metrics=metrics,
        feature_importances=feature_importances,
        coefficients=coefficients,
        intercept=intercept,
        train_size=train_size,
        test_size=test_size,
        dataframe_name=source_name,
        hyperparameters=hyperparameters,
    )
