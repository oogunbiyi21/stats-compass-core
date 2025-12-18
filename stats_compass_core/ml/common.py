"""Common models and utilities for ML tools."""

import os
from typing import Any

import joblib
import numpy as np
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


def create_predictions_dataframe(
    state: DataFrameState,
    model: object,
    source_name: str,
    target_column: str,
    feature_cols: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None,
    y_train: pd.Series,
    y_test: pd.Series | None,
    train_indices: np.ndarray | pd.Index,
    test_indices: np.ndarray | pd.Index | None,
    is_classifier: bool,
) -> tuple[str, str, list[str] | None, list[Any] | None]:
    """
    Create a DataFrame with predictions and save it to state.
    
    Args:
        state: DataFrameState to store predictions in
        model: Trained model object
        source_name: Name of the source DataFrame
        target_column: Target column name
        feature_cols: List of feature column names
        X_train: Training features
        X_test: Test features (if split was used)
        y_train: Training target
        y_test: Test target (if split was used)
        train_indices: Indices for training data
        test_indices: Indices for test data (if split was used)
        is_classifier: Whether this is a classification model
    
    Returns:
        Tuple of (predictions_df_name, prediction_column_name, 
                  probability_columns, class_labels)
    """
    # Get original DataFrame to preserve all columns
    original_df = state.get_dataframe(source_name)
    
    # Create predictions DataFrame starting with original data
    predictions_df = original_df.copy()
    
    # Generate prediction column name
    pred_col_name = f"pred_{target_column}"
    
    # Initialize prediction column with NaN
    predictions_df[pred_col_name] = np.nan
    
    # Generate predictions for train data
    y_train_pred = model.predict(X_train)
    predictions_df.loc[train_indices, pred_col_name] = y_train_pred
    
    # Generate predictions for test data if available
    if X_test is not None and test_indices is not None:
        y_test_pred = model.predict(X_test)
        predictions_df.loc[test_indices, pred_col_name] = y_test_pred
    
    # Add split indicator column
    split_col = f"{target_column}_split"
    predictions_df[split_col] = "unknown"
    predictions_df.loc[train_indices, split_col] = "train"
    if test_indices is not None:
        predictions_df.loc[test_indices, split_col] = "test"
    
    # Handle classification-specific outputs
    probability_columns: list[str] | None = None
    class_labels: list[Any] | None = None
    
    if is_classifier and hasattr(model, "predict_proba"):
        # Get class labels from model (convert numpy types to Python native for JSON serialization)
        class_labels = [label.item() if hasattr(label, 'item') else label for label in model.classes_]
        probability_columns = []
        
        # Generate probability columns for each class
        y_train_proba = model.predict_proba(X_train)
        if X_test is not None:
            y_test_proba = model.predict_proba(X_test)
        
        for i, class_label in enumerate(class_labels):
            # Create column name: prob_<target>_<class_label>
            prob_col_name = f"prob_{target_column}_{class_label}"
            probability_columns.append(prob_col_name)
            
            # Initialize with NaN
            predictions_df[prob_col_name] = np.nan
            
            # Fill in probabilities
            predictions_df.loc[train_indices, prob_col_name] = y_train_proba[:, i]
            if X_test is not None and test_indices is not None:
                predictions_df.loc[test_indices, prob_col_name] = y_test_proba[:, i]
    
    # Store predictions DataFrame with descriptive name
    predictions_df_name = f"{source_name}_predictions"
    state.set_dataframe(
        df=predictions_df,
        name=predictions_df_name,
        operation=f"predictions_{model.__class__.__name__}",
        set_active=False,
    )
    
    return predictions_df_name, pred_col_name, probability_columns, class_labels


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
    save_path: str | None = None,
    # New parameters for predictions
    X_train: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    y_test: pd.Series | None = None,
    train_indices: np.ndarray | pd.Index | None = None,
    test_indices: np.ndarray | pd.Index | None = None,
    is_classifier: bool = False,
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
        save_path: Optional path to save the model file
        X_train: Training features (for predictions)
        X_test: Test features (for predictions)
        y_train: Training target (for predictions)
        y_test: Test target (for predictions)
        train_indices: Indices for training data
        test_indices: Indices for test data
        is_classifier: Whether this is a classification model
    
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

    # Save model to disk if requested
    if save_path:
        filepath = os.path.expanduser(save_path)
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(model, filepath)

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

    # Create predictions DataFrame if training data is provided
    predictions_dataframe: str | None = None
    prediction_column: str | None = None
    probability_columns: list[str] | None = None
    class_labels: list[Any] | None = None
    
    if X_train is not None and y_train is not None and train_indices is not None:
        (
            predictions_dataframe,
            prediction_column,
            probability_columns,
            class_labels,
        ) = create_predictions_dataframe(
            state=state,
            model=model,
            source_name=source_name,
            target_column=target_column,
            feature_cols=feature_cols,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_indices=train_indices,
            test_indices=test_indices,
            is_classifier=is_classifier,
        )

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
        predictions_dataframe=predictions_dataframe,
        prediction_column=prediction_column,
        probability_columns=probability_columns,
        class_labels=class_labels,
        hyperparameters=hyperparameters,
    )
