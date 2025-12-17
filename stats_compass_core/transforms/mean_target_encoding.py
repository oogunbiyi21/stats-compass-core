"""
Mean target encoding for categorical variables.

Uses sklearn's TargetEncoder with cross-validation to prevent target leakage.
Supports binary, multiclass, and continuous targets.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import MeanTargetEncodingResult
from stats_compass_core.state import DataFrameState

# Lazy import sklearn to allow graceful failure if not installed
try:
    from sklearn.preprocessing import TargetEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MeanTargetEncodingInput(StrictToolInput):
    """Input schema for mean target encoding tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    categorical_columns: list[str] = Field(
        description="List of categorical column names to encode"
    )
    target_column: str = Field(
        description="Target variable column name for computing means"
    )
    cv: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of cross-validation folds to prevent target leakage (2-20)"
    )
    smooth: float | Literal["auto"] = Field(
        default="auto",
        description="Smoothing parameter: 'auto' or a float value for regularization strength"
    )
    target_type: Literal["auto", "continuous", "binary", "multiclass"] = Field(
        default="auto",
        description="Target type: 'auto' infers from data, or specify explicitly"
    )
    create_new_columns: bool = Field(
        default=True,
        description="If True, creates new encoded columns. If False, replaces original columns."
    )
    save_as: str | None = Field(
        default=None,
        description="Name to save the result DataFrame. If None, modifies in place."
    )


def _infer_target_type(series: pd.Series) -> str:
    """Infer the target type from the data."""
    n_unique = series.nunique()

    if n_unique == 2:
        return "binary"
    elif pd.api.types.is_numeric_dtype(series) and n_unique > 10:
        return "continuous"
    else:
        return "multiclass"


def _get_encoding_stats(
    df: pd.DataFrame,
    original_col: str,
    encoded_cols: list[str],
) -> dict[str, Any]:
    """Get statistics for an encoded column."""
    stats = {
        "original_unique_categories": int(df[original_col].nunique()),
        "encoded_columns_count": len(encoded_cols),
    }

    for enc_col in encoded_cols:
        col_stats = {
            "min": float(df[enc_col].min()),
            "max": float(df[enc_col].max()),
            "mean": float(df[enc_col].mean()),
            "std": float(df[enc_col].std()),
        }
        stats[enc_col] = col_stats

    return stats


@registry.register(
    category="transforms",
    input_schema=MeanTargetEncodingInput,
    description="Apply mean target encoding to categorical variables using cross-validated target means",
)
def mean_target_encoding(
    state: DataFrameState, params: MeanTargetEncodingInput
) -> MeanTargetEncodingResult:
    """
    Apply mean target encoding to categorical columns.

    Uses sklearn's TargetEncoder with cross-validation to encode categorical
    variables by replacing each category with the mean of the target variable
    for that category. Includes smoothing to prevent overfitting.

    Args:
        state: DataFrameState containing the DataFrame to operate on
        params: Parameters for encoding

    Returns:
        MeanTargetEncodingResult with encoding details and column mappings

    Raises:
        ImportError: If scikit-learn is not installed
        ValueError: If columns don't exist or aren't categorical
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for mean target encoding. "
            "Install it with: pip install stats-compass-core[ml]"
        )

    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    # Validate target column exists
    if params.target_column not in df.columns:
        raise ValueError(
            f"Target column '{params.target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Validate categorical columns exist
    missing_cols = [col for col in params.categorical_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Categorical columns not found: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Remove target column if mistakenly included
    categorical_columns = [
        col for col in params.categorical_columns
        if col != params.target_column
    ]

    if not categorical_columns:
        raise ValueError("No valid categorical columns to encode after removing target column.")

    # Validate columns are categorical
    valid_columns = []
    for col in categorical_columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            valid_columns.append(col)
        elif df[col].nunique() < 20:
            # Allow numeric columns with few unique values (likely categorical)
            valid_columns.append(col)
        else:
            raise ValueError(
                f"Column '{col}' appears to be numeric with {df[col].nunique()} unique values, "
                "not categorical. Convert to string/category first if intended as categorical."
            )

    # Create working copy
    df_encoded = df.copy()

    # Determine effective target type
    if params.target_type == "auto":
        effective_target_type = _infer_target_type(df_encoded[params.target_column])
    else:
        effective_target_type = params.target_type

    # Configure smoothing parameter
    smooth_param = params.smooth
    if isinstance(smooth_param, str) and smooth_param == "auto":
        smooth_param = "auto"
    else:
        smooth_param = float(smooth_param)

    # Create and configure the TargetEncoder
    encoder = TargetEncoder(
        categories="auto",
        target_type=effective_target_type,
        smooth=smooth_param,
        cv=params.cv,
        shuffle=True,
        random_state=42,
    )

    # Prepare data for encoding
    X_categorical = df_encoded[valid_columns]
    y_target = df_encoded[params.target_column]

    # Handle missing values by filling with placeholder
    X_categorical_filled = X_categorical.fillna("_MISSING_")

    # Fit and transform
    encoded_features = encoder.fit_transform(X_categorical_filled, y_target)

    # Determine column naming based on output shape
    n_features = len(valid_columns)
    n_output_cols = encoded_features.shape[1]

    # Build column names and mapping
    encoded_column_names = []
    column_mapping: dict[str, str | list[str]] = {}

    if n_output_cols == n_features:
        # Simple case: one column per feature (binary/continuous targets)
        for col in valid_columns:
            enc_name = f"{col}_encoded"
            encoded_column_names.append(enc_name)
            column_mapping[col] = enc_name
    else:
        # Multiclass case: multiple columns per feature
        n_classes = n_output_cols // n_features
        for i, col in enumerate(valid_columns):
            col_enc_names = []
            for class_idx in range(n_classes):
                enc_name = f"{col}_encoded_class_{class_idx}"
                encoded_column_names.append(enc_name)
                col_enc_names.append(enc_name)
            column_mapping[col] = col_enc_names

    # Add encoded columns to DataFrame
    encoded_df = pd.DataFrame(
        encoded_features,
        columns=encoded_column_names,
        index=df_encoded.index,
    )
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

    # Calculate encoding statistics BEFORE potentially dropping original columns
    encoding_stats = {}
    for col in valid_columns:
        mapped = column_mapping[col]
        enc_cols = mapped if isinstance(mapped, list) else [mapped]
        encoding_stats[col] = _get_encoding_stats(df_encoded, col, enc_cols)

    # Optionally remove original columns
    if not params.create_new_columns:
        df_encoded = df_encoded.drop(columns=valid_columns)

    # Store encoder for later use on new data

    # Store encoder for later use on new data
    encoder_name = f"target_encoder_{'_'.join(valid_columns[:3])}"
    encoder_id = state.store_model(
        model=encoder,
        model_type="target_encoder",
        target_column=params.target_column,
        feature_columns=valid_columns,
        source_dataframe=source_name,
        custom_name=encoder_name,
    )

    # Save result to state
    if params.save_as:
        result_name = params.save_as
    else:
        result_name = source_name  # Modify in place

    stored_name = state.set_dataframe(df_encoded, name=result_name, operation="mean_target_encoding")

    # Build result
    message_parts = [
        f"Applied mean target encoding to {len(valid_columns)} column(s)",
        f"using '{params.target_column}' as target ({effective_target_type}).",
        f"Created {len(encoded_column_names)} encoded column(s).",
        f"Encoder stored as '{encoder_id}' for applying to new data.",
    ]

    return MeanTargetEncodingResult(
        success=True,
        operation="mean_target_encoding",
        dataframe_name=stored_name,
        source_dataframe=source_name,
        rows_affected=len(df_encoded),
        encoded_columns=encoded_column_names,
        original_columns=valid_columns,
        target_column=params.target_column,
        column_mapping=column_mapping,
        encoding_stats=encoding_stats,
        encoder_id=encoder_id,
        parameters={
            "cv": params.cv,
            "smooth": params.smooth,
            "target_type": params.target_type,
            "effective_target_type": effective_target_type,
            "create_new_columns": params.create_new_columns,
        },
        message=" ".join(message_parts),
    )
