"""
Feature Engineering Steps for ML Workflows.

Shared logic for bin_rare_categories and mean_target_encoding
that can be reused across classification and regression workflows.
"""

from typing import Any

import pandas as pd

from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState
from stats_compass_core.transforms._cardinality import (
    high_cardinality_reason,
    high_cardinality_warning,
)

from .configs import FeatureEngineeringConfig
from .results import WorkflowStepResult
from .utils import run_step

# =============================================================================
# Tool Registry Mappings
# =============================================================================

FEATURE_TOOLS: dict[str, tuple[str, str]] = {
    # step_name: (category, tool_name)
    "bin_rare_categories": ("transforms", "bin_rare_categories"),
    "target_encode": ("transforms", "mean_target_encoding"),
}


# =============================================================================
# Helper Functions
# =============================================================================

def _resolve_feature_step(step_name: str) -> tuple[Any, type]:
    """
    Get a tool function and its input schema from the registry.
    
    Args:
        step_name: Key in FEATURE_TOOLS mapping
    
    Returns:
        Tuple of (tool_function, InputSchemaClass)
    
    Raises:
        ValueError: If tool not found in registry or mapping
    """
    if step_name not in FEATURE_TOOLS:
        raise ValueError(f"Unknown feature engineering step: {step_name}")

    category, tool_name = FEATURE_TOOLS[step_name]
    metadata = registry.get_tool_metadata(category, tool_name)

    if metadata is None:
        raise ValueError(f"Tool not found in registry: {category}.{tool_name}")

    return metadata.function, metadata.input_schema


def _detect_categorical_columns(
    df: pd.DataFrame,
    target_column: str,
) -> list[str]:
    """
    Auto-detect categorical columns suitable for encoding.
    
    Detects object and category dtype columns, excluding the target.
    Only runs AFTER bin_rare_categories to ensure corrupted numeric
    columns have been cleaned/validated.
    
    Args:
        df: DataFrame to analyze
        target_column: Target column to exclude
    
    Returns:
        List of column names that are categorical
    """
    categorical_cols = []

    for col in df.columns:
        if col == target_column:
            continue

        dtype = df[col].dtype
        if dtype == "object" or dtype.name == "category":
            categorical_cols.append(col)

    return categorical_cols


def _run_feature_step(
    state: DataFrameState,
    step_name: str,
    step_index: int,
    params_dict: dict[str, Any],
    summary_template: str,
) -> WorkflowStepResult:
    """
    Run a single feature engineering step using registry dispatch.
    
    Args:
        state: DataFrameState instance
        step_name: Key in FEATURE_TOOLS mapping
        step_index: Current step number
        params_dict: Parameters to pass to the tool
        summary_template: Template for success summary
    
    Returns:
        WorkflowStepResult
    """
    try:
        tool_func, InputSchema = _resolve_feature_step(step_name)

        # Filter params to only those the schema accepts
        schema_fields = set(InputSchema.model_fields.keys())
        filtered_params = {k: v for k, v in params_dict.items() if k in schema_fields}

        params = InputSchema(**filtered_params)

        return run_step(
            step_name=step_name,
            step_index=step_index,
            func=tool_func,
            state=state,
            params=params,
            summary_template=summary_template,
        )
    except Exception as e:
        return WorkflowStepResult(
            step_name=step_name,
            step_index=step_index,
            status="failed",
            duration_ms=0,
            summary=f"Failed {step_name}: {str(e)}",
            error=str(e),
        )


# =============================================================================
# Main Feature Engineering Function
# =============================================================================

def run_feature_engineering_steps(
    state: DataFrameState,
    config: FeatureEngineeringConfig,
    source_name: str,
    target_column: str,
    start_step_index: int = 0,
    feature_columns: list[str] | None = None,
) -> tuple[list[WorkflowStepResult], list[str], str, int, dict[str, str], list[str]]:
    """
    Run feature engineering steps before model training.
    
    Steps (in order):
    1. Bin rare categories (if enabled) - cleans high-cardinality columns
    2. Auto-detect categorical columns (if not specified)
    3. Target encode categorical columns (if enabled)
    
    Args:
        state: DataFrameState instance
        config: FeatureEngineeringConfig with settings
        source_name: Name of the source DataFrame
        target_column: Target column for encoding
        start_step_index: Starting step number
        feature_columns: The declared features, if any. Only categoricals among
            them are binned and encoded. Encoding every non-target categorical
            is how a post-outcome column reached the model: it became numeric,
            and the trainer's numeric fallback then picked it up.
    
    Returns:
        Tuple of:
        - List of WorkflowStepResults
        - List of created DataFrame names
        - Final DataFrame name to use for training
        - Final step index
        - Mapping of original column name -> encoded column name. Encoding
          replaces the originals, so a caller holding user-supplied feature
          names must translate them or ask the trainer for columns that no
          longer exist.
        - Columns excluded as identifiers. They stay in the DataFrame unencoded,
          so a caller holding declared feature names must drop them too.
    """
    steps: list[WorkflowStepResult] = []
    dataframes_created: list[str] = []
    column_mapping: dict[str, str] = {}
    current_df_name = source_name
    step_index = start_step_index

    # Get current DataFrame for column detection
    df = state.get_dataframe(source_name)

    excluded: dict[str, str] = {}

    def declared(columns: list[str]) -> list[str]:
        return [
            col for col in columns
            if (feature_columns is None or col in feature_columns)
            and col not in excluded
        ]

    # Determine which categorical columns to process
    categorical_columns = config.categorical_columns
    if categorical_columns is None:
        # Will auto-detect after binning (safer)
        categorical_columns = _detect_categorical_columns(df, target_column)
    categorical_columns = declared(categorical_columns)

    # Identifiers are not categories: binning collapses them to a constant and
    # encoding turns the constant into a feature. Screened here rather than
    # left to the tools so the answer does not depend on whether binning runs.
    for col in categorical_columns:
        reason = high_cardinality_reason(df[col])
        if reason:
            excluded[col] = reason
    if excluded:
        categorical_columns = declared(categorical_columns)
        warnings = [
            high_cardinality_warning(col, reason, "kept out of the model")
            for col, reason in excluded.items()
        ]
        for warning in warnings:
            state.record_warning(
                warning.code, source_name, warning.message, warning.columns
            )
        step_index += 1
        steps.append(WorkflowStepResult(
            step_name="screen_categoricals",
            step_index=step_index,
            status="success",
            duration_ms=0,
            summary=f"Kept {len(excluded)} identifier-like column(s) out of the model: "
            + ", ".join(excluded),
            result={
                "excluded_columns": excluded,
                "warnings": [w.model_dump() for w in warnings],
            },
        ))

    # Skip if no categorical columns found
    if not categorical_columns:
        step_index += 1
        steps.append(WorkflowStepResult(
            step_name="feature_engineering",
            step_index=step_index,
            status="skipped",
            summary="No categorical columns found for feature engineering",
            skip_reason=(
                "No object/category dtype columns detected (excluding target)"
                if feature_columns is None
                else "None of the declared feature columns is categorical"
            ),
        ))
        return (
        steps, dataframes_created, current_df_name, step_index, column_mapping,
        list(excluded),
    )

    # =========================================================================
    # Step 1: Bin Rare Categories (if enabled)
    # =========================================================================
    if config.bin_rare_categories:
        step_index += 1
        intermediate_name = f"{source_name}_binned"

        step_result = _run_feature_step(
            state=state,
            step_name="bin_rare_categories",
            step_index=step_index,
            params_dict={
                "dataframe_name": current_df_name,
                "categorical_columns": categorical_columns,
                "threshold": config.rare_threshold,
                "bin_label": config.bin_label,
                "save_as": intermediate_name,
            },
            summary_template=f"Binned rare categories in {len(categorical_columns)} column(s)",
        )
        steps.append(step_result)

        if step_result.status == "success":
            current_df_name = intermediate_name
            dataframes_created.append(intermediate_name)

            # Re-detect categoricals after binning (some may have been cleaned)
            if config.categorical_columns is None:
                df = state.get_dataframe(current_df_name)
                categorical_columns = declared(
                    _detect_categorical_columns(df, target_column)
                )

    # =========================================================================
    # Step 2: Target Encode Categorical Columns (if enabled)
    # =========================================================================
    if config.encode_categoricals and categorical_columns:
        step_index += 1
        encoded_name = f"{source_name}_encoded"

        step_result = _run_feature_step(
            state=state,
            step_name="target_encode",
            step_index=step_index,
            params_dict={
                "dataframe_name": current_df_name,
                "categorical_columns": categorical_columns,
                "target_column": target_column,
                "create_new_columns": False,  # Replace originals for cleaner training
                "save_as": encoded_name,
            },
            summary_template=f"Target-encoded {len(categorical_columns)} categorical column(s)",
        )
        steps.append(step_result)

        if step_result.status == "success":
            current_df_name = encoded_name
            dataframes_created.append(encoded_name)
            # Encoding renames the columns it replaces; without this the caller
            # asks the trainer for names the DataFrame no longer has.
            mapping = (step_result.result or {}).get("column_mapping") or {}
            column_mapping.update(mapping)
    elif config.encode_categoricals and not categorical_columns:
        step_index += 1
        steps.append(WorkflowStepResult(
            step_name="target_encode",
            step_index=step_index,
            status="skipped",
            summary="No categorical columns to encode",
            skip_reason="No valid categorical columns after binning",
        ))

    return (
        steps, dataframes_created, current_df_name, step_index, column_mapping,
        list(excluded),
    )


def map_feature_columns(
    feature_columns: list[str] | None, column_mapping: dict[str, str]
) -> list[str] | None:
    """Translate user-supplied feature names through feature engineering.

    Target encoding replaces a categorical column with an encoded one and drops
    the original, so feature names captured before that step no longer exist by
    the time training runs. Passing them through unchanged fails with "Feature
    columns not found" after the encoding step has already reported success,
    which reads as a bug in the encoding rather than in the plumbing.
    """
    if not feature_columns or not column_mapping:
        return feature_columns
    return [column_mapping.get(col, col) for col in feature_columns]
