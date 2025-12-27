"""
Feature Engineering Steps for ML Workflows.

Shared logic for bin_rare_categories and mean_target_encoding
that can be reused across classification and regression workflows.
"""

from typing import Any

import pandas as pd

from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState

from .configs import FeatureEngineeringConfig
from .utils import run_step
from .results import WorkflowStepResult


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

def _get_tool(step_name: str) -> tuple[Any, type]:
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
        tool_func, InputSchema = _get_tool(step_name)
        
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
) -> tuple[list[WorkflowStepResult], list[str], str, int]:
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
    
    Returns:
        Tuple of:
        - List of WorkflowStepResults
        - List of created DataFrame names
        - Final DataFrame name to use for training
        - Final step index
    """
    steps: list[WorkflowStepResult] = []
    dataframes_created: list[str] = []
    current_df_name = source_name
    step_index = start_step_index
    
    # Get current DataFrame for column detection
    df = state.get_dataframe(source_name)
    
    # Determine which categorical columns to process
    categorical_columns = config.categorical_columns
    if categorical_columns is None:
        # Will auto-detect after binning (safer)
        categorical_columns = _detect_categorical_columns(df, target_column)
    
    # Skip if no categorical columns found
    if not categorical_columns:
        step_index += 1
        steps.append(WorkflowStepResult(
            step_name="feature_engineering",
            step_index=step_index,
            status="skipped",
            summary="No categorical columns found for feature engineering",
            skip_reason="No object/category dtype columns detected (excluding target)",
        ))
        return steps, dataframes_created, current_df_name, step_index
    
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
                categorical_columns = _detect_categorical_columns(df, target_column)
    
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
    elif config.encode_categoricals and not categorical_columns:
        step_index += 1
        steps.append(WorkflowStepResult(
            step_name="target_encode",
            step_index=step_index,
            status="skipped",
            summary="No categorical columns to encode",
            skip_reason="No valid categorical columns after binning",
        ))
    
    return steps, dataframes_created, current_df_name, step_index
