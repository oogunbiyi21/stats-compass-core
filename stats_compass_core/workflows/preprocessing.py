"""
Preprocessing Workflow.

Orchestrates multiple cleaning tools to prepare data for analysis
or machine learning using registry-based dispatch.
"""

from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState

from .configs import PreprocessingConfig, ImputationConfig, OutlierConfig, DateCleaningConfig
from .utils import run_step
from .results import (
    WorkflowArtifacts,
    WorkflowResult,
    WorkflowStepResult,
)


# =============================================================================
# Tool Registry Mappings
# =============================================================================
# Maps step names to (category, tool_name) tuples.
# Adding a new preprocessing step = add entry here + ensure tool is registered.

CLEANING_TOOLS: dict[str, tuple[str, str]] = {
    # step_name: (category, tool_name)
    "analyze_missing_data": ("eda", "analyze_missing_data"),
    "clean_dates": ("cleaning", "clean_dates"),
    "apply_imputation": ("cleaning", "apply_imputation"),
    "drop_na": ("cleaning", "drop_na"),
    "handle_outliers": ("cleaning", "handle_outliers"),
    "dedupe": ("cleaning", "dedupe"),
}


# =============================================================================
# Input Schema
# =============================================================================

class RunPreprocessingInput(StrictToolInput):
    """Input schema for run_preprocessing workflow."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to preprocess. Uses active if not specified."
    )
    config: PreprocessingConfig | None = Field(
        default=None,
        description="Optional configuration to customize preprocessing. Uses sensible defaults if not provided."
    )
    save_as: str | None = Field(
        default=None,
        description="Name for the preprocessed DataFrame. If not provided, auto-generates name."
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _get_tool(step_name: str) -> tuple[Any, type]:
    """
    Get a tool function and its input schema from the registry.
    
    Args:
        step_name: Key in CLEANING_TOOLS mapping
    
    Returns:
        Tuple of (tool_function, InputSchemaClass)
    
    Raises:
        ValueError: If tool not found in registry or mapping
    """
    if step_name not in CLEANING_TOOLS:
        raise ValueError(f"Unknown step: {step_name}")
    
    category, tool_name = CLEANING_TOOLS[step_name]
    metadata = registry.get_tool_metadata(category, tool_name)
    
    if metadata is None:
        raise ValueError(f"Tool not found in registry: {category}.{tool_name}")
    
    return metadata.function, metadata.input_schema


def _run_preprocessing_step(
    state: DataFrameState,
    step_name: str,
    step_index: int,
    params_dict: dict[str, Any],
    summary_template: str,
) -> WorkflowStepResult:
    """
    Run a single preprocessing step using registry dispatch.
    
    Args:
        state: DataFrameState instance
        step_name: Key in CLEANING_TOOLS mapping
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
# Main Workflow
# =============================================================================

@registry.register(
    category="workflows",
    name="run_preprocessing",
    input_schema=RunPreprocessingInput,
    description=(
        "Run a data preprocessing pipeline: analyze missing data, apply imputation, "
        "handle outliers, and remove duplicates. Creates a cleaned DataFrame ready "
        "for analysis or modeling."
    ),
    tier="workflow",
)
def run_preprocessing(state: DataFrameState, params: RunPreprocessingInput) -> WorkflowResult:
    """
    Execute a preprocessing workflow on a DataFrame.
    
    Steps:
    1. Analyze missing data (informational)
    2. Apply imputation for missing values (or drop rows)
    3. Handle outliers in numeric columns
    4. Remove duplicate rows
    
    Each step operates on the result of the previous step.
    The workflow creates a new DataFrame with the cleaned data.
    """
    started_at = datetime.now()
    
    # Get config with defaults
    config = params.config or PreprocessingConfig()
    imputation_config = config.imputation or ImputationConfig()
    outlier_config = config.outliers or OutlierConfig()
    date_cleaning_config = config.date_cleaning
    
    # Resolve DataFrame
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()
    
    # Determine output name
    output_name = params.save_as or f"{source_name}_preprocessed"
    
    steps: list[WorkflowStepResult] = []
    step_index = 0
    current_df_name = source_name
    dataframes_created: list[str] = []
    
    # =========================================================================
    # Step 1: Analyze Missing Data (informational - doesn't modify data)
    # =========================================================================
    step_index += 1
    step_result = _run_preprocessing_step(
        state=state,
        step_name="analyze_missing_data",
        step_index=step_index,
        params_dict={"dataframe_name": current_df_name},
        summary_template="Analyzed missing data patterns",
    )
    steps.append(step_result)
    
    # Extract missing columns info for imputation
    columns_with_missing = []
    if step_result.status == "success" and step_result.result:
        missing_summary = step_result.result.get("missing_summary", {})
        missing_by_col = missing_summary.get("missing_by_column", {})
        columns_with_missing = list(missing_by_col.keys())
    
    # =========================================================================
    # Step 2: Clean Date Columns (if configured)
    # =========================================================================
    if date_cleaning_config and date_cleaning_config.date_column:
        # Find date columns - use specified column or detect datetime columns
        date_col = date_cleaning_config.date_column
        
        current_df = state.get_dataframe(current_df_name)
        
        if date_col in current_df.columns:
            step_index += 1
            intermediate_name = f"{current_df_name}_dates_cleaned"
            
            params_dict = {
                "dataframe_name": current_df_name,
                "date_column": date_col,
                "fill_method": date_cleaning_config.fill_method,
                "infer_frequency": date_cleaning_config.infer_frequency,
                "create_missing_dates": date_cleaning_config.create_missing_dates,
                "save_as": intermediate_name,
            }
            
            step_result = _run_preprocessing_step(
                state=state,
                step_name="clean_dates",
                step_index=step_index,
                params_dict=params_dict,
                summary_template=f"Cleaned date column '{date_col}'",
            )
            steps.append(step_result)
            
            if step_result.status == "success":
                current_df_name = intermediate_name
                dataframes_created.append(intermediate_name)
        else:
            step_index += 1
            steps.append(WorkflowStepResult(
                step_name="clean_dates",
                step_index=step_index,
                status="skipped",
                summary=f"Date column '{date_col}' not found",
                skip_reason=f"Column '{date_col}' does not exist in DataFrame",
            ))
    else:
        step_index += 1
        steps.append(WorkflowStepResult(
            step_name="clean_dates",
            step_index=step_index,
            status="skipped",
            summary="No date column specified",
            skip_reason="date_cleaning.date_column not set in config",
        ))
    
    # =========================================================================
    # Step 3: Apply Imputation (if there are missing values)
    # =========================================================================
    if columns_with_missing:
        step_index += 1
        intermediate_name = f"{source_name}_imputed"
        
        strategy = imputation_config.strategy
        
        if strategy == "drop":
            # Use drop_na tool
            step_result = _run_preprocessing_step(
                state=state,
                step_name="drop_na",
                step_index=step_index,
                params_dict={
                    "dataframe_name": current_df_name,
                    "how": "any",
                    "save_as": intermediate_name,
                },
                summary_template="Dropped rows with missing values",
            )
        else:
            # Map our config strategy to tool strategy
            tool_strategy = "most_frequent" if strategy == "mode" else strategy
            
            params_dict = {
                "dataframe_name": current_df_name,
                "strategy": tool_strategy,
                "columns": imputation_config.columns,
                "save_as": intermediate_name,
            }
            
            if strategy == "constant" and imputation_config.constant_value is not None:
                params_dict["fill_value"] = str(imputation_config.constant_value)
            
            step_result = _run_preprocessing_step(
                state=state,
                step_name="apply_imputation",
                step_index=step_index,
                params_dict=params_dict,
                summary_template="Applied imputation for missing values",
            )
        
        steps.append(step_result)
        
        if step_result.status == "success":
            current_df_name = intermediate_name
            dataframes_created.append(intermediate_name)
    else:
        step_index += 1
        steps.append(WorkflowStepResult(
            step_name="apply_imputation",
            step_index=step_index,
            status="skipped",
            summary="No missing values found",
            skip_reason="No columns had missing values",
        ))
    
    # =========================================================================
    # Step 4: Handle Outliers (if enabled)
    # =========================================================================
    if outlier_config.method != "none":
        # Get numeric columns from current state
        current_df = state.get_dataframe(current_df_name)
        numeric_cols = current_df.select_dtypes(include=["number"]).columns.tolist()
        
        if outlier_config.columns:
            # Filter to specified columns
            numeric_cols = [c for c in outlier_config.columns if c in numeric_cols]
        
        if numeric_cols:
            # Map config method/action to tool method
            method_map = {
                ("iqr", "cap"): "clip_iqr",
                ("iqr", "remove"): "remove",
                ("zscore", "cap"): "cap",
                ("zscore", "remove"): "remove",
            }
            tool_method = method_map.get(
                (outlier_config.method, outlier_config.action),
                "clip_iqr"
            )
            
            # Handle outliers for each numeric column
            for col in numeric_cols:
                step_index += 1
                intermediate_name = f"{current_df_name}_outliers_{col}"
                
                step_result = _run_preprocessing_step(
                    state=state,
                    step_name="handle_outliers",
                    step_index=step_index,
                    params_dict={
                        "dataframe_name": current_df_name,
                        "column": col,
                        "method": tool_method,
                        "save_as": intermediate_name,
                    },
                    summary_template=f"Handled outliers in column '{col}'",
                )
                steps.append(step_result)
                
                if step_result.status == "success":
                    current_df_name = intermediate_name
                    dataframes_created.append(intermediate_name)
        else:
            step_index += 1
            steps.append(WorkflowStepResult(
                step_name="handle_outliers",
                step_index=step_index,
                status="skipped",
                summary="No numeric columns to check for outliers",
                skip_reason="No numeric columns found or specified",
            ))
    else:
        step_index += 1
        steps.append(WorkflowStepResult(
            step_name="handle_outliers",
            step_index=step_index,
            status="skipped",
            summary="Outlier handling disabled",
            skip_reason="method='none' in config",
        ))
    
    # =========================================================================
    # Step 5: Remove Duplicates (if enabled)
    # =========================================================================
    if config.dedupe:
        step_index += 1
        
        step_result = _run_preprocessing_step(
            state=state,
            step_name="dedupe",
            step_index=step_index,
            params_dict={
                "dataframe_name": current_df_name,
                "save_as": output_name,
            },
            summary_template="Removed duplicate rows",
        )
        steps.append(step_result)
        
        if step_result.status == "success":
            current_df_name = output_name
            if output_name not in dataframes_created:
                dataframes_created.append(output_name)
    else:
        step_index += 1
        steps.append(WorkflowStepResult(
            step_name="dedupe",
            step_index=step_index,
            status="skipped",
            summary="Deduplication disabled",
            skip_reason="dedupe=False in config",
        ))
    
    # =========================================================================
    # Final Cleanup: Ensure output DataFrame exists
    # =========================================================================
    if output_name not in dataframes_created and current_df_name == source_name:
        # No changes were made, create a copy as the output
        current_df = state.get_dataframe(source_name)
        state.set_dataframe(current_df.copy(), name=output_name, operation="preprocessing_copy")
        dataframes_created.append(output_name)
        current_df_name = output_name
    
    # =========================================================================
    # Build Final Result
    # =========================================================================
    completed_at = datetime.now()
    total_duration_ms = int((completed_at - started_at).total_seconds() * 1000)
    
    failed_steps = [s for s in steps if s.status == "failed"]
    success_steps = [s for s in steps if s.status == "success"]
    
    if not failed_steps:
        status = "success"
        error_summary = None
        suggestion = None
    elif success_steps:
        status = "partial_failure"
        failed_names = [s.step_name for s in failed_steps]
        error_summary = f"{len(failed_steps)} step(s) failed: {', '.join(failed_names)}"
        suggestion = "Review failed steps. The output may still be usable."
    else:
        status = "failed"
        error_summary = "All steps failed"
        suggestion = "Check that the DataFrame exists and has valid data."
    
    # Build artifacts
    artifacts = WorkflowArtifacts(
        dataframes_created=dataframes_created,
        models_created=[],
        charts=[],
        final_dataframe=current_df_name,
    )
    
    return WorkflowResult(
        workflow_name="run_preprocessing",
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        total_duration_ms=total_duration_ms,
        input_dataframe=source_name,
        steps=steps,
        artifacts=artifacts,
        error_summary=error_summary,
        suggestion=suggestion,
        recoverable=True,
    )
