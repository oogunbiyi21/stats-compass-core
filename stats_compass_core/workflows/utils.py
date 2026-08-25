"""
Shared helper functions for workflow execution.

Provides common utilities for running workflow steps with timing,
error handling, and result serialization.
"""

import time
from datetime import datetime
from typing import Any, Callable

from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState

from .results import WorkflowStepResult


def get_tool(category: str, name: str) -> tuple[Any, type]:
    """
    Get a tool function and its input schema from the registry.

    This lived as four byte-identical private copies across classification,
    regression, eda_report and timeseries. Two *other* modules
    (feature_engineering, preprocessing) defined a different function under the
    same name, taking a step name rather than a category — those are now
    _resolve_feature_step and _resolve_cleaning_step, because one name for two
    contracts is how a call ends up in the wrong place.

    Returns:
        Tuple of (tool_function, InputSchemaClass)

    Raises:
        ValueError: If tool not found in registry
    """
    metadata = registry.get_tool_metadata(category, name)
    if metadata is None:
        raise ValueError(f"Tool not found: {category}.{name}")
    return metadata.function, metadata.input_schema


def generate_model_save_path(
    model_type: str,
    target_column: str,
    custom_path: str | None = None,
) -> str | None:
    """
    Generate a save path for a trained model.
    
    Args:
        model_type: Type of model (e.g., "random_forest", "logistic")
        target_column: Name of the target column
        custom_path: User-provided path (returned as-is if provided)
    
    Returns:
        Path string, or None if saving is disabled
    """
    import os
    import tempfile

    if custom_path:
        return custom_path

    # Auto-generate path in temp directory (avoids read-only filesystem issues)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_type}_{target_column}_{timestamp}.joblib"
    return os.path.join(tempfile.gettempdir(), filename)


def build_training_params(
    *,
    input_schema: type,
    source_name: str,
    target_column: str,
    feature_columns: list[str] | None,
    config: Any,
) -> Any:
    """
    Build training parameters dynamically based on the input schema.

    Keyword-only on purpose. This existed as two near-identical copies, and
    classification called it by keyword while regression called it
    positionally — so the fix that taught it to pass encoding-renamed feature
    columns matched one call site and silently missed the other. Regression
    went on asking the trainer for columns that no longer existed. Keyword-only
    makes that divergence impossible to recreate.

    Merge order is load-bearing: config.hyperparameters override the common
    params, and the schema filter then drops anything this particular trainer
    does not accept. Unknown hyperparameter keys are dropped silently, which
    test_model_tuning_params.py relies on.

    Args:
        input_schema: The trainer's input schema; also the filter.
        source_name: DataFrame to train on (post feature engineering).
        target_column: Name of the target column.
        feature_columns: Feature names, already translated through any encoding.
        config: ClassificationConfig or RegressionConfig.
    """
    save_path = None
    if config.save_model:
        save_path = generate_model_save_path(
            model_type=config.model_type,
            target_column=target_column,
            custom_path=config.model_save_path,
        )

    all_params = {
        "dataframe_name": source_name,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "test_size": config.test_size,
        "random_state": config.random_state,
        "save_path": save_path,
        **(config.hyperparameters or {}),
    }

    schema_fields = set(input_schema.model_fields.keys())
    return input_schema(**{k: v for k, v in all_params.items() if k in schema_fields})


def run_step(
    step_name: str,
    step_index: int,
    func: Callable,
    state: DataFrameState,
    params: Any,
    summary_template: str,
) -> WorkflowStepResult:
    """
    Execute a single workflow step with timing and error handling.
    
    Args:
        step_name: Name of the step for reporting
        step_index: Index of the step in the workflow
        func: The tool function to execute
        state: DataFrameState to pass to the function
        params: Parameters to pass to the function
        summary_template: Template string for the summary (can use {result})
    
    Returns:
        WorkflowStepResult with status, timing, and result data
    """
    start = time.time()
    try:
        result = func(state, params)
        duration_ms = int((time.time() - start) * 1000)

        # Check if result indicates failure (OperationError or success=False)
        is_error = False
        error_message = None
        if hasattr(result, "success") and result.success is False:
            is_error = True
            error_message = getattr(result, "error_message", None) or getattr(result, "message", "Operation failed")
        elif hasattr(result, "error_type") and result.error_type:
            is_error = True
            error_message = getattr(result, "error_message", result.error_type)

        if is_error:
            result_data = result.model_dump() if hasattr(result, "model_dump") else {"error": error_message}
            return WorkflowStepResult(
                step_name=step_name,
                step_index=step_index,
                status="failed",
                duration_ms=duration_ms,
                summary=f"Failed: {error_message}",
                error=error_message,
                result=result_data,
            )

        # Serialize result
        if hasattr(result, "model_dump"):
            result_data = result.model_dump()
        elif isinstance(result, dict):
            result_data = result.copy()
        else:
            result_data = {"result": str(result)}

        # Extract dataframe_name if present
        df_produced = None
        if hasattr(result, "dataframe_name"):
            df_produced = result.dataframe_name
        elif hasattr(result, "predictions_dataframe"):
            df_produced = result.predictions_dataframe

        # Check for image in result and remove from result_data to avoid duplication
        image_base64 = None
        if hasattr(result, "image_base64") and result.image_base64:
            image_base64 = result.image_base64
            result_data.pop("image_base64", None)
        elif hasattr(result, "base64_image") and result.base64_image:
            image_base64 = result.base64_image
            result_data.pop("base64_image", None)

        return WorkflowStepResult(
            step_name=step_name,
            step_index=step_index,
            status="success",
            duration_ms=duration_ms,
            summary=summary_template.format(result=result),
            result=result_data,
            dataframe_produced=df_produced,
            image_base64=image_base64,
        )
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        return WorkflowStepResult(
            step_name=step_name,
            step_index=step_index,
            status="failed",
            duration_ms=duration_ms,
            summary=f"Failed: {str(e)}",
            error=str(e),
        )


def build_holdout_predictions(
    state, predictions_df_name: str, target_column: str
) -> tuple[str, str]:
    """Create a test-only view of a predictions frame, for charts.

    Diagnostic plots were handed the full predictions frame, so a confusion
    matrix, ROC curve and PR curve all described the training data alongside a
    correctly held-out accuracy figure. The chart is the thing people screenshot,
    so leaving it inflated while fixing the number would have been worse than
    fixing neither.

    Returns (dataframe_name, evaluated_on) — falling back to the original frame
    when there is no split to filter on.
    """
    split_col = f"{target_column}_split"
    df = state.get_dataframe(predictions_df_name)

    if split_col not in df.columns:
        return predictions_df_name, "all"

    test_rows = df[df[split_col] == "test"]
    if test_rows.empty:
        return predictions_df_name, "all"

    holdout_name = f"{predictions_df_name}_holdout"
    state.set_dataframe(
        df=test_rows,
        name=holdout_name,
        operation="holdout_predictions",
        set_active=False,
    )
    return holdout_name, "test"
