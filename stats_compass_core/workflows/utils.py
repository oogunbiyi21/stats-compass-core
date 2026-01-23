"""
Shared helper functions for workflow execution.

Provides common utilities for running workflow steps with timing,
error handling, and result serialization.
"""

import time
from datetime import datetime
from typing import Any, Callable

from stats_compass_core.state import DataFrameState

from .results import WorkflowStepResult


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
    if custom_path:
        return custom_path
    
    # Auto-generate path using model type and target column
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_type}_{target_column}_{timestamp}.joblib"


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
