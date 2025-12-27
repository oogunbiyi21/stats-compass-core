"""
Shared helper functions for workflow execution.

Provides common utilities for running workflow steps with timing,
error handling, and result serialization.
"""

import time
from typing import Any, Callable

from stats_compass_core.state import DataFrameState

from .results import ChartArtifact, WorkflowStepResult


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
            result_data = result
        else:
            result_data = {"result": str(result)}
        
        # Extract dataframe_name if present
        df_produced = None
        if hasattr(result, "dataframe_name"):
            df_produced = result.dataframe_name
        elif hasattr(result, "predictions_dataframe"):
            df_produced = result.predictions_dataframe
        
        # Check for image in result
        image_base64 = None
        if hasattr(result, "image_base64") and result.image_base64:
            image_base64 = result.image_base64
        elif hasattr(result, "base64_image") and result.base64_image:
            image_base64 = result.base64_image
        
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


def run_chart_step(
    step_name: str,
    step_index: int,
    func: Callable,
    state: DataFrameState,
    params: Any,
    chart_type: str,
    summary_template: str,
) -> tuple[WorkflowStepResult, ChartArtifact | None]:
    """
    Execute a chart generation step with timing and error handling.
    
    Args:
        step_name: Name of the step for reporting
        step_index: Index of the step in the workflow
        func: The chart tool function to execute
        state: DataFrameState to pass to the function
        params: Parameters to pass to the function
        chart_type: Type of chart being generated
        summary_template: Template string for the summary (can use {result})
    
    Returns:
        Tuple of (WorkflowStepResult, ChartArtifact or None if failed)
    """
    start = time.time()
    chart_artifact = None
    try:
        result = func(state, params)
        duration_ms = int((time.time() - start) * 1000)
        
        # Create chart artifact
        # Note: ChartResult uses 'image_base64' but ChartArtifact uses 'base64_image'
        chart_artifact = ChartArtifact(
            chart_type=chart_type,
            title=getattr(result, "title", None) or chart_type,
            description=getattr(result, "description", None),
            base64_image=getattr(result, "image_base64", None),
            format=getattr(result, "image_format", "png"),
        )
        
        step_result = WorkflowStepResult(
            step_name=step_name,
            step_index=step_index,
            status="success",
            duration_ms=duration_ms,
            summary=summary_template.format(result=result),
            result=result.model_dump() if hasattr(result, "model_dump") else {"chart": chart_type},
        )
        return step_result, chart_artifact
        
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        step_result = WorkflowStepResult(
            step_name=step_name,
            step_index=step_index,
            status="failed",
            duration_ms=duration_ms,
            summary=f"Failed to generate {chart_type}: {str(e)}",
            error=str(e),
        )
        return step_result, None
