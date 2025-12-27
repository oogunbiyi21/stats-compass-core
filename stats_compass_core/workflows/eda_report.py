"""
EDA Report Workflow.

Orchestrates multiple EDA tools to produce a comprehensive
exploratory data analysis report with statistics, correlations,
and visualizations using registry-based dispatch.
"""

from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState

from .configs import EDAConfig
from .utils import run_step, run_chart_step
from .results import (
    ChartArtifact,
    WorkflowArtifacts,
    WorkflowResult,
    WorkflowStepResult,
)


# =============================================================================
# Tool Registry Mappings
# =============================================================================
# Maps step names to (category, tool_name) tuples.
# Adding a new EDA step = add entry here + ensure tool is registered.

EDA_TOOLS: dict[str, tuple[str, str]] = {
    # step_name: (category, tool_name)
    "describe": ("eda", "describe"),
    "correlations": ("eda", "correlations"),
    "analyze_missing_data": ("eda", "analyze_missing_data"),
    "data_quality_report": ("eda", "data_quality_report"),
}

# Visualization tools
PLOT_TOOLS: dict[str, tuple[str, str]] = {
    # plot_type: (category, tool_name)
    "histogram": ("plots", "histogram"),
    "bar_chart": ("plots", "bar_chart"),
}


# =============================================================================
# Input Schema
# =============================================================================

class RunEDAReportInput(StrictToolInput):
    """Input schema for run_eda_report workflow."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to analyze. Uses active if not specified."
    )
    config: EDAConfig | None = Field(
        default=None,
        description="Optional configuration to customize the report. Uses sensible defaults if not provided."
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _get_tool(category: str, name: str) -> tuple[Any, type]:
    """
    Get a tool function and its input schema from the registry.
    
    Returns:
        Tuple of (tool_function, InputSchemaClass)
    
    Raises:
        ValueError: If tool not found in registry
    """
    metadata = registry.get_tool_metadata(category, name)
    if metadata is None:
        raise ValueError(f"Tool not found: {category}.{name}")
    return metadata.function, metadata.input_schema


def _run_eda_step(
    state: DataFrameState,
    step_name: str,
    step_index: int,
    source_name: str,
    extra_params: dict[str, Any] | None = None,
) -> WorkflowStepResult:
    """Run a single EDA step using registry dispatch."""
    if step_name not in EDA_TOOLS:
        return WorkflowStepResult(
            step_name=step_name,
            step_index=step_index,
            status="failed",
            duration_ms=0,
            summary=f"Unknown EDA step: {step_name}",
            error=f"Step '{step_name}' not found in EDA_TOOLS mapping",
        )
    
    category, tool_name = EDA_TOOLS[step_name]
    
    try:
        tool_func, InputSchema = _get_tool(category, tool_name)
        
        # Build params - all EDA tools take dataframe_name
        params_dict = {"dataframe_name": source_name}
        if extra_params:
            # Filter to only params the schema accepts
            schema_fields = set(InputSchema.model_fields.keys())
            for k, v in extra_params.items():
                if k in schema_fields:
                    params_dict[k] = v
        
        params = InputSchema(**params_dict)
        
        return run_step(
            step_name=step_name,
            step_index=step_index,
            func=tool_func,
            state=state,
            params=params,
            summary_template=f"Completed {step_name.replace('_', ' ')}",
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
    name="run_eda_report",
    input_schema=RunEDAReportInput,
    description=(
        "Run a comprehensive EDA report: descriptive statistics, correlations, "
        "missing data analysis, data quality report, and auto-generated visualizations "
        "(histograms for numeric columns, bar charts for categorical columns)."
    ),
    tier="workflow",
)
def run_eda_report(state: DataFrameState, params: RunEDAReportInput) -> WorkflowResult:
    """
    Execute a full EDA workflow on a DataFrame.
    
    Steps (configurable via EDAConfig):
    1. Descriptive statistics (describe)
    2. Correlation analysis (correlations)
    3. Missing data analysis (analyze_missing_data)
    4. Data quality report (data_quality_report)
    5. Histograms for numeric columns
    6. Bar charts for categorical columns
    
    All steps are independent and will continue even if some fail.
    """
    started_at = datetime.now()
    
    # Get config with defaults
    config = params.config or EDAConfig()
    
    # Resolve DataFrame
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()
    
    steps: list[WorkflowStepResult] = []
    charts: list[ChartArtifact] = []
    step_index = 0
    
    # =========================================================================
    # Step 1: Descriptive Statistics
    # =========================================================================
    if config.include_describe:
        step_index += 1
        step_result = _run_eda_step(state, "describe", step_index, source_name)
        steps.append(step_result)
    
    # =========================================================================
    # Step 2: Correlations
    # =========================================================================
    if config.include_correlations:
        step_index += 1
        step_result = _run_eda_step(
            state, "correlations", step_index, source_name,
            extra_params={"method": config.correlation_method}
        )
        steps.append(step_result)
    
    # =========================================================================
    # Step 3: Missing Data Analysis
    # =========================================================================
    if config.include_missing_analysis:
        step_index += 1
        step_result = _run_eda_step(state, "analyze_missing_data", step_index, source_name)
        steps.append(step_result)
    
    # =========================================================================
    # Step 4: Data Quality Report
    # =========================================================================
    if config.include_quality_report:
        step_index += 1
        step_result = _run_eda_step(state, "data_quality_report", step_index, source_name)
        steps.append(step_result)
    
    # =========================================================================
    # Step 5: Histograms for Numeric Columns
    # =========================================================================
    if config.generate_histograms and "histogram" in PLOT_TOOLS:
        try:
            category, tool_name = PLOT_TOOLS["histogram"]
            plot_func, PlotInputSchema = _get_tool(category, tool_name)
            
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            
            for col in numeric_cols[:config.max_histograms]:
                step_index += 1
                try:
                    plot_params = PlotInputSchema(
                        dataframe_name=source_name,
                        column=col,
                    )
                    step_result, chart = run_chart_step(
                        step_name=f"histogram_{col}",
                        step_index=step_index,
                        func=plot_func,
                        state=state,
                        params=plot_params,
                        chart_type="histogram",
                        summary_template=f"Generated histogram for '{col}'",
                    )
                    steps.append(step_result)
                    if chart:
                        charts.append(chart)
                except Exception as e:
                    steps.append(WorkflowStepResult(
                        step_name=f"histogram_{col}",
                        step_index=step_index,
                        status="failed",
                        duration_ms=0,
                        summary=f"Failed to generate histogram for '{col}'",
                        error=str(e),
                    ))
        except Exception as e:
            step_index += 1
            steps.append(WorkflowStepResult(
                step_name="histograms",
                step_index=step_index,
                status="failed",
                duration_ms=0,
                summary="Failed to load histogram tool",
                error=str(e),
            ))
    
    # =========================================================================
    # Step 6: Bar Charts for Categorical Columns
    # =========================================================================
    if config.generate_bar_charts and "bar_chart" in PLOT_TOOLS:
        try:
            category, tool_name = PLOT_TOOLS["bar_chart"]
            plot_func, PlotInputSchema = _get_tool(category, tool_name)
            
            categorical_cols = df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()
            
            for col in categorical_cols[:config.max_bar_charts]:
                step_index += 1
                try:
                    plot_params = PlotInputSchema(
                        dataframe_name=source_name,
                        column=col,
                    )
                    step_result, chart = run_chart_step(
                        step_name=f"bar_chart_{col}",
                        step_index=step_index,
                        func=plot_func,
                        state=state,
                        params=plot_params,
                        chart_type="bar_chart",
                        summary_template=f"Generated bar chart for '{col}'",
                    )
                    steps.append(step_result)
                    if chart:
                        charts.append(chart)
                except Exception as e:
                    steps.append(WorkflowStepResult(
                        step_name=f"bar_chart_{col}",
                        step_index=step_index,
                        status="failed",
                        duration_ms=0,
                        summary=f"Failed to generate bar chart for '{col}'",
                        error=str(e),
                    ))
        except Exception as e:
            step_index += 1
            steps.append(WorkflowStepResult(
                step_name="bar_charts",
                step_index=step_index,
                status="failed",
                duration_ms=0,
                summary="Failed to load bar chart tool",
                error=str(e),
            ))
    
    # =========================================================================
    # Build Final Result
    # =========================================================================
    completed_at = datetime.now()
    total_duration_ms = int((completed_at - started_at).total_seconds() * 1000)
    
    # Determine overall status
    failed_steps = [s for s in steps if s.status == "failed"]
    success_steps = [s for s in steps if s.status == "success"]
    
    if not success_steps:
        overall_status = "failed"
    elif failed_steps:
        overall_status = "partial_failure"
    else:
        overall_status = "success"
    
    # Build artifacts
    artifacts = WorkflowArtifacts(
        dataframes_created=[],
        models_created=[],
        charts=charts,
    )
    
    return WorkflowResult(
        workflow_name="run_eda_report",
        status=overall_status,
        started_at=started_at,
        completed_at=completed_at,
        total_duration_ms=total_duration_ms,
        input_dataframe=source_name,
        steps=steps,
        artifacts=artifacts,
    )
