"""
Time Series Forecast Workflow.

Orchestrates ARIMA model fitting, forecasting, and visualization
using registry-based dispatch.
"""

from datetime import datetime
from typing import Any

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState
from stats_compass_core.cleaning.clean_dates import validate_date_column

from .configs import TimeSeriesConfig
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
# Maps workflow step names to registry tool names.

ARIMA_TOOLS: dict[str, str] = {
    "stationarity": "check_stationarity",
    "infer_frequency": "infer_frequency",
    "find_optimal": "find_optimal_arima",
    "fit": "fit_arima",
    "forecast": "forecast_arima",
}

PLOT_TOOLS: dict[str, tuple[str, str]] = {
    # config_name: (tool_name, chart_type_label)
    "forecast_plot": ("forecast_plot", "forecast"),
}


# =============================================================================
# Input Schema
# =============================================================================

class RunTimeseriesForecastInput(StrictToolInput):
    """Input schema for run_timeseries_forecast workflow."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame containing time series data. Uses active if not specified."
    )
    target_column: str = Field(
        description="Name of the column containing values to forecast"
    )
    date_column: str | None = Field(
        default=None,
        description="Name of the date/time column. If not provided, uses row index."
    )
    config: TimeSeriesConfig | None = Field(
        default=None,
        description="Optional configuration to customize the workflow."
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


def _parse_forecast_periods(
    forecast_periods: int | str,
) -> tuple[int | None, int | None, str | None]:
    """
    Parse forecast_periods which can be int or natural language string.
    
    Returns:
        Tuple of (n_periods, forecast_number, forecast_unit)
        Only one of n_periods or (forecast_number, forecast_unit) will be set.
    """
    if isinstance(forecast_periods, int):
        return forecast_periods, None, None
    
    # Parse natural language like "30 days", "1 month", "2 weeks"
    text = str(forecast_periods).lower().strip()
    
    # Extract number and unit
    import re
    match = re.match(r"(\d+)\s*(day|week|month|quarter|year)s?", text)
    if match:
        number = int(match.group(1))
        unit = match.group(2)
        
        # Map to valid forecast_unit values
        unit_map = {
            "day": "days",
            "week": "weeks",
            "month": "months",
            "quarter": "quarters",
            "year": "years",
        }
        return None, number, unit_map.get(unit, "days")
    
    # Fallback - try to parse as int
    try:
        return int(text), None, None
    except ValueError:
        return 30, None, None  # Default to 30 periods


# =============================================================================
# Main Workflow
# =============================================================================

@registry.register(
    category="workflows",
    name="run_timeseries_forecast",
    input_schema=RunTimeseriesForecastInput,
    description=(
        "Run a complete time series forecasting workflow: check stationarity, "
        "optionally find optimal ARIMA parameters, fit an ARIMA model, generate "
        "forecasts, and create a forecast visualization."
    ),
    tier="workflow",
)
def run_timeseries_forecast(
    state: DataFrameState,
    params: RunTimeseriesForecastInput,
) -> WorkflowResult:
    """
    Execute a time series forecasting workflow.
    
    Steps:
    1. (Optional) Check stationarity via ADF/KPSS tests
    2. (Optional) Find optimal ARIMA parameters via grid search
    3. Fit ARIMA model
    4. Generate forecast
    5. (Optional) Create forecast visualization
    
    The workflow stores the fitted model and returns forecast values.
    """
    started_at = datetime.now()
    
    # Get config with defaults
    # If no config provided, create a minimal one
    if params.config is None:
        # Use params from input since TimeSeriesConfig requires date_column/target_column
        from .configs import TimeSeriesConfig
        config = TimeSeriesConfig(
            date_column=params.date_column or "",
            target_column=params.target_column,
        )
    else:
        config = params.config
    
    # Resolve DataFrame
    source_name = params.dataframe_name or state.get_active_dataframe_name()
    df = state.get_dataframe(source_name)
    
    steps: list[WorkflowStepResult] = []
    step_index = 0
    charts: list[ChartArtifact] = []
    dataframes_created: list[str] = []
    models_created: list[str] = []
    
    # Track state across steps
    model_id: str | None = None
    recommended_d: int = 1  # Default differencing order
    optimal_order: tuple[int, int, int] | None = None
    
    # Current dataframe name (may change if we clean dates)
    current_df_name = source_name
    
    # =========================================================================
    # Step 0: Validate Date Column (if enabled)
    # =========================================================================
    if config.validate_dates and params.date_column:
        step_index += 1
        
        try:
            validation = validate_date_column(
                df=df,
                date_column=params.date_column,
                check_nulls=True,
                check_duplicates=True,
                check_chronological=True,
                check_gaps=True,
            )
            
            has_errors = not validation["is_valid"]
            error_msg = "; ".join(validation["errors"]) if validation["errors"] else None
            
            if has_errors:
                if config.handle_missing_dates == "error":
                    # Fail immediately with clear error
                    steps.append(WorkflowStepResult(
                        step_name="validate_dates",
                        step_index=step_index,
                        status="failed",
                        duration_ms=0,
                        summary=f"Date validation failed: {error_msg}",
                        error=error_msg,
                    ))
                    
                    # Build and return early failure result
                    completed_at = datetime.now()
                    total_duration_ms = int((completed_at - started_at).total_seconds() * 1000)
                    
                    return WorkflowResult(
                        workflow_name="run_timeseries_forecast",
                        status="failed",
                        started_at=started_at,
                        completed_at=completed_at,
                        total_duration_ms=total_duration_ms,
                        input_dataframe=source_name,
                        steps=steps,
                        artifacts=WorkflowArtifacts(
                            dataframes_created=[],
                            models_created=[],
                            charts=[],
                            final_dataframe=None,
                        ),
                        error_summary=f"Date validation failed: {error_msg}",
                        suggestion="Clean the date column using preprocessing workflow with date_cleaning config, or set handle_missing_dates='ffill' to auto-fix.",
                        recoverable=True,
                    )
                
                elif config.handle_missing_dates in ["ffill", "bfill", "drop"]:
                    # Attempt to fix automatically using clean_dates tool
                    try:
                        clean_func, CleanInput = _get_tool("cleaning", "clean_dates")
                        
                        cleaned_df_name = f"{source_name}_dates_fixed"
                        
                        clean_params_dict = {
                            "dataframe_name": source_name,
                            "date_column": params.date_column,
                            "fill_method": config.handle_missing_dates,
                            "infer_frequency": True,
                            "create_missing_dates": False,
                            "save_as": cleaned_df_name,
                        }
                        
                        # Filter to schema fields
                        schema_fields = set(CleanInput.model_fields.keys())
                        valid_params = {k: v for k, v in clean_params_dict.items() if k in schema_fields}
                        clean_params = CleanInput(**valid_params)
                        
                        step_result = run_step(
                            step_name="clean_dates",
                            step_index=step_index,
                            func=clean_func,
                            state=state,
                            params=clean_params,
                            summary_template=f"Cleaned date column '{params.date_column}'",
                        )
                        
                        if step_result.status == "success":
                            current_df_name = cleaned_df_name
                            dataframes_created.append(cleaned_df_name)
                            
                            # Update the step to show it was validation + cleaning
                            step_result.step_name = "validate_dates"
                            step_result.summary = f"Fixed date issues: {step_result.summary}"
                            if step_result.result:
                                step_result.result["validation"] = validation
                                step_result.result["cleaned"] = True
                        else:
                            step_result.step_name = "validate_dates"
                            step_result.summary = f"Failed to fix dates: {error_msg}"
                            
                        steps.append(step_result)
                        
                    except Exception as e:
                        steps.append(WorkflowStepResult(
                            step_name="validate_dates",
                            step_index=step_index,
                            status="failed",
                            duration_ms=0,
                            summary=f"Failed to clean dates: {str(e)}",
                            error=str(e),
                        ))
                else:
                    # Unknown handle_missing_dates value - fail immediately
                    steps.append(WorkflowStepResult(
                        step_name="validate_dates",
                        step_index=step_index,
                        status="failed",
                        duration_ms=0,
                        summary=f"Invalid handle_missing_dates value: {config.handle_missing_dates}",
                        error=f"Invalid config value: {config.handle_missing_dates}",
                    ))
                    
                    # Build and return early failure result
                    completed_at = datetime.now()
                    total_duration_ms = int((completed_at - started_at).total_seconds() * 1000)
                    
                    return WorkflowResult(
                        workflow_name="run_timeseries_forecast",
                        status="failed",
                        started_at=started_at,
                        completed_at=completed_at,
                        total_duration_ms=total_duration_ms,
                        input_dataframe=source_name,
                        steps=steps,
                        artifacts=WorkflowArtifacts(
                            dataframes_created=[],
                            models_created=[],
                            charts=[],
                            final_dataframe=None,
                        ),
                        error_summary=f"Invalid handle_missing_dates value: {config.handle_missing_dates}. Must be 'error', 'ffill', or 'drop'.",
                        suggestion="Update config.handle_missing_dates to a valid value: 'error', 'ffill', or 'drop'.",
                        recoverable=True,
                    )
            else:
                # Validation passed
                warning_msg = "; ".join(validation["warnings"]) if validation["warnings"] else "No issues found"
                steps.append(WorkflowStepResult(
                    step_name="validate_dates",
                    step_index=step_index,
                    status="success",
                    duration_ms=0,
                    summary=f"Date column validated: {warning_msg}",
                    result={"validation": validation},
                ))
                
        except Exception as e:
            steps.append(WorkflowStepResult(
                step_name="validate_dates",
                step_index=step_index,
                status="failed",
                duration_ms=0,
                summary=f"Date validation error: {str(e)}",
                error=str(e),
            ))
    
    # =========================================================================
    # Step 1: Check Stationarity (optional)
    # =========================================================================
    if config.check_stationarity:
        step_index += 1
        
        try:
            stationarity_func, StationarityInput = _get_tool("ml", ARIMA_TOOLS["stationarity"])
            
            stationarity_params_dict = {
                "dataframe_name": current_df_name,
                "target_column": params.target_column,
                "test_type": "both",
            }
            
            # Filter to schema fields
            schema_fields = set(StationarityInput.model_fields.keys())
            valid_params = {k: v for k, v in stationarity_params_dict.items() if k in schema_fields}
            stationarity_params = StationarityInput(**valid_params)
            
            step_result = run_step(
                step_name="check_stationarity",
                step_index=step_index,
                func=stationarity_func,
                state=state,
                params=stationarity_params,
                summary_template="Checked time series stationarity",
            )
            steps.append(step_result)
            
            # Extract stationarity info for ARIMA fitting
            if step_result.status == "success" and step_result.result:
                is_stationary = step_result.result.get("is_stationary", False)
                if is_stationary:
                    recommended_d = 0
                else:
                    recommended_d = 1
                    
        except Exception as e:
            steps.append(WorkflowStepResult(
                step_name="check_stationarity",
                step_index=step_index,
                status="failed",
                duration_ms=0,
                summary=f"Failed to check stationarity: {str(e)}",
                error=str(e),
            ))
    
    # =========================================================================
    # Step 2: Find Optimal ARIMA Parameters (optional)
    # =========================================================================
    if config.auto_find_params:
        step_index += 1
        
        try:
            find_func, FindInput = _get_tool("ml", ARIMA_TOOLS["find_optimal"])
            
            find_params_dict = {
                "dataframe_name": current_df_name,
                "target_column": params.target_column,
                "date_column": params.date_column,
                "max_p": 3,
                "max_d": 2,
                "max_q": 3,
                "criterion": "aic",
                "top_n": 3,
            }
            
            # Filter to schema fields
            schema_fields = set(FindInput.model_fields.keys())
            valid_params = {k: v for k, v in find_params_dict.items() if k in schema_fields}
            find_params = FindInput(**valid_params)
            
            step_result = run_step(
                step_name="find_optimal_params",
                step_index=step_index,
                func=find_func,
                state=state,
                params=find_params,
                summary_template="Found optimal ARIMA parameters",
            )
            steps.append(step_result)
            
            # Extract optimal order
            if step_result.status == "success" and step_result.result:
                best_order = step_result.result.get("best_order")
                if best_order and len(best_order) == 3:
                    optimal_order = tuple(best_order)
                    
        except Exception as e:
            steps.append(WorkflowStepResult(
                step_name="find_optimal_params",
                step_index=step_index,
                status="failed",
                duration_ms=0,
                summary=f"Failed to find optimal parameters: {str(e)}",
                error=str(e),
            ))
    
    # =========================================================================
    # Step 3: Fit ARIMA Model
    # =========================================================================
    step_index += 1
    
    try:
        fit_func, FitInput = _get_tool("ml", ARIMA_TOOLS["fit"])
        
        # Determine ARIMA order
        if optimal_order:
            p, d, q = optimal_order
        elif config.arima_order:
            p, d, q = config.arima_order
        else:
            # Use default order with recommended_d from stationarity test
            p, d, q = 1, recommended_d, 1
        
        fit_params_dict = {
            "dataframe_name": current_df_name,
            "target_column": params.target_column,
            "date_column": params.date_column,
            "p": p,
            "d": d,
            "q": q,
        }
        
        # Add seasonal parameters if provided
        if config.seasonal_order:
            fit_params_dict["seasonal"] = True
            fit_params_dict["seasonal_order"] = list(config.seasonal_order)
        
        # Filter to schema fields
        schema_fields = set(FitInput.model_fields.keys())
        valid_params = {k: v for k, v in fit_params_dict.items() if k in schema_fields}
        fit_params = FitInput(**valid_params)
        
        step_result = run_step(
            step_name="fit_arima",
            step_index=step_index,
            func=fit_func,
            state=state,
            params=fit_params,
            summary_template=f"Fitted ARIMA({p},{d},{q}) model",
        )
        steps.append(step_result)
        
        # Extract model ID
        if step_result.status == "success" and step_result.result:
            model_id = step_result.result.get("model_id")
            if model_id:
                models_created.append(model_id)
                
    except Exception as e:
        steps.append(WorkflowStepResult(
            step_name="fit_arima",
            step_index=step_index,
            status="failed",
            duration_ms=0,
            summary=f"Failed to fit ARIMA model: {str(e)}",
            error=str(e),
        ))
    
    # =========================================================================
    # Step 4: Generate Forecast
    # =========================================================================
    forecast_result_data: dict | None = None
    
    if model_id:
        step_index += 1
        
        try:
            forecast_func, ForecastInput = _get_tool("ml", ARIMA_TOOLS["forecast"])
            
            # Parse forecast periods
            n_periods, forecast_number, forecast_unit = _parse_forecast_periods(
                config.forecast_periods
            )
            
            forecast_params_dict = {
                "model_id": model_id,
                "include_plot": False,  # We'll use the dedicated plot tool instead
            }
            
            if n_periods is not None:
                forecast_params_dict["n_periods"] = n_periods
            if forecast_number is not None:
                forecast_params_dict["forecast_number"] = forecast_number
            if forecast_unit is not None:
                forecast_params_dict["forecast_unit"] = forecast_unit
            
            # Filter to schema fields
            schema_fields = set(ForecastInput.model_fields.keys())
            valid_params = {k: v for k, v in forecast_params_dict.items() if k in schema_fields}
            forecast_params = ForecastInput(**valid_params)
            
            step_result = run_step(
                step_name="forecast",
                step_index=step_index,
                func=forecast_func,
                state=state,
                params=forecast_params,
                summary_template="Generated forecast",
            )
            steps.append(step_result)
            
            if step_result.status == "success" and step_result.result:
                forecast_result_data = step_result.result
                
        except Exception as e:
            steps.append(WorkflowStepResult(
                step_name="forecast",
                step_index=step_index,
                status="failed",
                duration_ms=0,
                summary=f"Failed to generate forecast: {str(e)}",
                error=str(e),
            ))
    
    # =========================================================================
    # Step 5: Generate Forecast Plot (optional)
    # =========================================================================
    if config.generate_forecast_plot and model_id:
        step_index += 1
        
        try:
            plot_func, PlotInput = _get_tool("plots", PLOT_TOOLS["forecast_plot"][0])
            
            # Determine n_periods for plot
            n_periods, forecast_number, forecast_unit = _parse_forecast_periods(
                config.forecast_periods
            )
            
            plot_params_dict = {
                "model_id": model_id,
                "n_periods": n_periods or 30,  # Default to 30 if using natural language
            }
            
            # Filter to schema fields
            schema_fields = set(PlotInput.model_fields.keys())
            valid_params = {k: v for k, v in plot_params_dict.items() if k in schema_fields}
            plot_params = PlotInput(**valid_params)
            
            step_result, chart = run_chart_step(
                step_name="plot_forecast",
                step_index=step_index,
                func=plot_func,
                state=state,
                params=plot_params,
                chart_type="forecast",
                summary_template="Generated forecast plot",
            )
            steps.append(step_result)
            if chart:
                charts.append(chart)
                
        except Exception as e:
            steps.append(WorkflowStepResult(
                step_name="plot_forecast",
                step_index=step_index,
                status="failed",
                duration_ms=0,
                summary=f"Failed to generate forecast plot: {str(e)}",
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
        error_summary = "All steps failed"
        suggestion = "Check that the DataFrame exists and has a valid numeric target column."
    elif failed_steps:
        overall_status = "partial_failure"
        failed_names = [s.step_name for s in failed_steps]
        error_summary = f"{len(failed_steps)} step(s) failed: {', '.join(failed_names)}"
        suggestion = "Review failed steps. The forecast may still be usable."
    else:
        overall_status = "success"
        error_summary = None
        suggestion = None
    
    # Build artifacts
    artifacts = WorkflowArtifacts(
        dataframes_created=dataframes_created,
        models_created=models_created,
        charts=charts,
        final_dataframe=None,  # Timeseries doesn't create a predictions DF by default
    )
    
    # Add forecast data to artifacts if available
    if forecast_result_data:
        artifacts_dict = artifacts.model_dump()
        artifacts_dict["forecast_data"] = {
            "values": forecast_result_data.get("forecast_values"),
            "index": forecast_result_data.get("forecast_index"),
            "lower_ci": forecast_result_data.get("lower_ci"),
            "upper_ci": forecast_result_data.get("upper_ci"),
            "n_periods": forecast_result_data.get("n_periods"),
        }
        # We'll just include forecast info in the result message instead
    
    return WorkflowResult(
        workflow_name="run_timeseries_forecast",
        status=overall_status,
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
