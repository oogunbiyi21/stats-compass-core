"""
Regression Workflow.

Orchestrates model training, evaluation, and visualization for
regression tasks using registry-based dispatch.
"""

from datetime import datetime
from typing import Any

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState

from .configs import RegressionConfig, FeatureEngineeringConfig
from .utils import run_step, run_chart_step
from .results import (
    ChartArtifact,
    WorkflowArtifacts,
    WorkflowResult,
    WorkflowStepResult,
)
from .feature_engineering import run_feature_engineering_steps


# =============================================================================
# Model Registry Mappings
# =============================================================================
# Maps user-friendly model type names to registry tool names.
# Adding a new model = add one entry here + ensure it's registered in ml/

REGRESSOR_TOOLS: dict[str, str] = {
    "random_forest": "train_random_forest_regressor",
    "gradient_boosting": "train_gradient_boosting_regressor",
    "linear": "train_linear_regression",
    # Possible future additions:
    # "ridge": "train_ridge_regression",
    # "lasso": "train_lasso_regression",
    # "xgboost": "train_xgboost_regressor",
}

# Human-readable labels for model types
MODEL_LABELS: dict[str, str] = {
    "random_forest": "Random Forest Regressor",
    "gradient_boosting": "Gradient Boosting Regressor",
    "linear": "Linear Regression",
}

# Plot tools - maps plot config names to registry tool names
PLOT_TOOLS: dict[str, tuple[str, str]] = {
    # config_name: (tool_name, chart_type_label)
    "feature_importance": ("feature_importance", "feature_importance"),
    # Note: residuals and predicted_vs_actual plots would need to be added
}


# =============================================================================
# Input Schema
# =============================================================================

class RunRegressionInput(StrictToolInput):
    """Input schema for run_regression workflow."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to train on. Uses active if not specified."
    )
    target_column: str = Field(
        description="Name of the target column (continuous values to predict)"
    )
    feature_columns: list[str] | None = Field(
        default=None,
        description="List of feature columns. If None, uses all numeric columns except target."
    )
    config: RegressionConfig | None = Field(
        default=None,
        description="Optional configuration to customize the workflow. Uses sensible defaults if not provided."
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


def _build_training_params(
    input_schema: type,
    source_name: str,
    target_column: str,
    feature_columns: list[str] | None,
    config: RegressionConfig,
) -> Any:
    """
    Build training parameters dynamically based on the input schema.
    
    Common parameters are set explicitly, model-specific hyperparameters
    are passed through from config.hyperparameters.
    """
    # Common parameters all training tools share
    common_params = {
        "dataframe_name": source_name,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "test_size": config.test_size,
        "random_state": config.random_state,
        "save_path": config.model_save_path if config.save_model else None,
    }
    
    # Merge with model-specific hyperparameters
    hyperparams = config.hyperparameters or {}
    all_params = {**common_params, **hyperparams}
    
    # Filter to only params the schema accepts
    schema_fields = set(input_schema.model_fields.keys())
    valid_params = {k: v for k, v in all_params.items() if k in schema_fields}
    
    return input_schema(**valid_params)


# =============================================================================
# Main Workflow
# =============================================================================

@registry.register(
    category="workflows",
    name="run_regression",
    input_schema=RunRegressionInput,
    description=(
        "Run a complete regression workflow: train a model, evaluate performance, "
        "and generate diagnostic plots (feature importance). Returns intermediate "
        "results from each step including metrics like RMSE, MAE, and R²."
    ),
    tier="workflow",
)
def run_regression(state: DataFrameState, params: RunRegressionInput) -> WorkflowResult:
    """
    Execute a regression workflow on a DataFrame.
    
    Steps:
    0. Feature engineering (optional): bin rare categories, target encode categoricals
    1. Train a regression model (dispatched via registry)
    2. Evaluate model performance (RMSE, MAE, R², etc.)
    3. Generate diagnostic plots (feature importance)
    
    The workflow creates a predictions DataFrame and stores the trained model.
    """
    started_at = datetime.now()
    
    # Get config with defaults
    config = params.config or RegressionConfig()
    
    # Resolve DataFrame
    source_name = params.dataframe_name or state.get_active_dataframe_name()
    current_df_name = source_name  # Track which DataFrame to use (may change after FE)
    
    steps: list[WorkflowStepResult] = []
    step_index = 0
    charts: list[ChartArtifact] = []
    dataframes_created: list[str] = []
    models_created: list[str] = []
    
    # Track training result for downstream steps
    model_id: str | None = None
    predictions_df_name: str | None = None
    
    # =========================================================================
    # Step 0: Feature Engineering (optional)
    # =========================================================================
    if config.feature_engineering:
        fe_steps, fe_dfs, current_df_name, step_index = run_feature_engineering_steps(
            state=state,
            config=config.feature_engineering,
            source_name=source_name,
            target_column=params.target_column,
            start_step_index=step_index,
        )
        steps.extend(fe_steps)
        dataframes_created.extend(fe_dfs)
    
    # =========================================================================
    # Step 1: Train Model (registry-based dispatch)
    # =========================================================================
    step_index += 1
    
    # Look up the training tool
    tool_name = REGRESSOR_TOOLS.get(config.model_type)
    if tool_name is None:
        available = ", ".join(REGRESSOR_TOOLS.keys())
        steps.append(WorkflowStepResult(
            step_name="train_model",
            step_index=step_index,
            status="failed",
            duration_ms=0,
            summary=f"Unknown model type: {config.model_type}",
            error=f"Unknown model type '{config.model_type}'. Available: {available}",
        ))
    else:
        try:
            train_func, TrainInputSchema = _get_tool("ml", tool_name)
            train_params = _build_training_params(
                TrainInputSchema,
                current_df_name,  # Use FE'd DataFrame if available
                params.target_column,
                params.feature_columns,
                config,
            )
            
            model_label = MODEL_LABELS.get(config.model_type, config.model_type)
            step_result = run_step(
                step_name="train_model",
                step_index=step_index,
                func=train_func,
                state=state,
                params=train_params,
                summary_template=f"Trained {model_label}",
            )
            steps.append(step_result)
            
            # Extract model info for downstream steps
            if step_result.status == "success" and step_result.result:
                model_id = step_result.result.get("model_id")
                predictions_df_name = step_result.result.get("predictions_dataframe")
                
                if model_id:
                    models_created.append(model_id)
                if predictions_df_name:
                    dataframes_created.append(predictions_df_name)
                    
        except Exception as e:
            steps.append(WorkflowStepResult(
                step_name="train_model",
                step_index=step_index,
                status="failed",
                duration_ms=0,
                summary=f"Failed to train model: {str(e)}",
                error=str(e),
            ))
    
    # =========================================================================
    # Step 2: Evaluate Model
    # =========================================================================
    if model_id and predictions_df_name:
        step_index += 1
        
        try:
            eval_func, EvalInputSchema = _get_tool("ml", "evaluate_regression_model")
            
            # Build evaluation params
            # Prediction column follows pattern: pred_{target_column}
            prediction_col = f"pred_{params.target_column}"
            eval_params_dict = {
                "dataframe_name": predictions_df_name,
                "target_column": params.target_column,
                "prediction_column": prediction_col,
            }
            
            # Filter to schema fields
            schema_fields = set(EvalInputSchema.model_fields.keys())
            valid_params = {k: v for k, v in eval_params_dict.items() if k in schema_fields}
            eval_params = EvalInputSchema(**valid_params)
            
            step_result = run_step(
                step_name="evaluate_model",
                step_index=step_index,
                func=eval_func,
                state=state,
                params=eval_params,
                summary_template="Evaluated regression model performance",
            )
            steps.append(step_result)
            
        except Exception as e:
            steps.append(WorkflowStepResult(
                step_name="evaluate_model",
                step_index=step_index,
                status="failed",
                duration_ms=0,
                summary=f"Failed to evaluate model: {str(e)}",
                error=str(e),
            ))
    
    # =========================================================================
    # Step 3: Generate Plots (if enabled)
    # =========================================================================
    if config.generate_plots and model_id:
        for plot_name in config.plots:
            if plot_name not in PLOT_TOOLS:
                # Skip unknown plot types silently (may not be implemented yet)
                continue
            
            tool_name, chart_type = PLOT_TOOLS[plot_name]
            step_index += 1
            
            try:
                plot_func, PlotInputSchema = _get_tool("plots", tool_name)
                
                # Build plot params
                if plot_name == "feature_importance":
                    plot_params_dict = {"model_id": model_id}
                else:
                    # For other plots that need predictions DataFrame
                    plot_params_dict = {
                        "dataframe_name": predictions_df_name,
                        "true_column": params.target_column,
                        "pred_column": "predicted",
                    }
                
                # Filter to schema fields
                schema_fields = set(PlotInputSchema.model_fields.keys())
                valid_params = {k: v for k, v in plot_params_dict.items() if k in schema_fields}
                plot_params = PlotInputSchema(**valid_params)
                
                step_result, chart = run_chart_step(
                    step_name=f"plot_{plot_name}",
                    step_index=step_index,
                    func=plot_func,
                    state=state,
                    params=plot_params,
                    chart_type=chart_type,
                    summary_template=f"Generated {plot_name.replace('_', ' ')} plot",
                )
                steps.append(step_result)
                if chart:
                    charts.append(chart)
                    
            except Exception as e:
                steps.append(WorkflowStepResult(
                    step_name=f"plot_{plot_name}",
                    step_index=step_index,
                    status="failed",
                    duration_ms=0,
                    summary=f"Failed to generate {plot_name} plot",
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
        suggestion = "Check that the DataFrame exists, has numeric features, and a valid target column."
    elif failed_steps:
        overall_status = "partial_failure"
        failed_names = [s.step_name for s in failed_steps]
        error_summary = f"{len(failed_steps)} step(s) failed: {', '.join(failed_names)}"
        suggestion = "Review failed steps. The trained model may still be usable."
    else:
        overall_status = "success"
        error_summary = None
        suggestion = None
    
    # Build artifacts
    artifacts = WorkflowArtifacts(
        dataframes_created=dataframes_created,
        models_created=models_created,
        charts=charts,
        final_dataframe=predictions_df_name,
    )
    
    return WorkflowResult(
        workflow_name="run_regression",
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
