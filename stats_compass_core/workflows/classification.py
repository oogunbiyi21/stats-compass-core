"""
Classification Workflow.

Orchestrates model training, evaluation, and visualization for
classification tasks using registry-based dispatch.
"""

from datetime import datetime
from typing import Any

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState

from .configs import ClassificationConfig, FeatureEngineeringConfig
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

CLASSIFIER_TOOLS: dict[str, str] = {
    "random_forest": "train_random_forest_classifier",
    "gradient_boosting": "train_gradient_boosting_classifier",
    "logistic": "train_logistic_regression",
    # Possible future additions:
    # "svm": "train_svm_classifier",
    # "xgboost": "train_xgboost_classifier",
    # "lightgbm": "train_lightgbm_classifier",
}

# Human-readable labels for model types
MODEL_LABELS: dict[str, str] = {
    "random_forest": "Random Forest Classifier",
    "gradient_boosting": "Gradient Boosting Classifier",
    "logistic": "Logistic Regression",
}

# Plot tools - maps plot config names to registry tool names
PLOT_TOOLS: dict[str, tuple[str, str]] = {
    # config_name: (tool_name, chart_type_label)
    "confusion_matrix": ("confusion_matrix_plot", "confusion_matrix"),
    "roc": ("roc_curve_plot", "roc_curve"),
    "precision_recall": ("precision_recall_curve_plot", "precision_recall_curve"),
    "feature_importance": ("feature_importance", "feature_importance"),
}


# =============================================================================
# Input Schema
# =============================================================================

class RunClassificationInput(StrictToolInput):
    """Input schema for run_classification workflow."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to train on. Uses active if not specified."
    )
    target_column: str = Field(
        description="Name of the target column (class labels)"
    )
    feature_columns: list[str] | None = Field(
        default=None,
        description="List of feature columns. If None, uses all numeric columns except target."
    )
    config: ClassificationConfig | None = Field(
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
    config: ClassificationConfig,
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
    name="run_classification",
    input_schema=RunClassificationInput,
    description=(
        "Run a complete classification workflow: train a model, evaluate performance, "
        "and generate diagnostic plots (confusion matrix, ROC curve, precision-recall curve, "
        "feature importance). Returns intermediate results from each step."
    ),
    tier="workflow",
)
def run_classification(state: DataFrameState, params: RunClassificationInput) -> WorkflowResult:
    """
    Execute a classification workflow on a DataFrame.
    
    Steps:
    0. Feature engineering (optional): bin rare categories, target encode categoricals
    1. Train a classification model (dispatched via registry)
    2. Evaluate model performance (accuracy, precision, recall, F1)
    3. Generate diagnostic plots (confusion matrix, ROC, PR, feature importance)
    
    The workflow creates a predictions DataFrame and stores the trained model.
    """
    started_at = datetime.now()
    
    # Get config with defaults
    config = params.config or ClassificationConfig()
    
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
    prediction_column: str | None = None
    probability_columns: list[str] | None = None
    class_labels: list[Any] | None = None
    
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
    tool_name = CLASSIFIER_TOOLS.get(config.model_type)
    if tool_name is None:
        available = ", ".join(CLASSIFIER_TOOLS.keys())
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
            train_func, InputSchema = _get_tool("ml", tool_name)
            train_params = _build_training_params(
                input_schema=InputSchema,
                source_name=current_df_name,  # Use FE'd DataFrame if available
                target_column=params.target_column,
                feature_columns=params.feature_columns,
                config=config,
            )
            
            model_label = MODEL_LABELS.get(config.model_type, config.model_type)
            step_result = run_step(
                step_name="train_model",
                step_index=step_index,
                func=train_func,
                state=state,
                params=train_params,
                summary_template=f"Trained {model_label} model",
            )
            steps.append(step_result)
            
            # Extract training info for downstream steps
            if step_result.status == "success" and step_result.result:
                result_data = step_result.result
                model_id = result_data.get("model_id")
                predictions_df_name = result_data.get("predictions_dataframe")
                prediction_column = result_data.get("prediction_column")
                probability_columns = result_data.get("probability_columns")
                class_labels = result_data.get("class_labels")
                
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
    if predictions_df_name and prediction_column:
        step_index += 1
        
        try:
            eval_func, EvalInputSchema = _get_tool("ml", "evaluate_classification_model")
            eval_params = EvalInputSchema(
                dataframe_name=predictions_df_name,
                target_column=params.target_column,
                prediction_column=prediction_column,
            )
            
            step_result = run_step(
                step_name="evaluate_model",
                step_index=step_index,
                func=eval_func,
                state=state,
                params=eval_params,
                summary_template="Evaluated model performance",
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
    # Step 3+: Generate Plots (registry-based dispatch)
    # =========================================================================
    if config.generate_plots and predictions_df_name and prediction_column:
        
        for plot_name in config.plots:
            if plot_name not in PLOT_TOOLS:
                continue
                
            tool_name, chart_type = PLOT_TOOLS[plot_name]
            step_index += 1
            
            try:
                plot_func, PlotInputSchema = _get_tool("plots", tool_name)
                
                # Build plot-specific parameters
                if plot_name == "confusion_matrix":
                    plot_params = PlotInputSchema(
                        dataframe_name=predictions_df_name,
                        true_column=params.target_column,
                        pred_column=prediction_column,
                    )
                    
                elif plot_name in ("roc", "precision_recall"):
                    # These require probability columns - binary classification only
                    if not (probability_columns and len(probability_columns) == 2 
                            and class_labels and len(class_labels) == 2):
                        steps.append(WorkflowStepResult(
                            step_name=chart_type,
                            step_index=step_index,
                            status="skipped",
                            duration_ms=0,
                            summary=f"{chart_type} skipped: only supported for binary classification",
                        ))
                        continue
                    
                    # Use probability of positive class (second class)
                    pos_prob_col = probability_columns[1]
                    plot_params = PlotInputSchema(
                        dataframe_name=predictions_df_name,
                        true_column=params.target_column,
                        prob_column=pos_prob_col,
                        model_id=model_id or "model",
                    )
                    
                elif plot_name == "feature_importance":
                    if not model_id:
                        steps.append(WorkflowStepResult(
                            step_name=chart_type,
                            step_index=step_index,
                            status="skipped",
                            duration_ms=0,
                            summary="Feature importance skipped: no model available",
                        ))
                        continue
                    plot_params = PlotInputSchema(model_id=model_id)
                    
                else:
                    continue
                
                step_result, chart = run_chart_step(
                    step_name=chart_type,
                    step_index=step_index,
                    func=plot_func,
                    state=state,
                    params=plot_params,
                    chart_type=chart_type,
                    summary_template=f"Generated {chart_type.replace('_', ' ')}",
                )
                steps.append(step_result)
                if chart:
                    charts.append(chart)
                    
            except Exception as e:
                steps.append(WorkflowStepResult(
                    step_name=chart_type,
                    step_index=step_index,
                    status="failed",
                    duration_ms=0,
                    summary=f"Failed to generate {chart_type}: {str(e)}",
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
        dataframes_created=dataframes_created,
        models_created=models_created,
        charts=charts,
    )
    
    # Build summary
    summary_parts = [f"Classification workflow completed with status: {overall_status}"]
    if model_id:
        summary_parts.append(f"Model: {model_id}")
    if predictions_df_name:
        summary_parts.append(f"Predictions: {predictions_df_name}")
    summary_parts.append(f"Steps: {len(success_steps)} succeeded, {len(failed_steps)} failed")
    if charts:
        summary_parts.append(f"Charts: {len(charts)} generated")
    
    return WorkflowResult(
        workflow_name="run_classification",
        status=overall_status,
        started_at=started_at,
        completed_at=completed_at,
        total_duration_ms=total_duration_ms,
        input_dataframe=source_name,
        steps=steps,
        artifacts=artifacts,
    )
