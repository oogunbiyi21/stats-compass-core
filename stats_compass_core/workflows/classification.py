"""
Classification Workflow.

Declares what makes classification different from regression; the sequence
itself lives in supervised.py.
"""


from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState

from .configs import ClassificationConfig
from .results import WorkflowResult
from .supervised import (
    PlotContext,
    PlotDecision,
    SupervisedSpec,
    run_supervised_workflow,
)

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
# Plot Dispatch
# =============================================================================

def _build_plot_params(ctx: PlotContext) -> PlotDecision:
    """Turn a requested classification plot into parameters, or a reason not to.

    ROC and precision-recall need a probability column for a positive class,
    which only exists for binary problems. Returning a reason rather than
    silently continuing keeps the skip visible in the step list.
    """
    if ctx.plot_name == "confusion_matrix":
        return PlotDecision(params=ctx.schema(
            dataframe_name=ctx.dataframe_name,
            true_column=ctx.target_column,
            pred_column=ctx.prediction_column,
        ))

    if ctx.plot_name in ("roc", "precision_recall"):
        is_binary = (
            ctx.probability_columns and len(ctx.probability_columns) == 2
            and ctx.class_labels and len(ctx.class_labels) == 2
        )
        if not is_binary:
            return PlotDecision(
                skip_reason="only supported for binary classification"
            )
        return PlotDecision(params=ctx.schema(
            dataframe_name=ctx.dataframe_name,
            true_column=ctx.target_column,
            prob_column=ctx.probability_columns[1],  # positive class
            model_id=ctx.model_id or "model",
        ))

    if ctx.plot_name == "feature_importance":
        if not ctx.model_id:
            return PlotDecision(skip_reason="no model available")
        return PlotDecision(params=ctx.schema(model_id=ctx.model_id))

    return PlotDecision(skip_reason=f"unsupported plot type '{ctx.plot_name}'")


CLASSIFICATION_SPEC = SupervisedSpec(
    kind="classification",
    tool_map=CLASSIFIER_TOOLS,
    model_labels=MODEL_LABELS,
    evaluator_tool="evaluate_classification_model",
    plot_map=PLOT_TOOLS,
    build_plot_params=_build_plot_params,
    default_config=ClassificationConfig,
    failure_suggestion=(
        "Check that the DataFrame exists, has usable features, and a target "
        "column with at least two classes."
    ),
)


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
def run_classification(
    state: DataFrameState,
    params: RunClassificationInput,
) -> WorkflowResult:
    """
    Execute a classification workflow on a DataFrame.

    Steps:
    0. Feature engineering (optional): bin rare categories, target encode categoricals
    1. Train a classification model (dispatched via registry)
    2. Evaluate model performance (accuracy, precision, recall, F1)
    3. Generate diagnostic plots (confusion matrix, ROC, PR, feature importance)

    The workflow creates a predictions DataFrame and stores the trained model.
    """
    return run_supervised_workflow(state, params, CLASSIFICATION_SPEC)
