"""
Regression Workflow.

Declares what makes regression different from classification; the sequence
itself lives in supervised.py.
"""

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState

from .configs import RegressionConfig
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
    # RegressionConfig.plots also defaults to residuals and
    # predicted_vs_actual. Neither has a plot tool yet, so both are reported
    # as skipped steps rather than dropped silently.
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
        description=(
            "List of feature columns. Only these are binned and encoded. If None, "
            "every categorical is encoded and every numeric column except the "
            "target is used, and the result carries a FEATURES_INFERRED warning."
        )
    )
    config: RegressionConfig | None = Field(
        default=None,
        description="Optional configuration to customize the workflow. Uses sensible defaults if not provided."
    )



# =============================================================================
# Plot Dispatch
# =============================================================================

def _build_plot_params(ctx: PlotContext) -> PlotDecision:
    """Turn a requested regression plot into parameters, or a reason not to.

    Only feature_importance has a registered tool; it describes the model's
    learned structure rather than any rows, so it takes a model id. Anything
    else configured is reported as skipped by the shared skeleton before it
    reaches here.
    """
    if ctx.plot_name == "feature_importance":
        if not ctx.model_id:
            return PlotDecision(skip_reason="no model available")
        return PlotDecision(params=ctx.schema(model_id=ctx.model_id))

    return PlotDecision(skip_reason=f"unsupported plot type '{ctx.plot_name}'")


REGRESSION_SPEC = SupervisedSpec(
    kind="regression",
    tool_map=REGRESSOR_TOOLS,
    model_labels=MODEL_LABELS,
    evaluator_tool="evaluate_regression_model",
    plot_map=PLOT_TOOLS,
    build_plot_params=_build_plot_params,
    default_config=RegressionConfig,
    failure_suggestion=(
        "Check that the DataFrame exists, has numeric features, and a valid "
        "target column."
    ),
)


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
    return run_supervised_workflow(state, params, REGRESSION_SPEC)
