"""
Workflow tools and result types for stats-compass-core.

Workflows are high-level pipelines that orchestrate multiple sub-tools
to accomplish common data science tasks.
"""

from stats_compass_core.workflows.configs import (
    ClassificationConfig,
    CompareModelsConfig,
    EDAConfig,
    FeatureEngineeringConfig,
    ImputationConfig,
    OutlierConfig,
    PreprocessingConfig,
    RegressionConfig,
    TimeSeriesConfig,
)
from stats_compass_core.workflows.results import (
    ChartArtifact,
    StepStatus,
    WorkflowArtifacts,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStepResult,
)

# Import workflow tools to ensure they get registered
from stats_compass_core.workflows.eda_report import run_eda_report
from stats_compass_core.workflows.preprocessing import run_preprocessing
from stats_compass_core.workflows.classification import run_classification
from stats_compass_core.workflows.regression import run_regression
from stats_compass_core.workflows.timeseries import run_timeseries_forecast

__all__ = [
    # Result types
    "WorkflowStepResult",
    "WorkflowResult",
    "WorkflowArtifacts",
    "ChartArtifact",
    "StepStatus",
    "WorkflowStatus",
    # Config types
    "PreprocessingConfig",
    "ImputationConfig",
    "OutlierConfig",
    "FeatureEngineeringConfig",
    "EDAConfig",
    "ClassificationConfig",
    "RegressionConfig",
    "TimeSeriesConfig",
    "CompareModelsConfig",
    # Workflow tools
    "run_eda_report",
    "run_preprocessing",
    "run_classification",
    "run_regression",
    "run_timeseries_forecast",
]
