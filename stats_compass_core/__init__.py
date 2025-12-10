"""
stats-compass-core: A clean toolkit of deterministic pandas-based data tools.

This package provides organized tools for data cleaning, transformation, EDA,
machine learning, and plotting, with automatic tool discovery and registration.

All tools are MCP-compatible: they accept a DataFrameState as first argument
and return JSON-serializable Pydantic models.
"""

from stats_compass_core.registry import ToolRegistry, registry
from stats_compass_core.results import (
    ChartResult,
    CorrelationsResult,
    DataFrameListResult,
    DataFrameLoadResult,
    DataFrameMutationResult,
    DataFrameQueryResult,
    DataFrameSampleResult,
    DataFrameSchemaResult,
    DescribeResult,
    ModelListResult,
    ModelPredictionResult,
    ModelTrainingResult,
    OperationError,
)
from stats_compass_core.state import DataFrameState

__version__ = "0.1.0"
__all__ = [
    # Core classes
    "ToolRegistry",
    "registry",
    "DataFrameState",
    # Result types
    "DataFrameMutationResult",
    "DataFrameQueryResult",
    "DataFrameSchemaResult",
    "DataFrameSampleResult",
    "DataFrameLoadResult",
    "DataFrameListResult",
    "DescribeResult",
    "CorrelationsResult",
    "ChartResult",
    "ModelTrainingResult",
    "ModelPredictionResult",
    "ModelListResult",
    "OperationError",
]

# Auto-discover and register all tools
registry.auto_discover()
