"""
Parent tools for category-level tool discovery and dispatch.

These tools provide:
- describe_* tools: Return schemas for all sub-tools in a category
- execute_* tools: Dispatch to a specific sub-tool with parameters
"""

from typing import Any

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.parent.schemas import (
    CategoryDescription,
    ExecuteCategoryInput,
    ExecuteResult,
    SubToolSchema,
)
from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState


# =============================================================================
# Category Descriptions (for describe_* tools)
# =============================================================================

CATEGORY_DESCRIPTIONS = {
    "cleaning": "Tools for cleaning and preparing data: handling missing values, duplicates, and outliers.",
    "transforms": "Tools for transforming data: filtering, grouping, pivoting, encoding, and column operations.",
    "eda": "Tools for exploratory data analysis: statistics, correlations, hypothesis tests, and data quality.",
    "plots": "Tools for data visualization: histograms, scatter plots, bar charts, and model evaluation plots.",
    "ml": "Tools for machine learning: training models, making predictions, and evaluating performance.",
    "data": "Tools for data management: loading, saving, and inspecting DataFrames.",
}


# =============================================================================
# Helper Functions
# =============================================================================

def _get_sub_tools_for_category(category: str) -> list[SubToolSchema]:
    """Get all sub-tools for a category with their schemas."""
    tools = registry.list_tools_by_tier(tiers=["sub"], category=category)
    
    sub_tool_schemas = []
    for tool in tools:
        # Get the JSON schema for the input
        if tool.input_schema:
            schema = tool.input_schema.model_json_schema()
            properties = schema.get("properties", {})
            required = schema.get("required", [])
        else:
            properties = {}
            required = []
        
        sub_tool_schemas.append(SubToolSchema(
            name=tool.name,
            description=tool.description,
            parameters=properties,
            required_params=required,
        ))
    
    return sub_tool_schemas


def _build_category_description(category: str) -> CategoryDescription:
    """Build a CategoryDescription for a given category."""
    sub_tools = _get_sub_tools_for_category(category)
    return CategoryDescription(
        category=category,
        description=CATEGORY_DESCRIPTIONS.get(category, f"Tools in the {category} category"),
        tool_count=len(sub_tools),
        tools=sub_tools,
    )


def _execute_sub_tool(
    state: DataFrameState,
    category: str,
    tool_name: str,
    params: dict[str, Any],
) -> ExecuteResult:
    """Execute a sub-tool and return the result."""
    try:
        # Get the tool metadata
        metadata = registry.get_tool_metadata(category, tool_name)
        if metadata is None:
            # Get available tools to help the LLM
            available_tools = [t.name for t in registry.list_tools_by_tier(tiers=["sub"], category=category)]
            available_list = ", ".join(available_tools) if available_tools else "none"
            return ExecuteResult(
                success=False,
                tool_name=tool_name,
                category=category,
                error=(
                    f"Tool '{tool_name}' not found in category '{category}'. "
                    f"Available tools: [{available_list}]. "
                    f"Call describe_{category} first to see all available tools and their parameters."
                ),
                error_type="ToolNotFound",
            )
        
        # Validate that this is a sub-tool
        if metadata.tier != "sub":
            return ExecuteResult(
                success=False,
                tool_name=tool_name,
                category=category,
                error=f"Tool '{tool_name}' is not a sub-tool (tier={metadata.tier})",
                error_type="InvalidTier",
            )
        
        # Validate params against schema
        if metadata.input_schema:
            validated_params = metadata.input_schema(**params)
        else:
            validated_params = params
        
        # Execute the tool
        result = metadata.function(state, validated_params)
        
        # Serialize the result
        if hasattr(result, "model_dump"):
            result_data = result.model_dump()
        elif hasattr(result, "to_dict"):
            result_data = result.to_dict()
        elif isinstance(result, dict):
            result_data = result
        else:
            result_data = {"result": str(result)}
        
        return ExecuteResult(
            success=True,
            tool_name=tool_name,
            category=category,
            result=result_data,
        )
        
    except Exception as e:
        return ExecuteResult(
            success=False,
            tool_name=tool_name,
            category=category,
            error=str(e),
            error_type=type(e).__name__,
        )


# =============================================================================
# Describe Input Schema (shared by all describe_* tools)
# =============================================================================

class DescribeCategoryInput(StrictToolInput):
    """Input schema for describe_* tools. No parameters needed."""
    pass


# =============================================================================
# CLEANING Category
# =============================================================================

@registry.register(
    category="cleaning",
    name="describe_cleaning",
    input_schema=DescribeCategoryInput,
    description="Get schemas for all cleaning sub-tools: drop_na, impute, dedupe, handle_outliers, etc.",
    tier="parent",
)
def describe_cleaning(state: DataFrameState, params: DescribeCategoryInput) -> CategoryDescription:
    """Return schemas for all cleaning sub-tools."""
    return _build_category_description("cleaning")


@registry.register(
    category="cleaning",
    name="execute_cleaning",
    input_schema=ExecuteCategoryInput,
    description="Execute a cleaning sub-tool. Use describe_cleaning first to see available tools and parameters.",
    tier="parent",
)
def execute_cleaning(state: DataFrameState, params: ExecuteCategoryInput) -> ExecuteResult:
    """Execute a cleaning sub-tool."""
    return _execute_sub_tool(state, "cleaning", params.tool_name, params.params)


# =============================================================================
# TRANSFORMS Category
# =============================================================================

@registry.register(
    category="transforms",
    name="describe_transforms",
    input_schema=DescribeCategoryInput,
    description="Get schemas for all transform sub-tools: filter, groupby, pivot, encode, etc.",
    tier="parent",
)
def describe_transforms(state: DataFrameState, params: DescribeCategoryInput) -> CategoryDescription:
    """Return schemas for all transform sub-tools."""
    return _build_category_description("transforms")


@registry.register(
    category="transforms",
    name="execute_transforms",
    input_schema=ExecuteCategoryInput,
    description="Execute a transform sub-tool. Use describe_transforms first to see available tools and parameters.",
    tier="parent",
)
def execute_transforms(state: DataFrameState, params: ExecuteCategoryInput) -> ExecuteResult:
    """Execute a transform sub-tool."""
    return _execute_sub_tool(state, "transforms", params.tool_name, params.params)


# =============================================================================
# EDA Category
# =============================================================================

@registry.register(
    category="eda",
    name="describe_eda",
    input_schema=DescribeCategoryInput,
    description="Get schemas for all EDA sub-tools: describe, correlations, hypothesis tests, data quality, etc.",
    tier="parent",
)
def describe_eda(state: DataFrameState, params: DescribeCategoryInput) -> CategoryDescription:
    """Return schemas for all EDA sub-tools."""
    return _build_category_description("eda")


@registry.register(
    category="eda",
    name="execute_eda",
    input_schema=ExecuteCategoryInput,
    description="Execute an EDA sub-tool. Use describe_eda first to see available tools and parameters.",
    tier="parent",
)
def execute_eda(state: DataFrameState, params: ExecuteCategoryInput) -> ExecuteResult:
    """Execute an EDA sub-tool."""
    return _execute_sub_tool(state, "eda", params.tool_name, params.params)


# =============================================================================
# PLOTS Category
# =============================================================================

@registry.register(
    category="plots",
    name="describe_plots",
    input_schema=DescribeCategoryInput,
    description="Get schemas for all visualization sub-tools: histogram, scatter, bar chart, ROC curve, etc.",
    tier="parent",
)
def describe_plots(state: DataFrameState, params: DescribeCategoryInput) -> CategoryDescription:
    """Return schemas for all visualization sub-tools."""
    return _build_category_description("plots")


@registry.register(
    category="plots",
    name="execute_plots",
    input_schema=ExecuteCategoryInput,
    description="Execute a visualization sub-tool. Use describe_plots first to see available tools and parameters.",
    tier="parent",
)
def execute_plots(state: DataFrameState, params: ExecuteCategoryInput) -> ExecuteResult:
    """Execute a visualization sub-tool."""
    return _execute_sub_tool(state, "plots", params.tool_name, params.params)


# =============================================================================
# ML Category
# =============================================================================

@registry.register(
    category="ml",
    name="describe_ml",
    input_schema=DescribeCategoryInput,
    description="Get schemas for all ML sub-tools: train models, evaluate, predict, save models, etc.",
    tier="parent",
)
def describe_ml(state: DataFrameState, params: DescribeCategoryInput) -> CategoryDescription:
    """Return schemas for all ML sub-tools."""
    return _build_category_description("ml")


@registry.register(
    category="ml",
    name="execute_ml",
    input_schema=ExecuteCategoryInput,
    description="Execute an ML sub-tool. Use describe_ml first to see available tools and parameters.",
    tier="parent",
)
def execute_ml(state: DataFrameState, params: ExecuteCategoryInput) -> ExecuteResult:
    """Execute an ML sub-tool."""
    return _execute_sub_tool(state, "ml", params.tool_name, params.params)


# =============================================================================
# DATA Category (for non-util data tools)
# =============================================================================

@registry.register(
    category="data",
    name="describe_data",
    input_schema=DescribeCategoryInput,
    description="Get schemas for data manipulation sub-tools: add_column, drop_columns, rename_columns, merge, concat.",
    tier="parent",
)
def describe_data(state: DataFrameState, params: DescribeCategoryInput) -> CategoryDescription:
    """Return schemas for data manipulation sub-tools."""
    return _build_category_description("data")


@registry.register(
    category="data",
    name="execute_data",
    input_schema=ExecuteCategoryInput,
    description="Execute a data manipulation sub-tool. Use describe_data first to see available tools and parameters.",
    tier="parent",
)
def execute_data(state: DataFrameState, params: ExecuteCategoryInput) -> ExecuteResult:
    """Execute a data manipulation sub-tool."""
    return _execute_sub_tool(state, "data", params.tool_name, params.params)
