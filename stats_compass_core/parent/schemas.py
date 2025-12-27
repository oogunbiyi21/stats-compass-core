"""
Schemas for parent tools (describe_* and execute_*).

These models define the input/output contracts for category-level
tool discovery and dispatch.
"""

from typing import Any

from pydantic import BaseModel, Field


class SubToolSchema(BaseModel):
    """Schema describing a single sub-tool within a category."""

    name: str = Field(description="Tool name (e.g., 'drop_na')")
    description: str = Field(description="What the tool does")
    parameters: dict[str, Any] = Field(
        description="JSON Schema of the tool's input parameters"
    )
    required_params: list[str] = Field(
        default_factory=list,
        description="List of required parameter names"
    )
    example: dict[str, Any] | None = Field(
        default=None,
        description="Example invocation parameters"
    )


class CategoryDescription(BaseModel):
    """
    Response from describe_* parent tools.
    
    Provides full schema information for all sub-tools in a category,
    enabling agents to discover available operations and their parameters.
    """

    category: str = Field(description="Category name (e.g., 'cleaning')")
    description: str = Field(description="What this category of tools does")
    tool_count: int = Field(description="Number of sub-tools in this category")
    tools: list[SubToolSchema] = Field(
        description="Full schemas for all sub-tools"
    )


class ExecuteCategoryInput(BaseModel):
    """
    Input schema for execute_* parent tools.
    
    Used to dispatch to a specific sub-tool within a category.
    """

    tool_name: str = Field(
        description="Name of the sub-tool to execute (e.g., 'drop_na')"
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to the sub-tool"
    )
    dataframe_name: str | None = Field(
        default=None,
        description="Override the active DataFrame for this operation"
    )


class ExecuteResult(BaseModel):
    """
    Wrapper result from execute_* parent tools.
    
    Contains the sub-tool's actual result plus metadata about the dispatch.
    """

    success: bool = Field(description="Whether the sub-tool executed successfully")
    tool_name: str = Field(description="Name of the sub-tool that was executed")
    category: str = Field(description="Category of the sub-tool")
    result: dict[str, Any] | None = Field(
        default=None,
        description="The sub-tool's result (serialized)"
    )
    error: str | None = Field(
        default=None,
        description="Error message if execution failed"
    )
    error_type: str | None = Field(
        default=None,
        description="Type of error (e.g., 'ValidationError', 'ToolNotFound')"
    )
