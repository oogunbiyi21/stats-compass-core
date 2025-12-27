"""
Result types for workflow tools.

These models represent the output of high-level workflow pipelines
that orchestrate multiple sub-tools.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# Status types
StepStatus = Literal["success", "failed", "skipped"]
WorkflowStatus = Literal["success", "partial_failure", "failed"]


class ChartArtifact(BaseModel):
    """A chart produced by a workflow step."""

    chart_type: str = Field(description="Type of chart: histogram, scatter, bar, confusion_matrix, etc.")
    title: str | None = Field(default=None, description="Chart title")
    description: str | None = Field(default=None, description="Chart description")
    base64_image: str | None = Field(default=None, description="PNG image encoded as base64")
    format: str = Field(default="png", description="Image format")


class WorkflowStepResult(BaseModel):
    """Result from a single step in a workflow."""

    step_name: str = Field(description="Name of the tool/operation executed")
    step_index: int = Field(description="Position in workflow (1-indexed)")
    status: StepStatus = Field(description="success, failed, or skipped")
    duration_ms: int | None = Field(
        default=None,
        description="Execution time in milliseconds"
    )
    summary: str = Field(description="Human-readable summary of what happened")
    result: dict[str, Any] | None = Field(
        default=None,
        description="Serialized tool output (if successful)"
    )
    image_base64: str | None = Field(
        default=None,
        description="Chart image if this step produced one"
    )
    error: str | None = Field(
        default=None,
        description="Error message if step failed"
    )
    skip_reason: str | None = Field(
        default=None,
        description="Why this step was skipped"
    )
    dataframe_produced: str | None = Field(
        default=None,
        description="Name of DataFrame created by this step, if any"
    )


class WorkflowArtifacts(BaseModel):
    """Summary of all artifacts produced by a workflow."""

    dataframes_created: list[str] = Field(
        default_factory=list,
        description="Names of DataFrames created during workflow"
    )
    models_created: list[str] = Field(
        default_factory=list,
        description="IDs of ML models trained during workflow"
    )
    charts_generated: int = Field(
        default=0,
        description="Total number of charts generated"
    )
    charts: list[ChartArtifact] = Field(
        default_factory=list,
        description="All chart artifacts with base64 images"
    )
    final_dataframe: str | None = Field(
        default=None,
        description="The primary output DataFrame from the workflow"
    )


class WorkflowResult(BaseModel):
    """
    Top-level result from a workflow tool.
    
    Contains the full execution history, all artifacts produced,
    and status/error information for debugging.
    """

    workflow_name: str = Field(description="Name of the workflow executed")
    status: WorkflowStatus = Field(
        description="Overall status: success, partial_failure, or failed"
    )
    started_at: datetime = Field(description="When workflow execution began")
    completed_at: datetime = Field(description="When workflow execution ended")
    total_duration_ms: int = Field(description="Total execution time in milliseconds")
    input_dataframe: str = Field(description="Name of the input DataFrame")
    steps: list[WorkflowStepResult] = Field(
        description="Results from each step in execution order"
    )
    artifacts: WorkflowArtifacts = Field(
        description="Summary of all outputs produced"
    )
    error_summary: str | None = Field(
        default=None,
        description="Top-level error explanation if workflow failed"
    )
    suggestion: str | None = Field(
        default=None,
        description="Suggested recovery action for agent"
    )
    recoverable: bool = Field(
        default=True,
        description="Whether agent can retry with different parameters"
    )

    @property
    def successful_steps(self) -> int:
        """Count of steps that completed successfully."""
        return sum(1 for s in self.steps if s.status == "success")

    @property
    def failed_steps(self) -> int:
        """Count of steps that failed."""
        return sum(1 for s in self.steps if s.status == "failed")

    @property
    def skipped_steps(self) -> int:
        """Count of steps that were skipped."""
        return sum(1 for s in self.steps if s.status == "skipped")
