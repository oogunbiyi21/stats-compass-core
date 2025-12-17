"""
Base classes for Stats Compass tools.
"""

from pydantic import BaseModel, ConfigDict


class StrictToolInput(BaseModel):
    """
    Base class for all top-level tool input models.
    Enforces strict schema validation (no extra fields allowed).
    """
    model_config = ConfigDict(extra="forbid")


class ToolComponent(BaseModel):
    """
    Base class for nested components in tool inputs.
    Must allow extra fields to avoid JSON Schema $ref conflicts.
    """
    model_config = ConfigDict(extra="ignore")
