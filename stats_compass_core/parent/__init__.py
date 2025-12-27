"""
Parent tools module for stats-compass-core.

Parent tools provide category-level access to sub-tools via
describe (schema discovery) and execute (dispatch) patterns.
"""

from stats_compass_core.parent.schemas import (
    CategoryDescription,
    ExecuteCategoryInput,
    ExecuteResult,
    SubToolSchema,
)

# Import tools module to trigger registration
from stats_compass_core.parent import tools  # noqa: F401

__all__ = [
    "CategoryDescription",
    "SubToolSchema",
    "ExecuteCategoryInput",
    "ExecuteResult",
    # Parent tools are registered via decorator, access them via registry
]
