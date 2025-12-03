"""
Tool registry for auto-loading and managing all stats-compass-core tools.
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict


class ToolMetadata(BaseModel):
    """Metadata for a registered tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    category: str
    function: Any
    input_schema: type[BaseModel] | None = None
    description: str = ""


class ToolRegistry:
    """Registry that automatically discovers and manages all tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolMetadata] = {}
        self._categories = ["cleaning", "transforms", "eda", "ml", "plots"]

    def register(
        self,
        category: str,
        name: str | None = None,
        input_schema: type[BaseModel] | None = None,
        description: str = "",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to register a tool function.

        Args:
            category: Tool category (cleaning, transforms, eda, ml, plots)
            name: Optional tool name (defaults to function name)
            input_schema: Optional Pydantic schema for input validation
            description: Optional tool description

        Returns:
            Decorated function
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            metadata = ToolMetadata(
                name=tool_name,
                category=category,
                function=func,
                input_schema=input_schema,
                description=description or func.__doc__ or "",
            )
            self._tools[f"{category}.{tool_name}"] = metadata
            return func

        return decorator

    def get_tool(self, category: str, name: str) -> Callable[..., Any] | None:
        """Get a specific tool by category and name."""
        key = f"{category}.{name}"
        metadata = self._tools.get(key)
        return metadata.function if metadata else None

    def list_tools(self, category: str | None = None) -> list[ToolMetadata]:
        """
        List all registered tools, optionally filtered by category.

        Args:
            category: Optional category to filter by

        Returns:
            List of tool metadata
        """
        if category:
            return [
                meta for key, meta in self._tools.items() if meta.category == category
            ]
        return list(self._tools.values())

    def get_categories(self) -> list[str]:
        """Get list of available categories."""
        return self._categories.copy()

    def auto_discover(self) -> None:
        """
        Automatically discover and import all tool modules.

        This method walks through all category folders and imports
        all Python modules to trigger tool registration.
        """
        package_dir = Path(__file__).parent

        for category in self._categories:
            category_path = package_dir / category
            if not category_path.exists():
                continue

            # Import all modules in the category
            for module_info in pkgutil.iter_modules([str(category_path)]):
                if module_info.name.startswith("_"):
                    continue

                module_name = f"stats_compass_core.{category}.{module_info.name}"
                try:
                    importlib.import_module(module_name)
                except Exception as e:
                    print(f"Warning: Failed to import {module_name}: {e}")


# Global registry instance
registry = ToolRegistry()
