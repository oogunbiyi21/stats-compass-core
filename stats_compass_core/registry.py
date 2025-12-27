"""
Tool registry for auto-loading and managing all stats-compass-core tools.
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict


# Tool tier determines MCP exposure level
ToolTier = Literal["util", "parent", "workflow", "sub"]


class ToolMetadata(BaseModel):
    """Metadata for a registered tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    category: str
    function: Any
    input_schema: type[BaseModel] | None = None
    description: str = ""
    tier: ToolTier = "sub"  # Default to sub for backward compatibility


class ToolRegistry:
    """Registry that automatically discovers and manages all tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolMetadata] = {}
        self._categories = ["data", "cleaning", "transforms", "eda", "ml", "plots"]

    def register(
        self,
        category: str,
        name: str | None = None,
        input_schema: type[BaseModel] | None = None,
        description: str = "",
        tier: ToolTier = "sub",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to register a tool function.

        Args:
            category: Tool category (data, cleaning, transforms, eda, ml, plots)
            name: Optional tool name (defaults to function name)
            input_schema: Optional Pydantic schema for input validation
            description: Optional tool description
            tier: Tool tier for MCP exposure (util, parent, workflow, sub)
                  - util: Always exposed, essential tools (load_csv, list_dataframes)
                  - parent: Category controllers (describe_*, execute_*)
                  - workflow: High-level pipelines (run_eda_report, run_preprocessing)
                  - sub: Granular tools, accessed via execute_* (default)

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
                tier=tier,
            )
            self._tools[f"{category}.{tool_name}"] = metadata
            # Mark function as a registered tool for auto-discovery filtering
            func.__tool_registered__ = True  # type: ignore
            return func

        return decorator

    def get_tool(self, category: str, name: str) -> Callable[..., Any] | None:
        """Get a specific tool by category and name."""
        key = f"{category}.{name}"
        metadata = self._tools.get(key)
        return metadata.function if metadata else None

    def get_tool_metadata(self, category: str, name: str) -> ToolMetadata | None:
        """Get metadata for a specific tool by category and name."""
        key = f"{category}.{name}"
        return self._tools.get(key)

    def invoke(
        self,
        category: str,
        name: str,
        state: Any,
        params: dict[str, Any] | BaseModel
    ) -> Any:
        """
        Invoke a tool with the given state and parameters.
        
        This is the primary way to call tools from an MCP server or other
        integrations. It handles:
        - Looking up the tool by category/name
        - Validating parameters against the input schema
        - Injecting the state as the first argument
        
        Args:
            category: Tool category
            name: Tool name
            state: DataFrameState instance to inject
            params: Tool parameters (dict or Pydantic model)
        
        Returns:
            Tool result (JSON-serializable Pydantic model)
        
        Raises:
            ValueError: If tool not found
            ValidationError: If params don't match schema
        """
        metadata = self.get_tool_metadata(category, name)
        if metadata is None:
            raise ValueError(f"Tool not found: {category}.{name}")

        # Validate and convert params if needed
        if metadata.input_schema:
            if isinstance(params, dict):
                validated_params = metadata.input_schema(**params)
            elif isinstance(params, metadata.input_schema):
                validated_params = params
            else:
                raise TypeError(
                    f"Expected dict or {metadata.input_schema.__name__}, "
                    f"got {type(params).__name__}"
                )
        else:
            validated_params = params

        # Call the tool with state injected
        return metadata.function(state, validated_params)

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

    def list_tools_by_tier(
        self, 
        tiers: list[ToolTier] | None = None,
        category: str | None = None
    ) -> list[ToolMetadata]:
        """
        List tools filtered by tier(s) and optionally by category.
        
        Args:
            tiers: List of tiers to include. If None, returns all tools.
            category: Optional category to filter by.
        
        Returns:
            List of tool metadata matching the filters.
        """
        tools = self.list_tools(category)
        if tiers is not None:
            tools = [t for t in tools if t.tier in tiers]
        return tools

    def list_exposed_tools(self) -> list[ToolMetadata]:
        """
        List tools meant for direct MCP exposure.
        
        Returns tools with tiers: util, parent, workflow.
        Sub-tools are excluded (accessed via execute_* dispatchers).
        """
        return self.list_tools_by_tier(tiers=["util", "parent", "workflow"])

    def get_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """
        Get JSON schemas for all tools (useful for MCP tool definitions).
        
        Returns:
            Dict mapping tool names to their schemas
        """
        schemas: dict[str, dict[str, Any]] = {}
        for key, metadata in self._tools.items():
            tool_schema: dict[str, Any] = {
                "name": metadata.name,
                "category": metadata.category,
                "description": metadata.description,
            }
            if metadata.input_schema:
                tool_schema["input_schema"] = metadata.input_schema.model_json_schema()
            schemas[key] = tool_schema
        return schemas

    def get_categories(self) -> list[str]:
        """Get list of available categories."""
        return self._categories.copy()

    def auto_discover(self) -> None:
        """
        Automatically discover and import all tool modules.

        This method walks through all category folders and imports
        Python modules that contain registered tools. Modules are only
        imported if they use the @registry.register decorator, which
        prevents accidental registration of helper files.
        """
        package_dir = Path(__file__).parent

        for category in self._categories:
            category_path = package_dir / category
            if not category_path.exists():
                continue

            # Import all non-private modules in the category
            for module_info in pkgutil.iter_modules([str(category_path)]):
                # Skip private modules (starting with _)
                if module_info.name.startswith("_"):
                    continue

                module_name = f"stats_compass_core.{category}.{module_info.name}"
                try:
                    importlib.import_module(module_name)
                except Exception as e:
                    # Log warning but continue with other modules
                    print(f"Warning: Failed to import {module_name}: {e}")
        
        # Also import parent tools module to trigger registration
        try:
            importlib.import_module("stats_compass_core.parent.tools")
        except Exception as e:
            print(f"Warning: Failed to import parent tools: {e}")
        
        # Also import workflow tools module to trigger registration
        try:
            importlib.import_module("stats_compass_core.workflows")
        except Exception as e:
            print(f"Warning: Failed to import workflow tools: {e}")


# Global registry instance
registry = ToolRegistry()
