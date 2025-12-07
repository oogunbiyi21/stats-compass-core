"""Tests for the tool registry."""

import pandas as pd
from pydantic import BaseModel, Field

from stats_compass_core.registry import ToolMetadata, ToolRegistry


class TestToolRegistry:
    """Test suite for ToolRegistry."""

    def test_registry_initialization(self) -> None:
        """Test that registry initializes correctly."""
        registry = ToolRegistry()

        assert isinstance(registry._tools, dict)
        assert len(registry._tools) == 0
        assert len(registry.get_categories()) == 5

    def test_get_categories(self) -> None:
        """Test getting available categories."""
        registry = ToolRegistry()
        categories = registry.get_categories()

        assert "cleaning" in categories
        assert "transforms" in categories
        assert "eda" in categories
        assert "ml" in categories
        assert "plots" in categories

    def test_register_tool(self) -> None:
        """Test registering a tool."""
        registry = ToolRegistry()

        class TestInput(BaseModel):
            value: int = Field(default=10)

        @registry.register(
            category="cleaning", input_schema=TestInput, description="Test tool"
        )
        def test_tool(df: pd.DataFrame, params: TestInput) -> pd.DataFrame:
            """Test tool function."""
            return df

        # Check tool was registered
        assert "cleaning.test_tool" in registry._tools
        metadata = registry._tools["cleaning.test_tool"]
        assert metadata.name == "test_tool"
        assert metadata.category == "cleaning"
        assert metadata.description == "Test tool"
        assert metadata.input_schema == TestInput

    def test_register_tool_custom_name(self) -> None:
        """Test registering a tool with custom name."""
        registry = ToolRegistry()

        @registry.register(category="transforms", name="custom_name")
        def original_name(df: pd.DataFrame) -> pd.DataFrame:
            """Tool function."""
            return df

        assert "transforms.custom_name" in registry._tools
        assert "transforms.original_name" not in registry._tools

    def test_get_tool(self) -> None:
        """Test getting a registered tool."""
        registry = ToolRegistry()

        @registry.register(category="eda")
        def my_tool(df: pd.DataFrame) -> pd.DataFrame:
            """My tool."""
            return df

        tool_func = registry.get_tool("eda", "my_tool")
        assert tool_func is not None
        assert callable(tool_func)
        assert tool_func.__name__ == "my_tool"

    def test_get_nonexistent_tool(self) -> None:
        """Test getting a tool that doesn't exist."""
        registry = ToolRegistry()

        tool_func = registry.get_tool("cleaning", "nonexistent")
        assert tool_func is None

    def test_list_all_tools(self) -> None:
        """Test listing all tools."""
        registry = ToolRegistry()

        @registry.register(category="cleaning")
        def tool1(df: pd.DataFrame) -> pd.DataFrame:
            """Tool 1."""
            return df

        @registry.register(category="transforms")
        def tool2(df: pd.DataFrame) -> pd.DataFrame:
            """Tool 2."""
            return df

        tools = registry.list_tools()
        assert len(tools) == 2
        assert all(isinstance(t, ToolMetadata) for t in tools)

    def test_list_tools_by_category(self) -> None:
        """Test listing tools filtered by category."""
        registry = ToolRegistry()

        @registry.register(category="cleaning")
        def clean_tool(df: pd.DataFrame) -> pd.DataFrame:
            """Cleaning tool."""
            return df

        @registry.register(category="transforms")
        def transform_tool(df: pd.DataFrame) -> pd.DataFrame:
            """Transform tool."""
            return df

        cleaning_tools = registry.list_tools(category="cleaning")
        assert len(cleaning_tools) == 1
        assert cleaning_tools[0].name == "clean_tool"
        assert cleaning_tools[0].category == "cleaning"

        transform_tools = registry.list_tools(category="transforms")
        assert len(transform_tools) == 1
        assert transform_tools[0].name == "transform_tool"

    def test_tool_metadata_with_docstring(self) -> None:
        """Test that tool metadata captures docstring as description."""
        registry = ToolRegistry()

        @registry.register(category="eda")
        def documented_tool(df: pd.DataFrame) -> pd.DataFrame:
            """This is a documented tool."""
            return df

        metadata = registry._tools["eda.documented_tool"]
        assert metadata.description == "This is a documented tool."

    def test_tool_metadata_custom_description(self) -> None:
        """Test that custom description overrides docstring."""
        registry = ToolRegistry()

        @registry.register(category="ml", description="Custom description")
        def tool_with_docstring(df: pd.DataFrame) -> pd.DataFrame:
            """Original docstring."""
            return df

        metadata = registry._tools["ml.tool_with_docstring"]
        assert metadata.description == "Custom description"

    def test_registered_tool_is_callable(self) -> None:
        """Test that registered tool can be called."""
        registry = ToolRegistry()

        @registry.register(category="cleaning")
        def multiply_values(df: pd.DataFrame) -> pd.DataFrame:
            """Multiply all values by 2."""
            return df * 2

        # Create test data
        df = pd.DataFrame({"A": [1, 2, 3]})

        # Call the tool directly
        result = multiply_values(df)
        assert result["A"].tolist() == [2, 4, 6]

        # Call via registry
        tool = registry.get_tool("cleaning", "multiply_values")
        assert tool is not None
        result2 = tool(df)
        assert result2["A"].tolist() == [2, 4, 6]
