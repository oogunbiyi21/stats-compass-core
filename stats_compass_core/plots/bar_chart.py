"""
Tool for creating bar charts from categorical columns.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Literal

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import ChartResult
from stats_compass_core.state import DataFrameState


class BarChartInput(StrictToolInput):
    """Input schema for bar_chart tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    column: str = Field(description="Categorical column to plot counts for")
    top_n: int | None = Field(
        default=10, ge=1, description="Limit to top N categories by count"
    )
    orientation: str = Field(
        default="vertical", pattern="^(vertical|horizontal)$", description="Bar orientation"
    )
    title: str | None = Field(
        default=None, description="Optional plot title (defaults to column name)"
    )
    figsize: list[float] = Field(
        default_factory=lambda: [10.0, 6.0],
        min_length=2,
        max_length=2,
        description="Figure size as [width, height]"
    )
    save_path: str | None = Field(
        default=None, description="Path to save the plot image (e.g., 'plot.png')"
    )
    format: Literal["png", "json"] = Field(
        default="png", description="Output format: 'png' for image, 'json' for raw data"
    )


@registry.register(
    category="plots",
    input_schema=BarChartInput,
    description="Create a bar chart. Use format='json' to get raw data for interactive visualizations.",
)
def bar_chart(state: DataFrameState, params: BarChartInput) -> ChartResult:
    """
    Create a bar chart of category counts.

    Note: Requires matplotlib installed (plots extra).

    Args:
        state: DataFrameState containing the DataFrame to visualize
        params: Parameters for bar chart creation

    Returns:
        ChartResult containing the base64-encoded PNG image

    Raises:
        ImportError: If matplotlib is not installed
        ValueError: If the column is missing
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting tools. "
            "Install with: pip install stats-compass-core[plots]"
        ) from exc

    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    if params.column not in df.columns:
        raise ValueError(f"Column '{params.column}' not found in DataFrame")

    counts = df[params.column].value_counts(dropna=False)
    if params.top_n:
        counts = counts.head(params.top_n)

    chart_title = params.title or f"Counts for {params.column}"

    # Handle JSON format
    if params.format == "json":
        chart_data = {
            "type": "bar",
            "title": chart_title,
            "xlabel": params.column if params.orientation == "vertical" else "Count",
            "ylabel": "Count" if params.orientation == "vertical" else params.column,
            "categories": counts.index.astype(str).tolist(),
            "values": counts.values.tolist(),
            "orientation": params.orientation
        }
        
        return ChartResult(
            image_base64=None,
            image_format="json",
            title=chart_title,
            chart_type="bar_chart",
            dataframe_name=source_name,
            data=chart_data,
            metadata={
                "column": params.column,
                "top_n": params.top_n,
                "categories_shown": len(counts),
                "orientation": params.orientation,
            }
        )

    figsize = tuple(params.figsize)
    fig, ax = plt.subplots(figsize=figsize)

    if params.orientation == "vertical":
        counts.plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_xlabel(params.column)
        ax.set_ylabel("Count")
    else:
        counts.plot(kind="barh", ax=ax, edgecolor="black")
        ax.set_ylabel(params.column)
        ax.set_xlabel("Count")

    chart_title = params.title or f"Counts for {params.column}"
    ax.set_title(chart_title)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if params.save_path:
        save_path = os.path.expanduser(params.save_path)
        fig.savefig(save_path, bbox_inches='tight')

    # Convert to base64 PNG
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    image_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return ChartResult(
        image_base64=image_b64,
        image_format="png",
        title=chart_title,
        chart_type="bar_chart",
        dataframe_name=source_name,
        metadata={
            "column": params.column,
            "top_n": params.top_n,
            "orientation": params.orientation,
            "categories_shown": len(counts),
        },
    )
