"""
Tool for creating bar charts from categorical columns.
"""

from __future__ import annotations

import base64
from io import BytesIO

from pydantic import BaseModel, Field

from stats_compass_core.registry import registry
from stats_compass_core.results import ChartResult
from stats_compass_core.state import DataFrameState


class BarChartInput(BaseModel):
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
    figsize: tuple[float, float] = Field(
        default=(10, 6), description="Figure size as (width, height)"
    )


@registry.register(
    category="plots",
    input_schema=BarChartInput,
    description="Create a bar chart showing category counts",
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

    fig, ax = plt.subplots(figsize=params.figsize)

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
