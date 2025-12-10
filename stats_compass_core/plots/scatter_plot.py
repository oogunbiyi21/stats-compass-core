"""
Tool for creating scatter plots for two numeric columns.
"""

from __future__ import annotations

import base64
from io import BytesIO

import pandas as pd
from pydantic import BaseModel, Field

from stats_compass_core.registry import registry
from stats_compass_core.results import ChartResult
from stats_compass_core.state import DataFrameState


class ScatterPlotInput(BaseModel):
    """Input schema for scatter_plot tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    x: str = Field(description="Column for x-axis")
    y: str = Field(description="Column for y-axis")
    hue: str | None = Field(
        default=None,
        description="Optional categorical column for coloring points",
    )
    title: str | None = Field(default=None, description="Optional plot title")
    figsize: tuple[float, float] = Field(
        default=(10, 6), description="Figure size as (width, height)"
    )
    alpha: float = Field(default=0.8, ge=0, le=1, description="Point opacity")


@registry.register(
    category="plots",
    input_schema=ScatterPlotInput,
    description="Create a scatter plot for two numeric columns",
)
def scatter_plot(state: DataFrameState, params: ScatterPlotInput) -> ChartResult:
    """
    Create a scatter plot for two numeric columns, optionally colored by a categorical hue.

    Note: Requires matplotlib installed (plots extra).

    Args:
        state: DataFrameState containing the DataFrame to visualize
        params: Plot parameters

    Returns:
        ChartResult containing the base64-encoded PNG image

    Raises:
        ImportError: If matplotlib is not installed
        ValueError: If required columns are missing or non-numeric
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

    for col in (params.x, params.y):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric")

    if params.hue and params.hue not in df.columns:
        raise ValueError(f"Hue column '{params.hue}' not found in DataFrame")

    fig, ax = plt.subplots(figsize=params.figsize)

    if params.hue:
        for category, group in df.groupby(params.hue):
            ax.scatter(
                group[params.x],
                group[params.y],
                alpha=params.alpha,
                label=str(category),
            )
        ax.legend(title=params.hue)
    else:
        ax.scatter(df[params.x], df[params.y], alpha=params.alpha)

    ax.set_xlabel(params.x)
    ax.set_ylabel(params.y)
    chart_title = params.title or f"{params.y} vs {params.x}"
    ax.set_title(chart_title)
    ax.grid(True, alpha=0.3)

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
        chart_type="scatter_plot",
        dataframe_name=source_name,
        metadata={
            "x": params.x,
            "y": params.y,
            "hue": params.hue,
            "alpha": params.alpha,
            "n_points": len(df),
        },
    )
