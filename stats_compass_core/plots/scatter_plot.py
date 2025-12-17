"""
Tool for creating scatter plots for two numeric columns.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Literal

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import ChartResult
from stats_compass_core.state import DataFrameState


class ScatterPlotInput(StrictToolInput):
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
    figsize: list[float] = Field(
        default_factory=lambda: [10.0, 6.0],
        min_length=2,
        max_length=2,
        description="Figure size as [width, height]"
    )
    alpha: float = Field(default=0.8, ge=0, le=1, description="Point opacity")
    save_path: str | None = Field(
        default=None, description="Path to save the plot image (e.g., 'plot.png')"
    )
    format: Literal["png", "json"] = Field(
        default="png", description="Output format: 'png' for image, 'json' for raw data"
    )


@registry.register(
    category="plots",
    input_schema=ScatterPlotInput,
    description="Create a scatter plot. Use format='json' to get raw data for interactive visualizations.",
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

    chart_title = params.title or f"{params.y} vs {params.x}"

    # Handle JSON format
    if params.format == "json":
        # Limit data points for JSON to avoid huge payloads
        MAX_POINTS = 5000
        plot_df = df[[params.x, params.y]].copy()
        if params.hue:
            plot_df[params.hue] = df[params.hue]
            
        if len(plot_df) > MAX_POINTS:
            plot_df = plot_df.sample(MAX_POINTS)
            
        chart_data = {
            "type": "scatter",
            "title": chart_title,
            "xlabel": params.x,
            "ylabel": params.y,
            "data": plot_df.to_dict(orient="records"),
            "hue": params.hue
        }
        
        return ChartResult(
            image_base64=None,
            image_format="json",
            title=chart_title,
            chart_type="scatter_plot",
            dataframe_name=source_name,
            data=chart_data,
            metadata={
                "x": params.x,
                "y": params.y,
                "hue": params.hue,
                "alpha": params.alpha,
                "data_points": len(plot_df),
                "total_points": len(df)
            }
        )

    figsize = tuple(params.figsize)
    fig, ax = plt.subplots(figsize=figsize)

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
