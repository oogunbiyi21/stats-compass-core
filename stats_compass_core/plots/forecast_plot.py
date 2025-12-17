"""
Tool for plotting ARIMA forecasts as a dedicated visualization.

This exists in the 'plots' category so that MCP clients (like VS Code)
properly render the output as an image, rather than embedding it in JSON data.
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


class ForecastPlotInput(StrictToolInput):
    """Input schema for forecast plot tool."""

    model_id: str = Field(description="ID of the fitted ARIMA model to plot forecast for")
    n_periods: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of periods to forecast",
    )
    history_periods: int | None = Field(
        default=None,
        ge=1,
        description="Number of historical periods to show. If None, automatically shows 3x the forecast period (so forecast is at most 25% of the plot).",
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Confidence level for intervals",
    )
    title: str | None = Field(
        default=None,
        description="Custom plot title. If None, auto-generated.",
    )
    figsize: list[float] = Field(
        default_factory=lambda: [12.0, 6.0],
        min_length=2,
        max_length=2,
        description="Figure size as [width, height] in inches",
    )
    dpi: int = Field(
        default=100,
        ge=50,
        le=300,
        description="Resolution in DPI",
    )
    save_path: str | None = Field(
        default=None,
        description="Path to save the plot image (e.g., 'plot.png')",
    )
    format: Literal["png", "json"] = Field(
        default="png",
        description="Output format: 'png' for image, 'json' for raw data",
    )


@registry.register(
    category="plots",
    input_schema=ForecastPlotInput,
    description="Plot ARIMA forecast with historical data and confidence intervals. Use format='json' to get raw data for interactive visualizations.",
)
def forecast_plot(state: DataFrameState, params: ForecastPlotInput) -> ChartResult:
    """
    Create a visualization of an ARIMA forecast.

    This tool is in the 'plots' category specifically so that MCP clients
    like VS Code render the output as an inline image.

    Args:
        state: DataFrameState containing the fitted model
        params: Parameters for forecast plot creation

    Returns:
        ChartResult containing base64-encoded PNG image or JSON data

    Raises:
        ImportError: If matplotlib is not installed
        ValueError: If the model is not found
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting tools. "
            "Install with: pip install stats-compass-core[plots]"
        ) from exc

    # Get fitted model
    model = state.get_model(params.model_id)
    if model is None:
        raise ValueError(
            f"Model '{params.model_id}' not found. "
            "Fit an ARIMA model first using ml_fit_arima."
        )

    # Generate forecast
    try:
        forecast_result = model.get_forecast(steps=params.n_periods)
        forecast_mean = forecast_result.predicted_mean

        # Get confidence intervals
        alpha = 1 - params.confidence_level
        conf_int = forecast_result.conf_int(alpha=alpha)
        lower_ci = conf_int.iloc[:, 0]
        upper_ci = conf_int.iloc[:, 1]

        # Get historical data from model
        endog = model.model.endog
        if hasattr(endog, "ndim") and endog.ndim > 1:
            endog = endog.flatten()
        full_historical = pd.Series(endog, index=model.model._index)

        # Limit historical data to show - by default, forecast should be at most 25% of plot
        # So we show 3x the forecast period of historical data
        if params.history_periods is not None:
            history_to_show = params.history_periods
        else:
            # Auto-scale: show 3x forecast period (forecast = 25% of plot)
            history_to_show = params.n_periods * 3

        # Take the last N periods of historical data
        if len(full_historical) > history_to_show:
            historical = full_historical.iloc[-history_to_show:]
        else:
            historical = full_historical

    except Exception as exc:
        raise ValueError(f"Failed to generate forecast from model: {exc}") from exc

    # Create title
    title = params.title or f"ARIMA Forecast ({params.n_periods} periods)"

    # Handle JSON format
    if params.format == "json":
        chart_data = {
            "type": "forecast",
            "title": title,
            "xlabel": "Time",
            "ylabel": "Value",
            "historical": {
                "x": [str(idx) for idx in historical.index],
                "y": historical.tolist(),
            },
            "forecast": {
                "x": [str(idx) for idx in forecast_mean.index],
                "y": forecast_mean.tolist(),
            },
            "confidence_interval": {
                "x": [str(idx) for idx in forecast_mean.index],
                "lower": lower_ci.tolist(),
                "upper": upper_ci.tolist(),
            },
            "confidence_level": params.confidence_level,
        }

        return ChartResult(
            image_base64=None,
            image_format="json",
            title=title,
            chart_type="forecast",
            dataframe_name=params.model_id,
            data=chart_data,
            metadata={
                "model_id": params.model_id,
                "n_periods": params.n_periods,
                "confidence_level": params.confidence_level,
                "historical_points": len(historical),
                "forecast_points": len(forecast_mean),
            },
        )

    # Create PNG plot
    figsize = tuple(params.figsize)
    fig, ax = plt.subplots(figsize=figsize)

    # Plot historical data
    ax.plot(
        historical.index,
        historical.values,
        label="Historical",
        color="blue",
        linewidth=1.5,
    )

    # Plot forecast
    ax.plot(
        forecast_mean.index,
        forecast_mean.values,
        label="Forecast",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    # Plot confidence interval
    ax.fill_between(
        forecast_mean.index,
        lower_ci.values,
        upper_ci.values,
        alpha=0.3,
        color="red",
        label=f"{int(params.confidence_level * 100)}% Confidence Interval",
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save to file if requested (before converting to base64)
    if params.save_path:
        save_path = os.path.expanduser(params.save_path)
        dir_name = os.path.dirname(os.path.abspath(save_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fig.savefig(save_path, format="png", dpi=params.dpi, bbox_inches="tight")

    # Convert to base64
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=params.dpi, bbox_inches="tight")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return ChartResult(
        image_base64=image_base64,
        image_format="png",
        title=title,
        chart_type="forecast",
        dataframe_name=params.model_id,
        data=None,
        metadata={
            "model_id": params.model_id,
            "n_periods": params.n_periods,
            "confidence_level": params.confidence_level,
            "historical_points": len(historical),
            "forecast_points": len(forecast_mean),
            "figsize": params.figsize,
            "dpi": params.dpi,
            "save_path": params.save_path,
        },
    )
