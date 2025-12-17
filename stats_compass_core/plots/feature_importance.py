"""
Tool for visualizing feature importance from trained models.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Literal

import numpy as np
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import ChartResult
from stats_compass_core.state import DataFrameState


class FeatureImportanceInput(StrictToolInput):
    """Input schema for feature_importance tool."""

    model_id: str = Field(description="ID of the trained model in state")
    top_n: int | None = Field(
        default=20, ge=1, description="Limit to top N features by absolute importance"
    )
    orientation: str = Field(
        default="horizontal",
        pattern="^(vertical|horizontal)$",
        description="Bar orientation",
    )
    title: str | None = Field(
        default=None, description="Optional plot title (defaults to 'Feature Importance')"
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


def _extract_importances(model: object) -> np.ndarray:
    """Extract feature importance or coefficients from a model."""
    if hasattr(model, "feature_importances_"):
        return np.asarray(getattr(model, "feature_importances_"), dtype=float)

    if hasattr(model, "coef_"):
        coefs = np.asarray(getattr(model, "coef_"), dtype=float)
        if coefs.ndim == 1:
            return np.abs(coefs)
        # Multiclass: average absolute coefficients across classes
        return np.mean(np.abs(coefs), axis=0)

    raise ValueError(
        "Model must expose feature_importances_ or coef_ to compute importance."
    )


@registry.register(
    category="plots",
    input_schema=FeatureImportanceInput,
    description="Visualize feature importance. Use format='json' to get raw data for interactive visualizations.",
)
def feature_importance(state: DataFrameState, params: FeatureImportanceInput) -> ChartResult:
    """
    Plot feature importance for models exposing feature_importances_ or coef_.

    Args:
        state: DataFrameState containing the trained model
        params: Parameters including the model_id

    Returns:
        ChartResult containing the base64-encoded PNG image

    Raises:
        ImportError: If matplotlib is not installed
        ValueError: If model not found or importance extraction fails
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting tools. "
            "Install with: pip install stats-compass-core[plots]"
        ) from exc

    # Get model from state
    model = state.get_model(params.model_id)
    model_info = state.get_model_info(params.model_id)

    if model is None:
        raise ValueError(f"Model '{params.model_id}' not found in state")

    importances = _extract_importances(model)
    feature_names = model_info.feature_columns

    if len(importances) != len(feature_names):
        raise ValueError(
            "Length of feature_importances_/coef_ does not match feature_names."
        )

    # Create importance DataFrame and sort
    import pandas as pd
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": np.abs(importances)}
    ).sort_values("importance", ascending=False)

    if params.top_n:
        importance_df = importance_df.head(params.top_n)

    chart_title = params.title or f"Feature Importance - {model_info.model_type}"

    # Handle JSON format
    if params.format == "json":
        chart_data = {
            "type": "bar",
            "title": chart_title,
            "xlabel": "Importance" if params.orientation == "horizontal" else "Feature",
            "ylabel": "Feature" if params.orientation == "horizontal" else "Importance",
            "categories": importance_df["feature"].tolist(),
            "values": importance_df["importance"].tolist(),
            "orientation": params.orientation
        }
        
        return ChartResult(
            image_base64=None,
            image_format="json",
            title=chart_title,
            chart_type="feature_importance",
            dataframe_name=model_info.source_dataframe,
            data=chart_data,
            metadata={
                "model_id": params.model_id,
                "model_type": model_info.model_type,
                "top_n": params.top_n,
                "orientation": params.orientation,
                "features_shown": len(importance_df),
            }
        )

    figsize = tuple(params.figsize)
    fig, ax = plt.subplots(figsize=figsize)

    if params.orientation == "horizontal":
        # Reverse for horizontal so highest is at top
        plot_df = importance_df.iloc[::-1]
        ax.barh(plot_df["feature"], plot_df["importance"], edgecolor="black")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
    else:
        ax.bar(importance_df["feature"], importance_df["importance"], edgecolor="black")
        ax.set_ylabel("Importance")
        ax.set_xlabel("Feature")
        plt.xticks(rotation=45, ha="right")

    chart_title = params.title or f"Feature Importance - {model_info.model_type}"
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
        chart_type="feature_importance",
        dataframe_name=model_info.source_dataframe,
        metadata={
            "model_id": params.model_id,
            "model_type": model_info.model_type,
            "top_n": params.top_n,
            "orientation": params.orientation,
            "features_shown": len(importance_df),
            "total_features": len(feature_names),
        },
    )
