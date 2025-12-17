"""
Classification curve visualization tools.

Provides ROC curve and Precision-Recall curve plots for evaluating
binary classification models.
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
from stats_compass_core.results import ClassificationCurveResult
from stats_compass_core.state import DataFrameState


class ROCCurveInput(StrictToolInput):
    """Input schema for ROC curve tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    true_column: str = Field(description="Column containing true binary labels (0/1)")
    prob_column: str = Field(
        description="Column containing predicted probabilities for positive class"
    )
    model_id: str = Field(
        default="model",
        description="Identifier for the model being evaluated",
    )
    title: str | None = Field(
        default=None,
        description="Custom title for the chart. If None, auto-generated.",
    )
    figsize: list[float] = Field(
        default_factory=lambda: [8.0, 6.0],
        min_length=2,
        max_length=2,
        description="Figure size as [width, height] in inches",
    )
    dpi: int = Field(default=100, ge=50, le=300, description="Resolution in DPI")
    save_path: str | None = Field(
        default=None, description="Path to save the plot image (e.g., 'plot.png')"
    )
    format: Literal["png", "json"] = Field(
        default="png", description="Output format: 'png' for image, 'json' for raw data"
    )


class PrecisionRecallCurveInput(StrictToolInput):
    """Input schema for Precision-Recall curve tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    true_column: str = Field(description="Column containing true binary labels (0/1)")
    prob_column: str = Field(
        description="Column containing predicted probabilities for positive class"
    )
    model_id: str = Field(
        default="model",
        description="Identifier for the model being evaluated",
    )
    title: str | None = Field(
        default=None,
        description="Custom title for the chart. If None, auto-generated.",
    )
    figsize: list[float] = Field(
        default_factory=lambda: [8.0, 6.0],
        min_length=2,
        max_length=2,
        description="Figure size as [width, height] in inches",
    )
    dpi: int = Field(default=100, ge=50, le=300, description="Resolution in DPI")
    save_path: str | None = Field(
        default=None, description="Path to save the plot image (e.g., 'plot.png')"
    )
    format: Literal["png", "json"] = Field(
        default="png", description="Output format: 'png' for image, 'json' for raw data"
    )


def _interpret_auc(auc_score: float) -> str:
    """Interpret AUC score."""
    if auc_score >= 0.9:
        return "Excellent"
    elif auc_score >= 0.8:
        return "Good"
    elif auc_score >= 0.7:
        return "Fair"
    elif auc_score >= 0.6:
        return "Poor"
    else:
        return "Very Poor"


def _interpret_ap(ap_score: float, baseline: float) -> str:
    """Interpret Average Precision score."""
    if ap_score >= 0.9:
        return "Excellent"
    elif ap_score >= 0.8:
        return "Very Good"
    elif ap_score >= 0.7:
        return "Good"
    elif ap_score >= 0.6:
        return "Fair"
    elif ap_score > baseline:
        return "Better than random"
    else:
        return "Poor"


@registry.register(
    category="plots",
    input_schema=ROCCurveInput,
    description="Create ROC curve for binary classification model evaluation. Use format='json' to get raw data for interactive visualizations.",
)
def roc_curve_plot(
    state: DataFrameState, params: ROCCurveInput
) -> ClassificationCurveResult:
    """
    Create ROC (Receiver Operating Characteristic) curve.
    
    Shows the trade-off between true positive rate (sensitivity) and
    false positive rate (1 - specificity) at various classification thresholds.

    Note: Requires scikit-learn and matplotlib (install with 'ml' and 'plots' extras).

    Args:
        state: DataFrameState containing the DataFrame with predictions
        params: Parameters for ROC curve generation

    Returns:
        ClassificationCurveResult with curve data and base64-encoded image

    Raises:
        ImportError: If sklearn or matplotlib is not installed
        ValueError: If columns are missing or data is invalid
    """
    try:
        from sklearn.metrics import auc, roc_curve
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for ROC curves. "
            "Install with: pip install stats-compass-core[ml]"
        ) from e

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install stats-compass-core[plots]"
        ) from e

    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    # Validate columns
    for col in (params.true_column, params.prob_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # Get data and drop missing values
    data = df[[params.true_column, params.prob_column]].dropna()

    if len(data) < 2:
        raise ValueError(
            f"Insufficient data: need at least 2 rows, got {len(data)}"
        )

    y_true = data[params.true_column].values
    y_prob = data[params.prob_column].values

    # Validate binary labels
    unique_labels = np.unique(y_true)
    if len(unique_labels) != 2:
        raise ValueError(
            f"ROC curves require binary labels. Found {len(unique_labels)} unique values: {unique_labels}"
        )

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Generate interpretation
    interpretation = (
        f"AUC = {roc_auc:.3f} ({_interpret_auc(roc_auc)}). "
        f"The model correctly ranks a random positive instance higher than a random negative instance "
        f"{roc_auc*100:.1f}% of the time."
    )

    # Handle JSON format
    if params.format == "json":
        return ClassificationCurveResult(
            curve_type="roc",
            x_values=[float(x) for x in fpr],
            y_values=[float(y) for y in tpr],
            thresholds=[float(t) for t in thresholds] if len(thresholds) == len(fpr) else None,
            auc_score=float(roc_auc),
            image_base64=None,
            model_id=params.model_id,
            dataframe_name=source_name,
            target_column=params.true_column,
            interpretation=interpretation,
        )

    # Create plot
    figsize = tuple(params.figsize)
    fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random (AUC = 0.5)")

    # Formatting
    title = params.title or f"ROC Curve: {params.true_column}"
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if params.save_path:
        save_path = os.path.expanduser(params.save_path)
        fig.savefig(save_path, dpi=params.dpi, bbox_inches='tight')

    # Convert to base64
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=params.dpi, bbox_inches="tight")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)

    # Generate interpretation
    interpretation = (
        f"AUC = {roc_auc:.3f} ({_interpret_auc(roc_auc)}). "
        f"The model correctly ranks a random positive instance higher than a random negative instance "
        f"{roc_auc*100:.1f}% of the time."
    )

    return ClassificationCurveResult(
        curve_type="roc",
        x_values=[float(x) for x in fpr],
        y_values=[float(y) for y in tpr],
        thresholds=[float(t) for t in thresholds] if len(thresholds) == len(fpr) else None,
        auc_score=float(roc_auc),
        image_base64=image_base64,
        model_id=params.model_id,
        dataframe_name=source_name,
        target_column=params.true_column,
        interpretation=interpretation,
    )


@registry.register(
    category="plots",
    input_schema=PrecisionRecallCurveInput,
    description="Create Precision-Recall curve for binary classification, especially useful for imbalanced datasets. Use format='json' to get raw data for interactive visualizations.",
)
def precision_recall_curve_plot(
    state: DataFrameState, params: PrecisionRecallCurveInput
) -> ClassificationCurveResult:
    """
    Create Precision-Recall curve.
    
    Shows the trade-off between precision and recall at various classification
    thresholds. Particularly useful for imbalanced datasets where the positive
    class is rare.

    Note: Requires scikit-learn and matplotlib (install with 'ml' and 'plots' extras).

    Args:
        state: DataFrameState containing the DataFrame with predictions
        params: Parameters for PR curve generation

    Returns:
        ClassificationCurveResult with curve data and base64-encoded image

    Raises:
        ImportError: If sklearn or matplotlib is not installed
        ValueError: If columns are missing or data is invalid
    """
    try:
        from sklearn.metrics import average_precision_score, precision_recall_curve
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for PR curves. "
            "Install with: pip install stats-compass-core[ml]"
        ) from e

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install stats-compass-core[plots]"
        ) from e

    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    # Validate columns
    for col in (params.true_column, params.prob_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # Get data and drop missing values
    data = df[[params.true_column, params.prob_column]].dropna()

    if len(data) < 2:
        raise ValueError(
            f"Insufficient data: need at least 2 rows, got {len(data)}"
        )

    y_true = data[params.true_column].values
    y_prob = data[params.prob_column].values

    # Validate binary labels
    unique_labels = np.unique(y_true)
    if len(unique_labels) != 2:
        raise ValueError(
            f"PR curves require binary labels. Found {len(unique_labels)} unique values: {unique_labels}"
        )

    # Calculate PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)

    # Baseline (random classifier for imbalanced data)
    positive_ratio = float(np.mean(y_true))

    # Generate interpretation
    interpretation = (
        f"Average Precision = {ap_score:.3f} ({_interpret_ap(ap_score, positive_ratio)}). "
        f"Baseline (random) = {positive_ratio:.3f} (positive class ratio). "
        f"PR curves are more informative than ROC for imbalanced datasets."
    )

    # Handle JSON format
    if params.format == "json":
        return ClassificationCurveResult(
            curve_type="precision_recall",
            x_values=[float(x) for x in recall],
            y_values=[float(y) for y in precision],
            thresholds=[float(t) for t in thresholds] if len(thresholds) > 0 else None,
            auc_score=float(ap_score),
            image_base64=None,
            model_id=params.model_id,
            dataframe_name=source_name,
            target_column=params.true_column,
            interpretation=interpretation,
        )

    # Create plot
    figsize = tuple(params.figsize)
    fig, ax = plt.subplots(figsize=figsize)

    # Plot PR curve
    ax.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AP = {ap_score:.3f})")

    # Plot baseline (horizontal line at positive class ratio)
    ax.axhline(y=positive_ratio, color="gray", lw=1, linestyle="--",
               label=f"Random (AP = {positive_ratio:.3f})")

    # Formatting
    title = params.title or f"Precision-Recall Curve: {params.true_column}"
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if params.save_path:
        save_path = os.path.expanduser(params.save_path)
        fig.savefig(save_path, dpi=params.dpi, bbox_inches='tight')

    # Convert to base64
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=params.dpi, bbox_inches="tight")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)

    # Generate interpretation
    interpretation = (
        f"Average Precision = {ap_score:.3f} ({_interpret_ap(ap_score, positive_ratio)}). "
        f"Baseline (random) = {positive_ratio:.3f} (positive class ratio). "
        f"PR curves are more informative than ROC for imbalanced datasets."
    )

    return ClassificationCurveResult(
        curve_type="precision_recall",
        x_values=[float(x) for x in recall],
        y_values=[float(y) for y in precision],
        thresholds=[float(t) for t in thresholds] if len(thresholds) > 0 else None,
        auc_score=float(ap_score),
        image_base64=image_base64,
        model_id=params.model_id,
        dataframe_name=source_name,
        target_column=params.true_column,
        interpretation=interpretation,
    )
