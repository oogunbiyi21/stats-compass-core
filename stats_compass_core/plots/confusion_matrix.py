"""
Confusion matrix visualization tool.

Provides confusion matrix heatmaps for evaluating classification models
with support for binary and multi-class classification.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import ChartResult
from stats_compass_core.state import DataFrameState


class ConfusionMatrixInput(StrictToolInput):
    """Input schema for confusion matrix tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    true_column: str = Field(description="Column containing true labels")
    pred_column: str = Field(
        description="Column containing predicted labels"
    )
    normalize: Literal["true", "pred", "all", "none"] = Field(
        default="none",
        description=(
            "Normalization mode: 'true' normalizes by row (true label), "
            "'pred' by column (predicted label), 'all' by total count, "
            "'none' shows raw counts"
        ),
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
    cmap: str = Field(
        default="Blues",
        description="Matplotlib colormap name for the heatmap",
    )
    save_path: str | None = Field(
        default=None, description="Path to save the plot image (e.g., 'plot.png')"
    )
    format: Literal["png", "json"] = Field(
        default="png", description="Output format: 'png' for image, 'json' for raw data"
    )


def _calculate_metrics_from_cm(
    cm: np.ndarray, labels: list[Any]
) -> dict[str, Any]:
    """Calculate per-class and overall metrics from confusion matrix."""
    n_classes = len(labels)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[str(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[i, :].sum()),
        }
    
    # Calculate overall metrics
    total = cm.sum()
    correct = np.diag(cm).sum()
    accuracy = correct / total if total > 0 else 0.0
    
    # Macro averages
    macro_precision = np.mean([m["precision"] for m in per_class_metrics.values()])
    macro_recall = np.mean([m["recall"] for m in per_class_metrics.values()])
    macro_f1 = np.mean([m["f1"] for m in per_class_metrics.values()])
    
    # Weighted averages
    supports = np.array([m["support"] for m in per_class_metrics.values()])
    total_support = supports.sum()
    if total_support > 0:
        weighted_precision = sum(
            m["precision"] * m["support"] for m in per_class_metrics.values()
        ) / total_support
        weighted_recall = sum(
            m["recall"] * m["support"] for m in per_class_metrics.values()
        ) / total_support
        weighted_f1 = sum(
            m["f1"] * m["support"] for m in per_class_metrics.values()
        ) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0
    
    return {
        "per_class": per_class_metrics,
        "overall": {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1),
            "total_samples": int(total),
            "n_classes": n_classes,
        }
    }


def _interpret_accuracy(accuracy: float, n_classes: int) -> str:
    """Interpret accuracy based on number of classes."""
    # Random baseline for multi-class
    baseline = 1.0 / n_classes
    
    if accuracy >= 0.95:
        return "Excellent"
    elif accuracy >= 0.90:
        return "Very Good"
    elif accuracy >= 0.80:
        return "Good"
    elif accuracy >= 0.70:
        return "Fair"
    elif accuracy > baseline * 1.5:
        return "Better than random"
    else:
        return "Poor"


@registry.register(
    category="plots",
    input_schema=ConfusionMatrixInput,
    description=(
        "Create a confusion matrix heatmap for classification model evaluation. "
        "Supports binary and multi-class classification. "
        "Use format='json' to get raw data for interactive visualizations."
    ),
)
def confusion_matrix_plot(
    state: DataFrameState, params: ConfusionMatrixInput
) -> ChartResult:
    """
    Create a confusion matrix heatmap.
    
    Visualizes the performance of a classification model by showing
    the counts (or normalized proportions) of true vs predicted labels.
    Supports both binary and multi-class classification.

    Note: Requires scikit-learn and matplotlib (install with 'ml' and 'plots' extras).

    Args:
        state: DataFrameState containing the DataFrame with predictions
        params: Parameters for confusion matrix generation

    Returns:
        PlotResult with confusion matrix data and base64-encoded image

    Raises:
        ImportError: If sklearn or matplotlib is not installed
        ValueError: If columns are missing or data is invalid
    """
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for confusion matrix. "
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
    for col in (params.true_column, params.pred_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # Get data and drop missing values
    data = df[[params.true_column, params.pred_column]].dropna()

    if len(data) < 2:
        raise ValueError(
            f"Insufficient data: need at least 2 rows, got {len(data)}"
        )

    y_true = data[params.true_column].values
    y_pred = data[params.pred_column].values

    # Get unique labels (sorted for consistency)
    labels = sorted(list(set(y_true) | set(y_pred)))
    n_classes = len(labels)

    if n_classes < 2:
        raise ValueError(
            f"Confusion matrix requires at least 2 classes. Found {n_classes} unique values."
        )

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalize if requested
    normalize_mode = params.normalize if params.normalize != "none" else None
    if normalize_mode:
        if normalize_mode == "true":
            cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        elif normalize_mode == "pred":
            cm_normalized = cm.astype('float') / cm.sum(axis=0, keepdims=True)
        elif normalize_mode == "all":
            cm_normalized = cm.astype('float') / cm.sum()
        else:
            cm_normalized = cm.astype('float')
        # Handle division by zero
        cm_normalized = np.nan_to_num(cm_normalized)
        display_cm = cm_normalized
    else:
        display_cm = cm

    # Calculate metrics
    metrics = _calculate_metrics_from_cm(cm, labels)
    accuracy = metrics["overall"]["accuracy"]

    # Generate interpretation
    interpretation = (
        f"Accuracy = {accuracy:.1%} ({_interpret_accuracy(accuracy, n_classes)}) "
        f"on {n_classes} classes with {metrics['overall']['total_samples']} samples. "
        f"Macro F1 = {metrics['overall']['macro_f1']:.3f}, "
        f"Weighted F1 = {metrics['overall']['weighted_f1']:.3f}."
    )

    # Build title
    title = params.title
    if not title:
        normalize_label = {
            "true": " (Normalized by True Label)",
            "pred": " (Normalized by Prediction)",
            "all": " (Normalized)",
            "none": "",
        }.get(params.normalize, "")
        title = f"Confusion Matrix{normalize_label}\nAccuracy: {accuracy:.1%}"

    # JSON format - return raw data
    if params.format == "json":
        return ChartResult(
            chart_type="confusion_matrix",
            title=title,
            dataframe_name=source_name,
            data={
                "confusion_matrix": cm.tolist(),
                "labels": [str(l) for l in labels],
                "normalized_matrix": display_cm.tolist() if normalize_mode else None,
                "normalization": params.normalize,
                "metrics": metrics,
            },
            metadata={
                "interpretation": interpretation,
                "n_classes": n_classes,
                "total_samples": metrics["overall"]["total_samples"],
            },
            image_base64=None,
        )

    # Create the plot
    fig, ax = plt.subplots(figsize=tuple(params.figsize), dpi=params.dpi)

    # Create heatmap
    im = ax.imshow(display_cm, interpolation='nearest', cmap=params.cmap)
    ax.figure.colorbar(im, ax=ax)

    # Set tick labels
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=[str(l) for l in labels],
        yticklabels=[str(l) for l in labels],
        ylabel='True label',
        xlabel='Predicted label',
        title=title,
    )

    # Rotate tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = '.2f' if normalize_mode else 'd'
    thresh = display_cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, format(display_cm[i, j], fmt),
                ha="center", va="center",
                color="white" if display_cm[i, j] > thresh else "black"
            )

    fig.tight_layout()

    # Save to bytes
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    # Save to file if requested
    if params.save_path:
        save_path = os.path.expanduser(params.save_path)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(base64.b64decode(image_base64))

    return ChartResult(
        chart_type="confusion_matrix",
        title=title,
        dataframe_name=source_name,
        data={
            "confusion_matrix": cm.tolist(),
            "labels": [str(l) for l in labels],
            "normalized_matrix": display_cm.tolist() if normalize_mode else None,
            "normalization": params.normalize,
            "metrics": metrics,
        },
        metadata={
            "interpretation": interpretation,
            "n_classes": n_classes,
            "total_samples": metrics["overall"]["total_samples"],
        },
        image_base64=image_base64,
    )
