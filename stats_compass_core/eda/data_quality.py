"""
Data quality analysis tools.

Provides comprehensive analysis of data quality including:
- Missing data analysis with patterns and correlations
- Outlier detection using multiple statistical methods
- Quality score calculation and recommendations
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import DataQualityResult
from stats_compass_core.state import DataFrameState


class AnalyzeMissingDataInput(StrictToolInput):
    """Input schema for missing data analysis tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to analyze. Uses active if not specified.",
    )
    correlation_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Threshold for identifying correlated missing patterns",
    )


class DetectOutliersInput(StrictToolInput):
    """Input schema for outlier detection tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to analyze. Uses active if not specified.",
    )
    method: Literal["iqr", "zscore", "modified_zscore"] = Field(
        default="iqr",
        description=(
            "Method for outlier detection: "
            "'iqr' - Interquartile range method, "
            "'zscore' - Standard z-score (>3), "
            "'modified_zscore' - Modified z-score using MAD"
        ),
    )
    threshold: float | None = Field(
        default=None,
        description="Custom threshold. Default: 1.5 for IQR, 3.0 for z-score methods",
    )


class DataQualityReportInput(StrictToolInput):
    """Input schema for comprehensive data quality report."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to analyze. Uses active if not specified.",
    )
    include_outliers: bool = Field(
        default=True,
        description="Whether to include outlier analysis",
    )
    outlier_method: Literal["iqr", "zscore", "modified_zscore"] = Field(
        default="iqr",
        description="Method for outlier detection",
    )


def _analyze_missing_patterns(df: pd.DataFrame, threshold: float) -> list[dict]:
    """Find columns with correlated missing patterns."""
    # Create missing indicator matrix
    missing_matrix = df.isnull().astype(int)

    # Find columns with missing data
    cols_with_missing = missing_matrix.columns[missing_matrix.sum() > 0].tolist()

    if len(cols_with_missing) < 2:
        return []

    correlated_patterns = []

    for i, col1 in enumerate(cols_with_missing):
        for col2 in cols_with_missing[i+1:]:
            # Calculate correlation between missing indicators
            corr = missing_matrix[col1].corr(missing_matrix[col2])
            if not np.isnan(corr) and abs(corr) >= threshold:
                correlated_patterns.append({
                    "column1": col1,
                    "column2": col2,
                    "correlation": round(float(corr), 3),
                })

    return correlated_patterns


def _detect_outliers_iqr(series: pd.Series, threshold: float = 1.5) -> dict:
    """Detect outliers using IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    outliers = series[(series < lower_bound) | (series > upper_bound)]

    return {
        "method": "iqr",
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "outlier_count": len(outliers),
        "outlier_percentage": round(len(outliers) / len(series) * 100, 2) if len(series) > 0 else 0,
        "outlier_indices": outliers.index.tolist()[:20],  # Limit to first 20
    }


def _detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> dict:
    """Detect outliers using z-score method."""
    mean = series.mean()
    std = series.std()

    if std == 0:
        return {
            "method": "zscore",
            "mean": float(mean),
            "std": 0,
            "threshold": threshold,
            "outlier_count": 0,
            "outlier_percentage": 0,
            "outlier_indices": [],
        }

    z_scores = (series - mean) / std
    outliers = series[abs(z_scores) > threshold]

    return {
        "method": "zscore",
        "mean": float(mean),
        "std": float(std),
        "threshold": threshold,
        "outlier_count": len(outliers),
        "outlier_percentage": round(len(outliers) / len(series) * 100, 2) if len(series) > 0 else 0,
        "outlier_indices": outliers.index.tolist()[:20],
    }


def _detect_outliers_modified_zscore(series: pd.Series, threshold: float = 3.5) -> dict:
    """Detect outliers using modified z-score (MAD-based) method."""
    median = series.median()
    mad = np.median(np.abs(series - median))

    if mad == 0:
        return {
            "method": "modified_zscore",
            "median": float(median),
            "mad": 0,
            "threshold": threshold,
            "outlier_count": 0,
            "outlier_percentage": 0,
            "outlier_indices": [],
        }

    modified_z_scores = 0.6745 * (series - median) / mad
    outliers = series[abs(modified_z_scores) > threshold]

    return {
        "method": "modified_zscore",
        "median": float(median),
        "mad": float(mad),
        "threshold": threshold,
        "outlier_count": len(outliers),
        "outlier_percentage": round(len(outliers) / len(series) * 100, 2) if len(series) > 0 else 0,
        "outlier_indices": outliers.index.tolist()[:20],
    }


def _calculate_quality_score(
    missing_pct: float,
    outlier_pct: float,
    duplicate_pct: float,
) -> float:
    """
    Calculate overall data quality score (0-100).
    
    Scoring:
    - Missing data: -2 points per percentage point
    - Outliers: -1 point per percentage point
    - Duplicates: -1 point per percentage point
    """
    score = 100
    score -= min(missing_pct * 2, 40)  # Cap at 40 point penalty
    score -= min(outlier_pct * 1, 30)  # Cap at 30 point penalty
    score -= min(duplicate_pct * 1, 20)  # Cap at 20 point penalty
    return max(0, round(score, 1))


def _generate_recommendations(
    missing_summary: dict,
    outlier_summary: dict | None,
    duplicate_count: int,
    total_rows: int,
) -> list[str]:
    """Generate data quality improvement recommendations."""
    recommendations = []

    # Missing data recommendations
    empty_cols = missing_summary.get("completely_empty_columns", [])
    if empty_cols:
        recommendations.append(
            f"Remove completely empty columns: {', '.join(empty_cols[:5])}"
            + (f" (+{len(empty_cols)-5} more)" if len(empty_cols) > 5 else "")
        )

    mostly_missing = missing_summary.get("mostly_missing_columns", [])
    if mostly_missing:
        recommendations.append(
            f"Consider removing high-missing columns (>80%): {', '.join(mostly_missing[:5])}"
            + (f" (+{len(mostly_missing)-5} more)" if len(mostly_missing) > 5 else "")
        )

    partial_missing = [col for col, pct in missing_summary.get("missing_by_column", {}).items()
                       if 0 < pct < 80 and col not in mostly_missing]
    if partial_missing:
        recommendations.append(
            f"Consider imputation for partially missing columns: {', '.join(partial_missing[:5])}"
            + (f" (+{len(partial_missing)-5} more)" if len(partial_missing) > 5 else "")
        )

    # Outlier recommendations
    if outlier_summary:
        high_outlier_cols = [
            col for col, info in outlier_summary.get("by_column", {}).items()
            if info.get("outlier_percentage", 0) > 5
        ]
        if high_outlier_cols:
            recommendations.append(
                f"Investigate columns with >5% outliers: {', '.join(high_outlier_cols[:5])}"
            )

    # Duplicate recommendations
    if duplicate_count > 0:
        dup_pct = duplicate_count / total_rows * 100
        recommendations.append(
            f"Remove {duplicate_count} duplicate rows ({dup_pct:.1f}% of data)"
        )

    if not recommendations:
        recommendations.append("Data quality looks good! No major issues detected.")

    return recommendations


@registry.register(
    category="eda",
    input_schema=AnalyzeMissingDataInput,
    description="Comprehensive missing data analysis with patterns and correlations",
)
def analyze_missing_data(
    state: DataFrameState, params: AnalyzeMissingDataInput
) -> DataQualityResult:
    """
    Perform comprehensive missing data analysis.
    
    Analyzes missing data patterns, identifies correlations between missing values
    across columns, and provides recommendations.

    Args:
        state: DataFrameState containing the DataFrame to analyze
        params: Analysis parameters

    Returns:
        DataQualityResult with missing data summary and recommendations
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    total_rows = len(df)
    total_cols = len(df.columns)

    # Calculate missing data statistics
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / total_rows * 100).round(2) if total_rows > 0 else missing_counts * 0

    # Categorize columns by missing percentage
    columns_with_missing = missing_counts[missing_counts > 0].to_dict()
    completely_empty = [col for col, count in columns_with_missing.items() if count == total_rows]
    mostly_missing = [col for col, pct in missing_pcts.items() if 80 <= pct < 100]

    # Find correlated missing patterns
    correlated_patterns = _analyze_missing_patterns(df, params.correlation_threshold)

    # Calculate average missing per row
    avg_missing_per_row = float(df.isnull().sum(axis=1).mean())

    missing_summary = {
        "total_missing_values": int(missing_counts.sum()),
        "columns_with_missing_count": len(columns_with_missing),
        "missing_by_column": {col: float(pct) for col, pct in missing_pcts.items() if pct > 0},
        "completely_empty_columns": completely_empty,
        "mostly_missing_columns": mostly_missing,
        "avg_missing_per_row": round(avg_missing_per_row, 2),
        "correlated_missing_patterns": correlated_patterns,
    }

    # Generate recommendations
    recommendations = _generate_recommendations(missing_summary, None, 0, total_rows)

    # Calculate quality score (based on missing data only)
    overall_missing_pct = missing_counts.sum() / (total_rows * total_cols) * 100 if total_rows * total_cols > 0 else 0
    quality_score = _calculate_quality_score(overall_missing_pct, 0, 0)

    return DataQualityResult(
        dataframe_name=source_name,
        total_rows=total_rows,
        total_columns=total_cols,
        missing_summary=missing_summary,
        outlier_summary=None,
        recommendations=recommendations,
        quality_score=quality_score,
    )


@registry.register(
    category="eda",
    input_schema=DetectOutliersInput,
    description="Detect outliers in numeric columns using statistical methods",
)
def detect_outliers(
    state: DataFrameState, params: DetectOutliersInput
) -> DataQualityResult:
    """
    Detect outliers in numeric columns using the specified statistical method.
    
    Methods:
    - IQR: Values beyond 1.5 * IQR from Q1/Q3
    - Z-score: Values with |z| > 3
    - Modified Z-score: Uses median and MAD, more robust to outliers

    Args:
        state: DataFrameState containing the DataFrame to analyze
        params: Detection parameters

    Returns:
        DataQualityResult with outlier analysis summary
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    total_rows = len(df)
    total_cols = len(df.columns)

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return DataQualityResult(
            dataframe_name=source_name,
            total_rows=total_rows,
            total_columns=total_cols,
            missing_summary={"message": "Missing data not analyzed"},
            outlier_summary={"message": "No numeric columns found for outlier detection"},
            recommendations=["No numeric columns available for outlier analysis"],
            quality_score=None,
        )

    # Set default threshold based on method
    if params.threshold is None:
        if params.method == "iqr":
            threshold = 1.5
        elif params.method == "zscore":
            threshold = 3.0
        else:  # modified_zscore
            threshold = 3.5
    else:
        threshold = params.threshold

    # Detect outliers in each numeric column
    outlier_results = {}
    total_outliers = 0

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        if params.method == "iqr":
            result = _detect_outliers_iqr(series, threshold)
        elif params.method == "zscore":
            result = _detect_outliers_zscore(series, threshold)
        else:
            result = _detect_outliers_modified_zscore(series, threshold)

        outlier_results[col] = result
        total_outliers += result["outlier_count"]

    # Find columns with significant outliers
    columns_with_outliers = [
        col for col, info in outlier_results.items()
        if info["outlier_count"] > 0
    ]

    outlier_summary = {
        "method": params.method,
        "threshold": threshold,
        "total_outliers_found": total_outliers,
        "columns_with_outliers": columns_with_outliers,
        "numeric_columns_analyzed": numeric_cols,
        "by_column": outlier_results,
    }

    # Generate recommendations
    recommendations = []
    high_outlier_cols = [
        col for col, info in outlier_results.items()
        if info.get("outlier_percentage", 0) > 5
    ]
    if high_outlier_cols:
        recommendations.append(
            f"Columns with >5% outliers may need attention: {', '.join(high_outlier_cols)}"
        )
    if total_outliers > 0:
        recommendations.append(
            "Use handle_outliers tool to cap, remove, or transform outlier values"
        )
    if not recommendations:
        recommendations.append("No significant outliers detected")

    return DataQualityResult(
        dataframe_name=source_name,
        total_rows=total_rows,
        total_columns=total_cols,
        missing_summary={"message": "Missing data not analyzed in this call"},
        outlier_summary=outlier_summary,
        recommendations=recommendations,
        quality_score=None,
    )


@registry.register(
    category="eda",
    input_schema=DataQualityReportInput,
    description="Comprehensive data quality report with missing data, outliers, and recommendations",
)
def data_quality_report(
    state: DataFrameState, params: DataQualityReportInput
) -> DataQualityResult:
    """
    Generate comprehensive data quality report.
    
    Combines missing data analysis, outlier detection, duplicate detection,
    and provides an overall quality score with recommendations.

    Args:
        state: DataFrameState containing the DataFrame to analyze
        params: Report parameters

    Returns:
        DataQualityResult with complete quality analysis
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    total_rows = len(df)
    total_cols = len(df.columns)

    # === Missing Data Analysis ===
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / total_rows * 100).round(2) if total_rows > 0 else missing_counts

    columns_with_missing = missing_counts[missing_counts > 0].to_dict()
    completely_empty = [col for col, count in columns_with_missing.items() if count == total_rows]
    mostly_missing = [col for col, pct in missing_pcts.items() if 80 <= pct < 100]

    correlated_patterns = _analyze_missing_patterns(df, 0.5)
    avg_missing_per_row = float(df.isnull().sum(axis=1).mean()) if total_rows > 0 else 0

    missing_summary = {
        "total_missing_values": int(missing_counts.sum()),
        "columns_with_missing_count": len(columns_with_missing),
        "missing_by_column": {col: float(pct) for col, pct in missing_pcts.items() if pct > 0},
        "completely_empty_columns": completely_empty,
        "mostly_missing_columns": mostly_missing,
        "avg_missing_per_row": round(avg_missing_per_row, 2),
        "correlated_missing_patterns": correlated_patterns,
    }

    # === Outlier Analysis ===
    outlier_summary = None
    total_outliers = 0
    overall_outlier_pct = 0

    if params.include_outliers:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            threshold = 1.5 if params.outlier_method == "iqr" else (3.0 if params.outlier_method == "zscore" else 3.5)
            outlier_results = {}

            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) == 0:
                    continue

                if params.outlier_method == "iqr":
                    result = _detect_outliers_iqr(series, threshold)
                elif params.outlier_method == "zscore":
                    result = _detect_outliers_zscore(series, threshold)
                else:
                    result = _detect_outliers_modified_zscore(series, threshold)

                outlier_results[col] = result
                total_outliers += result["outlier_count"]

            columns_with_outliers = [col for col, info in outlier_results.items() if info["outlier_count"] > 0]

            # Calculate overall outlier percentage (across all numeric values)
            total_numeric_values = sum(len(df[col].dropna()) for col in numeric_cols)
            overall_outlier_pct = total_outliers / total_numeric_values * 100 if total_numeric_values > 0 else 0

            outlier_summary = {
                "method": params.outlier_method,
                "threshold": threshold,
                "total_outliers_found": total_outliers,
                "columns_with_outliers": columns_with_outliers,
                "numeric_columns_analyzed": numeric_cols,
                "by_column": outlier_results,
            }

    # === Duplicate Detection ===
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = duplicate_count / total_rows * 100 if total_rows > 0 else 0

    # === Calculate Quality Score ===
    overall_missing_pct = missing_counts.sum() / (total_rows * total_cols) * 100 if total_rows * total_cols > 0 else 0
    quality_score = _calculate_quality_score(overall_missing_pct, overall_outlier_pct, duplicate_pct)

    # === Generate Recommendations ===
    recommendations = _generate_recommendations(missing_summary, outlier_summary, duplicate_count, total_rows)

    return DataQualityResult(
        dataframe_name=source_name,
        total_rows=total_rows,
        total_columns=total_cols,
        missing_summary=missing_summary,
        outlier_summary=outlier_summary,
        recommendations=recommendations,
        quality_score=quality_score,
    )
