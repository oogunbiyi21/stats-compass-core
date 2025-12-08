"""Exploratory data analysis tools."""

from stats_compass_core.eda.chi_square_tests import (
    chi_square_independence,
    chi_square_goodness_of_fit,
)
from stats_compass_core.eda.correlations import correlations
from stats_compass_core.eda.data_quality import (
    analyze_missing_data,
    detect_outliers,
    data_quality_report,
)
from stats_compass_core.eda.describe import describe
from stats_compass_core.eda.hypothesis_tests import t_test, z_test

__all__ = [
    "analyze_missing_data",
    "chi_square_independence",
    "chi_square_goodness_of_fit",
    "correlations",
    "data_quality_report",
    "describe",
    "detect_outliers",
    "t_test",
    "z_test",
]

