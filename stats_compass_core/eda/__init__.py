"""Exploratory data analysis tools."""

from stats_compass_core.eda.chi_square_tests import (
    chi_square_goodness_of_fit,
    chi_square_independence,
)
from stats_compass_core.eda.correlations import correlations
from stats_compass_core.eda.data_quality import (
    analyze_missing_data,
    data_quality_report,
    detect_outliers,
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

