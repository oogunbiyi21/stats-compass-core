"""Data transformation tools."""

from stats_compass_core.transforms.bin_rare_categories import bin_rare_categories
from stats_compass_core.transforms.filter_dataframe import filter_dataframe
from stats_compass_core.transforms.groupby_aggregate import groupby_aggregate
from stats_compass_core.transforms.pivot import pivot

# Conditional import for mean_target_encoding (requires sklearn)
try:
    from stats_compass_core.transforms.mean_target_encoding import mean_target_encoding
except ImportError:
    mean_target_encoding = None  # type: ignore

__all__ = [
    "groupby_aggregate",
    "pivot",
    "filter_dataframe",
    "bin_rare_categories",
    "mean_target_encoding",
]
