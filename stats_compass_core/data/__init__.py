"""Data loading and management tools."""

from stats_compass_core.data.add_column import add_column
from stats_compass_core.data.concat_dataframes import concat_dataframes
from stats_compass_core.data.drop_columns import drop_columns
from stats_compass_core.data.get_sample import get_sample
from stats_compass_core.data.get_schema import get_schema
from stats_compass_core.data.list_dataframes import list_dataframes
from stats_compass_core.data.list_files import list_files
from stats_compass_core.data.load_csv import load_csv
from stats_compass_core.data.load_excel import load_excel
from stats_compass_core.data.load_dataset import load_dataset
from stats_compass_core.data.merge_dataframes import merge_dataframes
from stats_compass_core.data.rename_columns import rename_columns
from stats_compass_core.data.save_csv import save_csv

__all__ = [
    "load_csv",
    "load_excel",
    "save_csv",
    "load_dataset",
    "get_schema",
    "get_sample",
    "list_dataframes",
    "list_files",
    "merge_dataframes",
    "concat_dataframes",
    "drop_columns",
    "rename_columns",
    "add_column",
    # Sample datasets
    "load_dataset",
    "list_datasets",
    "get_dataset_path",
]
