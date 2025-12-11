"""
Sample datasets included with stats-compass-core.

These datasets are provided for testing and demonstration purposes.
"""

from pathlib import Path
from typing import Literal

import pandas as pd

# Path to datasets directory (inside the package)
_DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def list_datasets() -> list[str]:
    """
    List available sample datasets.
    
    Returns:
        List of dataset names (without .csv extension)
    
    Example:
        >>> from stats_compass_core.data.datasets import list_datasets
        >>> list_datasets()
        ['Bukayo_Saka_7322', 'Housing', 'TATASTEEL']
    """
    if not _DATASETS_DIR.exists():
        return []
    return [f.stem for f in _DATASETS_DIR.glob("*.csv")]


def load_dataset(
    name: Literal["Bukayo_Saka_7322", "Housing", "TATASTEEL"] | str,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load a sample dataset by name.
    
    Args:
        name: Name of the dataset (without .csv extension)
        parse_dates: Columns to parse as dates (e.g., ['Date'])
    
    Returns:
        pandas DataFrame with the dataset
    
    Raises:
        FileNotFoundError: If the dataset doesn't exist
    
    Available datasets:
        - Bukayo_Saka_7322: Fantasy Premier League player stats
        - Housing: Housing prices dataset
        - TATASTEEL: TATA Steel historical stock prices (2000-2021)
    
    Example:
        >>> from stats_compass_core.data.datasets import load_dataset
        >>> df = load_dataset('TATASTEEL', parse_dates=['Date'])
        >>> df.head()
    """
    csv_path = _DATASETS_DIR / f"{name}.csv"

    if not csv_path.exists():
        available = list_datasets()
        raise FileNotFoundError(
            f"Dataset '{name}' not found. "
            f"Available datasets: {available}"
        )

    return pd.read_csv(csv_path, parse_dates=parse_dates)


def get_dataset_path(name: str) -> Path:
    """
    Get the file path to a dataset.
    
    Args:
        name: Name of the dataset (without .csv extension)
    
    Returns:
        Path to the CSV file
    
    Raises:
        FileNotFoundError: If the dataset doesn't exist
    """
    csv_path = _DATASETS_DIR / f"{name}.csv"

    if not csv_path.exists():
        available = list_datasets()
        raise FileNotFoundError(
            f"Dataset '{name}' not found. "
            f"Available datasets: {available}"
        )

    return csv_path
