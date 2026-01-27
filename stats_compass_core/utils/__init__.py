"""Utility modules for stats-compass-core."""

from .spreadsheet_safety import (
    FORMULA_TRIGGER_CHARS,
    sanitize_cell,
    sanitize_dataframe,
)
from .file_safety import (
    PROTECTED_EXTENSIONS,
    SAFE_OUTPUT_EXTENSIONS,
    UnsafePathError,
    get_unique_filepath,
    is_path_safe,
    safe_save,
    safe_save_figure,
    safe_write_path,
)

__all__ = [
    # Spreadsheet safety (formula injection protection)
    "FORMULA_TRIGGER_CHARS",
    "sanitize_cell",
    "sanitize_dataframe",
    # File safety
    "UnsafePathError",
    "is_path_safe",
    "get_unique_filepath",
    "safe_write_path",
    "safe_save_figure",
    "safe_save",
    "SAFE_OUTPUT_EXTENSIONS",
    "PROTECTED_EXTENSIONS",
]
