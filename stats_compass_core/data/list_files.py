"""
Tool for listing files in a local directory.
"""

from pathlib import Path

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import FileListResult
from stats_compass_core.state import DataFrameState
from stats_compass_core.utils.file_safety import UnsafePathError, check_read_path


class ListFilesInput(StrictToolInput):
    """Input schema for list_files tool."""

    directory: str = Field(
        default=".",
        description="Directory to list files from. Defaults to current working directory. Supports '~' expansion (e.g. '~/Downloads').",
    )


@registry.register(
    category="data",
    input_schema=ListFilesInput,
    description="List files in a local directory. Useful for finding the correct path for load_csv.",
    tier="util",
)
def list_files(state: DataFrameState, params: ListFilesInput) -> FileListResult:
    """
    List files in a local directory.

    Args:
        state: The DataFrameState object; its file_policy confines the listing.
        params: Parameters for listing files.

    Returns:
        FileListResult with list of files.
    """
    try:
        # Resolve directory path (handle ~ for home directory). Under a
        # session's FilePolicy it must lie inside a read root (security scan F8).
        directory = Path(check_read_path(params.directory, state.file_policy.read_roots)).resolve()

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        # List files (ignore hidden files and directories)
        files = []
        for item in directory.iterdir():
            if item.is_file() and not item.name.startswith("."):
                files.append(item.name)

        # Sort files
        files.sort()

        return FileListResult(
            directory=str(directory),
            files=files,
            count=len(files),
            message=f"Found {len(files)} files in {directory}",
        )

    except UnsafePathError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to list files in '{params.directory}': {str(e)}")
