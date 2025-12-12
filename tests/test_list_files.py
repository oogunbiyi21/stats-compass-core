"""Tests for list_files tool."""

import pytest
from pathlib import Path
from unittest.mock import patch

from stats_compass_core.data.list_files import list_files, ListFilesInput
from stats_compass_core.state import DataFrameState


class TestListFiles:
    """Tests for list_files tool."""

    @pytest.fixture
    def temp_dir_with_files(self, tmp_path):
        """Create a temporary directory with some files."""
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.csv").touch()
        (tmp_path / ".hidden").touch()
        (tmp_path / "subdir").mkdir()
        return tmp_path

    def test_list_files_current_dir(self, temp_dir_with_files):
        """Test listing files in the current directory (mocked)."""
        state = DataFrameState()
        
        # We pass the temp dir as the directory to list
        params = ListFilesInput(directory=str(temp_dir_with_files))
        result = list_files(state, params)
        
        assert result.count == 2
        assert "file1.txt" in result.files
        assert "file2.csv" in result.files
        assert ".hidden" not in result.files
        assert "subdir" not in result.files  # It only lists files, not dirs? Let's check implementation.

    def test_list_files_not_found(self):
        """Test listing files in a non-existent directory."""
        state = DataFrameState()
        params = ListFilesInput(directory="/non/existent/path")
        
        with pytest.raises(RuntimeError, match="Directory not found"):
            list_files(state, params)

    def test_list_files_not_a_directory(self, tmp_path):
        """Test listing files on a file path."""
        state = DataFrameState()
        file_path = tmp_path / "test.txt"
        file_path.touch()
        
        params = ListFilesInput(directory=str(file_path))
        
        with pytest.raises(RuntimeError, match="Path is not a directory"):
            list_files(state, params)
