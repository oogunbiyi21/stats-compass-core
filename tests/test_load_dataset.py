"""Tests for load_dataset tool."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd

from stats_compass_core.data.load_dataset import LoadDatasetInput, load_dataset, _list_available_datasets
from stats_compass_core.state import DataFrameState


class TestLoadDataset:
    """Tests for load_dataset tool."""

    @pytest.fixture
    def mock_datasets_dir(self, tmp_path):
        """Create a temporary datasets directory with sample files."""
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir()
        
        # Create a sample dataset
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df.to_csv(datasets_dir / "TestDataset.csv", index=False)
        
        return datasets_dir

    def test_list_available_datasets(self, mock_datasets_dir):
        """Test listing available datasets."""
        with patch("stats_compass_core.data.load_dataset._DATASETS_DIR", mock_datasets_dir):
            datasets = _list_available_datasets()
            assert "TestDataset" in datasets
            assert len(datasets) == 1

    def test_load_dataset_basic(self, mock_datasets_dir):
        """Test loading a dataset."""
        state = DataFrameState()
        
        with patch("stats_compass_core.data.load_dataset._DATASETS_DIR", mock_datasets_dir):
            # We need to patch the field validation or just bypass it since pydantic validates at instantiation
            # But since we are patching the dir, the validation in the class definition (if any dynamic) might have already run.
            # However, the class definition uses _list_available_datasets() at import time.
            # So the 'Available' description in Field might be stale, but that doesn't affect logic.
            
            params = LoadDatasetInput(name="TestDataset")
            result = load_dataset(state, params)

            assert result.dataframe_name == "TestDataset"
            assert result.shape[0] == 3
            assert len(result.columns) == 2
            assert state.get_dataframe("TestDataset") is not None

    def test_load_dataset_not_found(self, mock_datasets_dir):
        """Test loading a non-existent dataset."""
        state = DataFrameState()
        
        with patch("stats_compass_core.data.load_dataset._DATASETS_DIR", mock_datasets_dir):
            params = LoadDatasetInput(name="NonExistent")
            
            with pytest.raises(ValueError, match="Dataset 'NonExistent' not found"):
                load_dataset(state, params)

    def test_load_dataset_set_active(self, mock_datasets_dir):
        """Test setting the loaded dataset as active."""
        state = DataFrameState()
        
        with patch("stats_compass_core.data.load_dataset._DATASETS_DIR", mock_datasets_dir):
            params = LoadDatasetInput(name="TestDataset", set_active=True)
            load_dataset(state, params)
            
            assert state.get_active_dataframe_name() == "TestDataset"

    def test_load_dataset_not_active(self, mock_datasets_dir):
        """Test not setting the loaded dataset as active."""
        state = DataFrameState()
        
        with patch("stats_compass_core.data.load_dataset._DATASETS_DIR", mock_datasets_dir):
            params = LoadDatasetInput(name="TestDataset", set_active=False)
            load_dataset(state, params)
            
            assert state.get_active_dataframe_name() is None

    def test_load_dataset_with_extension(self, mock_datasets_dir):
        """Test loading a dataset when name includes .csv extension."""
        state = DataFrameState()
        
        with patch("stats_compass_core.data.load_dataset._DATASETS_DIR", mock_datasets_dir):
            params = LoadDatasetInput(name="TestDataset.csv")
            result = load_dataset(state, params)
            
            assert result.dataframe_name == "TestDataset"
            assert state.get_dataframe("TestDataset") is not None

