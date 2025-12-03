"""Tests for the drop_na cleaning tool."""
import pandas as pd
import pytest

from stats_compass_core.cleaning.dropna import DropNAInput, drop_na


class TestDropNA:
    """Test suite for drop_na tool."""
    
    def test_drop_rows_with_any_na(self) -> None:
        """Test dropping rows with any NA values."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [5, None, 7, 8],
            'C': [9, 10, 11, 12]
        })
        
        params = DropNAInput(axis=0, how='any')
        result = drop_na(df, params)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Only rows 0 and 3 have no NAs
        assert result['A'].tolist() == [1.0, 4.0]
    
    def test_drop_rows_with_all_na(self) -> None:
        """Test dropping rows where all values are NA."""
        df = pd.DataFrame({
            'A': [1, None, None],
            'B': [5, None, None],
            'C': [9, 10, 11]
        })
        
        params = DropNAInput(axis=0, how='all')
        result = drop_na(df, params)
        
        assert len(result) == 2  # Rows 0 and 2 kept
        assert result['C'].tolist() == [9, 11]
    
    def test_drop_columns_with_any_na(self) -> None:
        """Test dropping columns with any NA values."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, None, 6],
            'C': [7, 8, 9]
        })
        
        params = DropNAInput(axis=1, how='any')
        result = drop_na(df, params)
        
        assert list(result.columns) == ['A', 'C']
        assert len(result) == 3
    
    def test_drop_with_threshold(self) -> None:
        """Test dropping rows based on threshold of non-NA values."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [5, None, None, 8],
            'C': [9, 10, None, 12]
        })
        
        params = DropNAInput(axis=0, thresh=2)
        result = drop_na(df, params)
        
        assert len(result) == 3  # Row 2 has only 0 non-NA, dropped
    
    def test_drop_with_subset(self) -> None:
        """Test dropping rows based on NA in specific columns."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [5, 6, 7, 8],
            'C': [9, None, 11, 12]
        })
        
        params = DropNAInput(axis=0, how='any', subset=['B'])
        result = drop_na(df, params)
        
        assert len(result) == 4  # B has no NAs, all rows kept
        
        params = DropNAInput(axis=0, how='any', subset=['A'])
        result = drop_na(df, params)
        
        assert len(result) == 3  # Row 2 dropped (A is NA)
    
    def test_subset_column_not_found(self) -> None:
        """Test error when subset column doesn't exist."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        params = DropNAInput(subset=['B', 'C'])
        with pytest.raises(ValueError, match="Columns not found"):
            drop_na(df, params)
    
    def test_empty_dataframe(self) -> None:
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        
        params = DropNAInput()
        result = drop_na(df, params)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_no_na_values(self) -> None:
        """Test with DataFrame that has no NA values."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        params = DropNAInput(axis=0, how='any')
        result = drop_na(df, params)
        
        assert len(result) == 3
        pd.testing.assert_frame_equal(result, df)
