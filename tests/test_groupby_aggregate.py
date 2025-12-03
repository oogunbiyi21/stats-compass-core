"""Tests for the groupby_aggregate transform tool."""
import pandas as pd
import pytest

from stats_compass_core.transforms.groupby_aggregate import (
    GroupByAggregateInput,
    groupby_aggregate,
)


class TestGroupByAggregate:
    """Test suite for groupby_aggregate tool."""
    
    def test_basic_groupby_single_function(self) -> None:
        """Test basic groupby with single aggregation function."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40]
        })
        
        params = GroupByAggregateInput(
            by=['category'],
            agg_func={'value': 'sum'}
        )
        result = groupby_aggregate(df, params)
        
        assert isinstance(result, pd.DataFrame)
        assert result.loc['A', 'value'] == 40
        assert result.loc['B', 'value'] == 60
    
    def test_groupby_multiple_functions(self) -> None:
        """Test groupby with multiple aggregation functions."""
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'value1': [10, 20, 30, 40],
            'value2': [1, 2, 3, 4]
        })
        
        params = GroupByAggregateInput(
            by=['category'],
            agg_func={'value1': 'sum', 'value2': 'mean'}
        )
        result = groupby_aggregate(df, params)
        
        assert result.loc['A', 'value1'] == 30
        assert result.loc['A', 'value2'] == 1.5
        assert result.loc['B', 'value1'] == 70
        assert result.loc['B', 'value2'] == 3.5
    
    def test_groupby_list_of_functions(self) -> None:
        """Test groupby with list of aggregation functions per column."""
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })
        
        params = GroupByAggregateInput(
            by=['category'],
            agg_func={'value': ['sum', 'mean', 'count']}
        )
        result = groupby_aggregate(df, params)
        
        assert isinstance(result, pd.DataFrame)
        assert result.loc['A', ('value', 'sum')] == 30
        assert result.loc['A', ('value', 'mean')] == 15
        assert result.loc['A', ('value', 'count')] == 2
    
    def test_groupby_multiple_columns(self) -> None:
        """Test groupby with multiple group-by columns."""
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'subcategory': ['X', 'Y', 'X', 'Y'],
            'value': [10, 20, 30, 40]
        })
        
        params = GroupByAggregateInput(
            by=['category', 'subcategory'],
            agg_func={'value': 'sum'}
        )
        result = groupby_aggregate(df, params)
        
        assert isinstance(result, pd.DataFrame)
        assert result.loc[('A', 'X'), 'value'] == 10
        assert result.loc[('B', 'Y'), 'value'] == 40
    
    def test_groupby_as_index_false(self) -> None:
        """Test groupby with as_index=False."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40]
        })
        
        params = GroupByAggregateInput(
            by=['category'],
            agg_func={'value': 'sum'},
            as_index=False
        )
        result = groupby_aggregate(df, params)
        
        assert 'category' in result.columns
        assert result[result['category'] == 'A']['value'].values[0] == 40
        assert result[result['category'] == 'B']['value'].values[0] == 60
    
    def test_groupby_column_not_found(self) -> None:
        """Test error when group-by column doesn't exist."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        params = GroupByAggregateInput(
            by=['B'],
            agg_func={'A': 'sum'}
        )
        with pytest.raises(ValueError, match="Group-by columns not found"):
            groupby_aggregate(df, params)
    
    def test_agg_column_not_found(self) -> None:
        """Test error when aggregation column doesn't exist."""
        df = pd.DataFrame({'A': [1, 2, 3], 'category': ['X', 'X', 'Y']})
        
        params = GroupByAggregateInput(
            by=['category'],
            agg_func={'B': 'sum'}
        )
        with pytest.raises(ValueError, match="Aggregation columns not found"):
            groupby_aggregate(df, params)
    
    def test_invalid_aggregation_function(self) -> None:
        """Test error with invalid aggregation function."""
        df = pd.DataFrame({
            'category': ['A', 'B'],
            'value': ['x', 'y']  # Non-numeric
        })
        
        params = GroupByAggregateInput(
            by=['category'],
            agg_func={'value': 'sum'}  # Can't sum strings meaningfully in this context
        )
        # This might not fail for string concatenation, so let's use a clearly invalid func
        params = GroupByAggregateInput(
            by=['category'],
            agg_func={'value': 'invalid_function'}
        )
        with pytest.raises(TypeError, match="Aggregation failed"):
            groupby_aggregate(df, params)
    
    def test_empty_dataframe(self) -> None:
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=['category', 'value'])
        
        params = GroupByAggregateInput(
            by=['category'],
            agg_func={'value': 'sum'}
        )
        result = groupby_aggregate(df, params)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
