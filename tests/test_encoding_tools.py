"""
Tests for encoding tools: mean_target_encoding and bin_rare_categories.
"""

import pytest
import pandas as pd
import numpy as np

from stats_compass_core.state import DataFrameState
from stats_compass_core.results import MeanTargetEncodingResult, BinRareCategoriesResult
from stats_compass_core.transforms.bin_rare_categories import (
    bin_rare_categories,
    BinRareCategoriesInput,
)

# Conditional import for mean_target_encoding (requires sklearn)
try:
    from stats_compass_core.transforms.mean_target_encoding import (
        mean_target_encoding,
        MeanTargetEncodingInput,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def basic_categorical_df():
    """DataFrame with basic categorical columns and binary target."""
    return pd.DataFrame({
        "category_a": ["red", "blue", "green", "red", "blue", "green", "red", "blue"] * 10,
        "category_b": ["small", "medium", "large", "small", "medium", "large", "small", "medium"] * 10,
        "numeric": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 10,
        "target": [0, 1, 0, 1, 1, 0, 1, 0] * 10,
    })


@pytest.fixture
def rare_categories_df():
    """DataFrame with some rare categories."""
    # Most values are 'common1' and 'common2', a few are rare
    categories = ["common1"] * 40 + ["common2"] * 35 + ["rare1"] * 3 + ["rare2"] * 2
    np.random.shuffle(categories)
    
    return pd.DataFrame({
        "category": categories,
        "value": np.random.randn(80),
        "target": np.random.randint(0, 2, 80),
    })


@pytest.fixture
def multiclass_target_df():
    """DataFrame with multiclass target."""
    return pd.DataFrame({
        "category": ["A", "B", "C", "A", "B", "C"] * 20,
        "target": ["class1", "class2", "class3", "class1", "class2", "class3"] * 20,
    })


@pytest.fixture
def continuous_target_df():
    """DataFrame with continuous target."""
    return pd.DataFrame({
        "category": ["low", "medium", "high", "low", "medium", "high"] * 20,
        "target": np.random.randn(120),
    })


@pytest.fixture
def missing_values_df():
    """DataFrame with missing values in categorical columns."""
    return pd.DataFrame({
        "category": ["A", "B", None, "A", "B", None, "C", "C"] * 10,
        "target": [0, 1, 0, 1, 1, 0, 1, 0] * 10,
    })


# =============================================================================
# Tests for bin_rare_categories
# =============================================================================

class TestBinRareCategories:
    """Tests for bin_rare_categories function."""

    def test_basic_binning(self, rare_categories_df):
        """Test basic rare category binning."""
        state = DataFrameState()
        state.set_dataframe(rare_categories_df, name="test", operation="test")

        params = BinRareCategoriesInput(
            dataframe_name="test",
            categorical_columns=["category"],
            threshold=0.05,  # 5% threshold -> rare1 and rare2 should be binned
        )

        result = bin_rare_categories(state, params)

        assert isinstance(result, BinRareCategoriesResult)
        assert result.success is True
        assert "category" in result.columns_processed
        assert "category" in result.columns_modified
        
        # Check that rare categories were binned
        df_result = state.get_dataframe("test")
        unique_cats = df_result["category"].unique()
        assert "rare1" not in unique_cats
        assert "rare2" not in unique_cats
        assert "Other" in unique_cats

    def test_no_rare_categories(self, basic_categorical_df):
        """Test when no categories are rare."""
        state = DataFrameState()
        state.set_dataframe(basic_categorical_df, name="test", operation="test")

        params = BinRareCategoriesInput(
            dataframe_name="test",
            categorical_columns=["category_a"],
            threshold=0.01,  # Very low threshold - nothing should be binned
        )

        result = bin_rare_categories(state, params)

        assert result.success is True
        assert result.columns_modified == []
        assert "No rare categories found" in result.message

    def test_custom_bin_label(self, rare_categories_df):
        """Test custom bin label."""
        state = DataFrameState()
        state.set_dataframe(rare_categories_df, name="test", operation="test")

        params = BinRareCategoriesInput(
            dataframe_name="test",
            categorical_columns=["category"],
            threshold=0.05,
            bin_label="RARE",
        )

        result = bin_rare_categories(state, params)

        assert result.success is True
        assert result.bin_label == "RARE"
        
        df_result = state.get_dataframe("test")
        assert "RARE" in df_result["category"].unique()

    def test_min_count_threshold(self, rare_categories_df):
        """Test using min_count instead of threshold."""
        state = DataFrameState()
        state.set_dataframe(rare_categories_df, name="test", operation="test")

        params = BinRareCategoriesInput(
            dataframe_name="test",
            categorical_columns=["category"],
            threshold=0.5,  # This would be ignored
            min_count=5,  # Categories with <5 occurrences are binned
        )

        result = bin_rare_categories(state, params)

        assert result.success is True
        # rare1 (3) and rare2 (2) should be binned
        assert len(result.binning_details["category"]["rare_categories"]) >= 2

    def test_multiple_columns(self, basic_categorical_df):
        """Test binning multiple columns."""
        state = DataFrameState()
        # Add some rare categories
        df = basic_categorical_df.copy()
        df.loc[0, "category_a"] = "very_rare"
        df.loc[1, "category_b"] = "ultra_rare"
        state.set_dataframe(df, name="test", operation="test")

        params = BinRareCategoriesInput(
            dataframe_name="test",
            categorical_columns=["category_a", "category_b"],
            threshold=0.05,
        )

        result = bin_rare_categories(state, params)

        assert result.success is True
        assert len(result.columns_processed) == 2
        assert "category_a" in result.binning_details
        assert "category_b" in result.binning_details

    def test_save_as_new_dataframe(self, rare_categories_df):
        """Test saving result as new DataFrame."""
        state = DataFrameState()
        state.set_dataframe(rare_categories_df, name="original", operation="test")

        params = BinRareCategoriesInput(
            dataframe_name="original",
            categorical_columns=["category"],
            threshold=0.05,
            save_as="binned",
        )

        result = bin_rare_categories(state, params)

        assert result.dataframe_name == "binned"
        assert result.source_dataframe == "original"
        
        # Both DataFrames should exist
        df_names = [df.name for df in state.list_dataframes()]
        assert "original" in df_names
        assert "binned" in df_names

    def test_category_mapping(self, rare_categories_df):
        """Test that category mapping is correctly populated."""
        state = DataFrameState()
        state.set_dataframe(rare_categories_df, name="test", operation="test")

        params = BinRareCategoriesInput(
            dataframe_name="test",
            categorical_columns=["category"],
            threshold=0.05,
        )

        result = bin_rare_categories(state, params)

        assert "category" in result.category_mapping
        mapping = result.category_mapping["category"]
        
        # Common categories should map to themselves
        assert mapping["common1"] == "common1"
        assert mapping["common2"] == "common2"
        
        # Rare categories should map to 'Other'
        assert mapping["rare1"] == "Other"
        assert mapping["rare2"] == "Other"

    def test_invalid_column_error(self, basic_categorical_df):
        """Test error when column doesn't exist."""
        state = DataFrameState()
        state.set_dataframe(basic_categorical_df, name="test", operation="test")

        params = BinRareCategoriesInput(
            dataframe_name="test",
            categorical_columns=["nonexistent"],
            threshold=0.05,
        )

        with pytest.raises(ValueError, match="Columns not found"):
            bin_rare_categories(state, params)

    def test_numeric_column_error(self, basic_categorical_df):
        """Test error when column is numeric."""
        state = DataFrameState()
        state.set_dataframe(basic_categorical_df, name="test", operation="test")

        params = BinRareCategoriesInput(
            dataframe_name="test",
            categorical_columns=["numeric"],
            threshold=0.05,
        )

        with pytest.raises(ValueError, match="not categorical"):
            bin_rare_categories(state, params)


# =============================================================================
# Tests for mean_target_encoding
# =============================================================================

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestMeanTargetEncoding:
    """Tests for mean_target_encoding function."""

    def test_basic_encoding_binary_target(self, basic_categorical_df):
        """Test basic encoding with binary target."""
        state = DataFrameState()
        state.set_dataframe(basic_categorical_df, name="test", operation="test")

        params = MeanTargetEncodingInput(
            dataframe_name="test",
            categorical_columns=["category_a"],
            target_column="target",
        )

        result = mean_target_encoding(state, params)

        assert isinstance(result, MeanTargetEncodingResult)
        assert result.success is True
        assert "category_a" in result.original_columns
        assert "category_a_encoded" in result.encoded_columns
        
        # Check encoded column exists in DataFrame
        df_result = state.get_dataframe("test")
        assert "category_a_encoded" in df_result.columns

    def test_encoding_continuous_target(self, continuous_target_df):
        """Test encoding with continuous target."""
        state = DataFrameState()
        state.set_dataframe(continuous_target_df, name="test", operation="test")

        params = MeanTargetEncodingInput(
            dataframe_name="test",
            categorical_columns=["category"],
            target_column="target",
            target_type="continuous",
        )

        result = mean_target_encoding(state, params)

        assert result.success is True
        assert result.parameters["effective_target_type"] == "continuous"

    def test_encoding_multiclass_target(self, multiclass_target_df):
        """Test encoding with multiclass target creates multiple columns."""
        state = DataFrameState()
        state.set_dataframe(multiclass_target_df, name="test", operation="test")

        params = MeanTargetEncodingInput(
            dataframe_name="test",
            categorical_columns=["category"],
            target_column="target",
            target_type="multiclass",
        )

        result = mean_target_encoding(state, params)

        assert result.success is True
        # Multiclass should create multiple encoded columns per feature
        assert len(result.encoded_columns) >= 1
        
        # Column mapping should reflect multiple columns
        mapping = result.column_mapping["category"]
        if isinstance(mapping, list):
            assert len(mapping) > 1  # Multiple columns for multiclass

    def test_multiple_columns(self, basic_categorical_df):
        """Test encoding multiple columns."""
        state = DataFrameState()
        state.set_dataframe(basic_categorical_df, name="test", operation="test")

        params = MeanTargetEncodingInput(
            dataframe_name="test",
            categorical_columns=["category_a", "category_b"],
            target_column="target",
        )

        result = mean_target_encoding(state, params)

        assert result.success is True
        assert len(result.original_columns) == 2
        assert "category_a" in result.column_mapping
        assert "category_b" in result.column_mapping

    def test_encoder_stored_in_state(self, basic_categorical_df):
        """Test that encoder is stored for later use."""
        state = DataFrameState()
        state.set_dataframe(basic_categorical_df, name="test", operation="test")

        params = MeanTargetEncodingInput(
            dataframe_name="test",
            categorical_columns=["category_a"],
            target_column="target",
        )

        result = mean_target_encoding(state, params)

        assert result.encoder_id is not None
        
        # Verify encoder can be retrieved
        models = state.list_models()
        assert any(result.encoder_id in m.model_id for m in models)

    def test_custom_cv_folds(self, basic_categorical_df):
        """Test custom cross-validation folds."""
        state = DataFrameState()
        state.set_dataframe(basic_categorical_df, name="test", operation="test")

        params = MeanTargetEncodingInput(
            dataframe_name="test",
            categorical_columns=["category_a"],
            target_column="target",
            cv=3,
        )

        result = mean_target_encoding(state, params)

        assert result.success is True
        assert result.parameters["cv"] == 3

    def test_missing_values_handling(self, missing_values_df):
        """Test that missing values are handled."""
        state = DataFrameState()
        state.set_dataframe(missing_values_df, name="test", operation="test")

        params = MeanTargetEncodingInput(
            dataframe_name="test",
            categorical_columns=["category"],
            target_column="target",
        )

        result = mean_target_encoding(state, params)

        assert result.success is True
        # Should complete without error

    def test_replace_original_columns(self, basic_categorical_df):
        """Test replacing original columns instead of creating new ones."""
        state = DataFrameState()
        state.set_dataframe(basic_categorical_df, name="test", operation="test")

        params = MeanTargetEncodingInput(
            dataframe_name="test",
            categorical_columns=["category_a"],
            target_column="target",
            create_new_columns=False,
        )

        result = mean_target_encoding(state, params)

        assert result.success is True
        
        df_result = state.get_dataframe("test")
        # Original column should be removed
        assert "category_a" not in df_result.columns
        # Encoded column should exist
        assert "category_a_encoded" in df_result.columns

    def test_save_as_new_dataframe(self, basic_categorical_df):
        """Test saving result as new DataFrame."""
        state = DataFrameState()
        state.set_dataframe(basic_categorical_df, name="original", operation="test")

        params = MeanTargetEncodingInput(
            dataframe_name="original",
            categorical_columns=["category_a"],
            target_column="target",
            save_as="encoded",
        )

        result = mean_target_encoding(state, params)

        assert result.dataframe_name == "encoded"
        assert result.source_dataframe == "original"

    def test_encoding_stats(self, basic_categorical_df):
        """Test that encoding statistics are populated."""
        state = DataFrameState()
        state.set_dataframe(basic_categorical_df, name="test", operation="test")

        params = MeanTargetEncodingInput(
            dataframe_name="test",
            categorical_columns=["category_a"],
            target_column="target",
        )

        result = mean_target_encoding(state, params)

        assert "category_a" in result.encoding_stats
        stats = result.encoding_stats["category_a"]
        assert "original_unique_categories" in stats
        assert stats["original_unique_categories"] == 3  # red, blue, green

    def test_invalid_target_column_error(self, basic_categorical_df):
        """Test error when target column doesn't exist."""
        state = DataFrameState()
        state.set_dataframe(basic_categorical_df, name="test", operation="test")

        params = MeanTargetEncodingInput(
            dataframe_name="test",
            categorical_columns=["category_a"],
            target_column="nonexistent",
        )

        with pytest.raises(ValueError, match="Target column.*not found"):
            mean_target_encoding(state, params)

    def test_invalid_categorical_column_error(self, basic_categorical_df):
        """Test error when categorical column doesn't exist."""
        state = DataFrameState()
        state.set_dataframe(basic_categorical_df, name="test", operation="test")

        params = MeanTargetEncodingInput(
            dataframe_name="test",
            categorical_columns=["nonexistent"],
            target_column="target",
        )

        with pytest.raises(ValueError, match="Categorical columns not found"):
            mean_target_encoding(state, params)

    def test_target_column_excluded_from_encoding(self, basic_categorical_df):
        """Test that target column is excluded if accidentally included."""
        state = DataFrameState()
        # Add categorical target
        df = basic_categorical_df.copy()
        df["cat_target"] = ["yes", "no", "yes", "no", "yes", "no", "yes", "no"] * 10
        state.set_dataframe(df, name="test", operation="test")

        params = MeanTargetEncodingInput(
            dataframe_name="test",
            categorical_columns=["category_a", "cat_target"],
            target_column="cat_target",
        )

        result = mean_target_encoding(state, params)

        assert result.success is True
        # Target should be excluded from encoding
        assert "cat_target" not in result.original_columns
        assert "category_a" in result.original_columns


# =============================================================================
# Integration Tests
# =============================================================================

class TestEncodingIntegration:
    """Integration tests combining both encoding tools."""

    def test_bin_then_encode(self, rare_categories_df):
        """Test binning rare categories then target encoding."""
        if not SKLEARN_AVAILABLE:
            pytest.skip("scikit-learn not installed")
        
        state = DataFrameState()
        state.set_dataframe(rare_categories_df, name="data", operation="test")

        # First, bin rare categories
        bin_params = BinRareCategoriesInput(
            dataframe_name="data",
            categorical_columns=["category"],
            threshold=0.05,
        )
        bin_result = bin_rare_categories(state, bin_params)
        assert bin_result.success is True

        # Then, apply target encoding
        encode_params = MeanTargetEncodingInput(
            dataframe_name="data",
            categorical_columns=["category"],
            target_column="target",
        )
        encode_result = mean_target_encoding(state, encode_params)
        assert encode_result.success is True

        # Verify the pipeline worked
        df_result = state.get_dataframe("data")
        assert "category_encoded" in df_result.columns
        
        # The "Other" category from binning should now be encoded
        assert "Other" in df_result["category"].unique()
