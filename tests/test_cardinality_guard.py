"""An ID column is not a category.

The reproduced failure: a 500-unique customer ID went into bin_rare_categories.
Every value was rarer than the threshold, so the whole column collapsed to
'Other', target encoding turned that into a constant, and the constant was
handed to the model as a feature. Nothing failed and nothing said so.

The guard: a column with more unique values than half its non-null rows, or
more than 200, is not binned or encoded. It is skipped with a named warning.
"""

import numpy as np
import pandas as pd
import pytest

from stats_compass_core.state import DataFrameState
from stats_compass_core.transforms.bin_rare_categories import (
    BinRareCategoriesInput,
    bin_rare_categories,
)
from stats_compass_core.transforms.mean_target_encoding import (
    MeanTargetEncodingInput,
    mean_target_encoding,
)


def _customers(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "region": rng.choice(["north", "south", "east", "west"], n),
        "spend": rng.normal(100, 20, n),
        "target": rng.integers(0, 2, n),
    })


def _bin(frame, columns, **kwargs):
    state = DataFrameState()
    state.set_dataframe(frame, "d", operation="test")
    result = bin_rare_categories(state, BinRareCategoriesInput(
        dataframe_name="d", categorical_columns=columns, **kwargs
    ))
    return state, result


class TestBinRareCategoriesGuard:
    def test_id_column_is_rejected_with_a_named_warning(self):
        state, result = _bin(_customers(), ["customer_id", "region"])

        assert result.success is True
        warning = next(w for w in result.warnings if w.code == "HIGH_CARDINALITY")
        assert warning.columns == ["customer_id"]
        assert "customer_id" in result.skipped_columns
        assert "customer_id" not in result.columns_processed
        assert "region" in result.columns_processed

        # Not collapsed to 'Other': the column is left exactly as it was.
        after = state.get_dataframe("d")["customer_id"]
        assert after.nunique() == 500

    def test_only_an_id_column_skips_rather_than_raising(self):
        _, result = _bin(_customers(), ["customer_id"])
        assert result.success is True
        assert result.columns_processed == []
        assert [w.code for w in result.warnings] == ["HIGH_CARDINALITY"]

    def test_over_200_uniques_is_rejected_even_below_half_the_rows(self):
        n = 1000
        frame = pd.DataFrame({
            "sku": [f"S{i % 250:04d}" for i in range(n)],  # 250 uniques, 25% of rows
            "target": np.arange(n) % 2,
        })
        _, result = _bin(frame, ["sku"])
        assert "sku" in result.skipped_columns

    def test_ordinary_categorical_below_both_limits_is_binned(self):
        n = 1000
        frame = pd.DataFrame({
            "city": [f"city_{i % 150}" for i in range(n)],  # 150 uniques, 15% of rows
            "target": np.arange(n) % 2,
        })
        _, result = _bin(frame, ["city"])
        assert result.skipped_columns == {}
        assert result.columns_processed == ["city"]

    def test_the_skip_is_in_the_op_log(self):
        state, _ = _bin(_customers(), ["customer_id", "region"])
        logged = [
            h for h in state.get_history()
            if h.operation == "warning" and h.details.get("code") == "HIGH_CARDINALITY"
        ]
        assert [h.details["columns"] for h in logged] == [["customer_id"]]

    def test_a_non_categorical_column_alongside_a_valid_one_is_reported(self):
        """It used to be dropped from the run with no trace."""
        _, result = _bin(_customers(), ["region", "spend"])
        assert result.columns_processed == ["region"]
        assert "spend" in result.skipped_columns
        assert any(w.columns == ["spend"] for w in result.warnings)


class TestMeanTargetEncodingGuard:
    def test_id_column_is_not_encoded(self):
        state = DataFrameState()
        state.set_dataframe(_customers(), "d", operation="test")
        result = mean_target_encoding(state, MeanTargetEncodingInput(
            dataframe_name="d", categorical_columns=["customer_id", "region"],
            target_column="target",
        ))
        assert result.original_columns == ["region"]
        assert "customer_id_encoded" not in state.get_dataframe("d").columns
        assert [w.columns for w in result.warnings if w.code == "HIGH_CARDINALITY"] == [
            ["customer_id"]
        ]

    def test_nothing_left_to_encode_is_an_error_that_says_why(self):
        state = DataFrameState()
        state.set_dataframe(_customers(), "d", operation="test")
        with pytest.raises(ValueError, match="customer_id"):
            mean_target_encoding(state, MeanTargetEncodingInput(
                dataframe_name="d", categorical_columns=["customer_id"],
                target_column="target",
            ))
