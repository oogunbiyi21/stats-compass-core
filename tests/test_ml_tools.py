"""Tests for ML tools."""

import pandas as pd
import pytest

# Check if sklearn is available
try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from stats_compass_core.state import DataFrameState

# Only import ML tools if sklearn is available
if HAS_SKLEARN:
    from stats_compass_core.ml.supervised.train_linear_regression import (
        TrainLinearRegressionInput,
        train_linear_regression,
    )
    from stats_compass_core.ml.supervised.train_random_forest_classifier import (
        TrainRandomForestClassifierInput,
        train_random_forest_classifier,
    )


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
class TestTrainLinearRegression:
    """Tests for train_linear_regression tool."""

    @pytest.fixture
    def state_with_data(self):
        """Create state with training data."""
        state = DataFrameState()
        df = pd.DataFrame({
            "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "x2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "y": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
        })
        state.set_dataframe(df, name="train_data", operation="test")
        return state

    def test_train_basic(self, state_with_data):
        """Test basic training."""
        params = TrainLinearRegressionInput(
            target_column="y",
            feature_columns=["x1", "x2"],
        )

        result = train_linear_regression(state_with_data, params)

        assert result.model_type == "linear_regression"
        assert result.target_column == "y"
        assert "x1" in result.feature_columns
        assert result.model_id is not None

    def test_train_model_stored(self, state_with_data):
        """Test that model is stored in state."""
        params = TrainLinearRegressionInput(
            target_column="y",
            feature_columns=["x1"],
        )

        result = train_linear_regression(state_with_data, params)

        # Model should be retrievable from state
        model = state_with_data.get_model(result.model_id)
        assert model is not None

    def test_train_with_test_split(self, state_with_data):
        """Test training with test split."""
        params = TrainLinearRegressionInput(
            target_column="y",
            feature_columns=["x1"],
            test_size=0.2,
        )

        result = train_linear_regression(state_with_data, params)

        assert result.test_size is not None
        assert "test_score" in result.metrics

    def test_train_json_serializable(self, state_with_data):
        """Test that result is JSON serializable."""
        params = TrainLinearRegressionInput(
            target_column="y",
            feature_columns=["x1"],
        )

        result = train_linear_regression(state_with_data, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
class TestTrainRandomForestClassifier:
    """Tests for train_random_forest_classifier tool."""

    @pytest.fixture
    def state_with_data(self):
        """Create state with training data."""
        state = DataFrameState()
        df = pd.DataFrame({
            "f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "f2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        })
        state.set_dataframe(df, name="train_data", operation="test")
        return state

    def test_train_basic(self, state_with_data):
        """Test basic training."""
        params = TrainRandomForestClassifierInput(
            target_column="target",
            feature_columns=["f1", "f2"],
            n_estimators=10,
        )

        result = train_random_forest_classifier(state_with_data, params)

        assert result.model_type == "random_forest_classifier"
        assert result.model_id is not None

    def test_train_model_stored(self, state_with_data):
        """Test that model is stored in state."""
        params = TrainRandomForestClassifierInput(
            target_column="target",
            feature_columns=["f1"],
            n_estimators=5,
        )

        result = train_random_forest_classifier(state_with_data, params)

        model = state_with_data.get_model(result.model_id)
        assert model is not None

    def test_train_has_feature_importance(self, state_with_data):
        """Test that feature importance is returned."""
        params = TrainRandomForestClassifierInput(
            target_column="target",
            feature_columns=["f1", "f2"],
            n_estimators=10,
        )

        result = train_random_forest_classifier(state_with_data, params)

        assert result.feature_importances is not None
        assert "f1" in result.feature_importances

    def test_train_json_serializable(self, state_with_data):
        """Test that result is JSON serializable."""
        params = TrainRandomForestClassifierInput(
            target_column="target",
            feature_columns=["f1"],
            n_estimators=5,
        )

        result = train_random_forest_classifier(state_with_data, params)
        json_str = result.model_dump_json()

        assert isinstance(json_str, str)

    def test_train_creates_predictions_dataframe(self, state_with_data):
        """Test that training creates a predictions DataFrame."""
        params = TrainRandomForestClassifierInput(
            target_column="target",
            feature_columns=["f1", "f2"],
            n_estimators=10,
            test_size=0.2,
        )

        result = train_random_forest_classifier(state_with_data, params)

        # Should have predictions DataFrame
        assert result.predictions_dataframe is not None
        assert result.predictions_dataframe == "train_data_predictions"
        
        # Predictions DataFrame should be stored in state
        pred_df = state_with_data.get_dataframe(result.predictions_dataframe)
        assert pred_df is not None
        
        # Should have prediction column
        assert result.prediction_column is not None
        assert result.prediction_column == "pred_target"
        assert result.prediction_column in pred_df.columns
        
        # Should have split column
        assert "target_split" in pred_df.columns
        assert set(pred_df["target_split"].unique()) == {"train", "test"}

    def test_train_creates_probability_columns(self, state_with_data):
        """Test that classifier training creates probability columns."""
        params = TrainRandomForestClassifierInput(
            target_column="target",
            feature_columns=["f1", "f2"],
            n_estimators=10,
            test_size=0.2,
        )

        result = train_random_forest_classifier(state_with_data, params)

        # Should have probability columns for binary classification
        assert result.probability_columns is not None
        assert len(result.probability_columns) == 2  # 2 classes (0 and 1)
        
        # Should have class labels
        assert result.class_labels is not None
        assert set(result.class_labels) == {0, 1}
        
        # Probability columns should exist in predictions DataFrame
        pred_df = state_with_data.get_dataframe(result.predictions_dataframe)
        for prob_col in result.probability_columns:
            assert prob_col in pred_df.columns
            # Probabilities should be between 0 and 1
            assert pred_df[prob_col].dropna().between(0, 1).all()


@pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
class TestLinearRegressionPredictions:
    """Tests for linear regression predictions DataFrame."""

    @pytest.fixture
    def state_with_data(self):
        """Create state with training data."""
        state = DataFrameState()
        df = pd.DataFrame({
            "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "x2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "y": [3.1, 6.2, 8.9, 12.1, 15.0, 17.9, 21.2, 24.0, 27.1, 30.0],
        })
        state.set_dataframe(df, name="train_data", operation="test")
        return state

    def test_regression_creates_predictions_dataframe(self, state_with_data):
        """Test that regression training creates a predictions DataFrame."""
        params = TrainLinearRegressionInput(
            target_column="y",
            feature_columns=["x1", "x2"],
            test_size=0.2,
        )

        result = train_linear_regression(state_with_data, params)

        # Should have predictions DataFrame
        assert result.predictions_dataframe is not None
        
        # Predictions DataFrame should be stored in state
        pred_df = state_with_data.get_dataframe(result.predictions_dataframe)
        assert pred_df is not None
        
        # Should have prediction column
        assert result.prediction_column is not None
        assert result.prediction_column in pred_df.columns
        
        # Regression should NOT have probability columns
        assert result.probability_columns is None
        assert result.class_labels is None

    def test_regression_predictions_are_numeric(self, state_with_data):
        """Test that regression predictions are numeric."""
        params = TrainLinearRegressionInput(
            target_column="y",
            feature_columns=["x1"],
            test_size=0.2,
        )

        result = train_linear_regression(state_with_data, params)
        pred_df = state_with_data.get_dataframe(result.predictions_dataframe)
        
        # Predictions should be numeric
        assert pred_df[result.prediction_column].dtype in ['float64', 'float32']
