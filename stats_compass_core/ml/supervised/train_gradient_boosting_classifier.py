"""
Tool for training a gradient boosting classifier.
"""

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.ml.common import create_training_result, prepare_ml_data
from stats_compass_core.registry import registry
from stats_compass_core.results import ModelTrainingResult
from stats_compass_core.state import DataFrameState


class TrainGradientBoostingClassifierInput(StrictToolInput):
    """Input schema for train_gradient_boosting_classifier tool."""

    dataframe_name: str | None = Field(
        default=None, description="Name of DataFrame to train on. Uses active if not specified."
    )
    target_column: str = Field(description="Name of the target column to predict")
    feature_columns: list[str] | None = Field(
        default=None,
        description=(
            "List of feature columns. "
            "If None, uses all numeric columns except target"
        ),
    )
    test_size: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Fraction of data to use for testing"
    )
    random_state: int | None = Field(
        default=42, description="Random seed for reproducibility"
    )
    n_estimators: int = Field(
        default=100, ge=1, description="Number of boosting stages"
    )
    learning_rate: float = Field(
        default=0.1, gt=0.0, le=1.0, description="Learning rate shrinks contribution of each tree"
    )
    save_path: str | None = Field(
        default=None, description="Path to save the trained model (e.g., 'model.joblib')"
    )


@registry.register(
    category="ml",
    input_schema=TrainGradientBoostingClassifierInput,
    description="Train a gradient boosting classifier",
)
def train_gradient_boosting_classifier(
    state: DataFrameState, params: TrainGradientBoostingClassifierInput
) -> ModelTrainingResult:
    """
    Train a gradient boosting classifier on DataFrame data.

    Note: Requires scikit-learn to be installed (install with 'ml' extra).

    Args:
        state: DataFrameState containing the DataFrame to train on
        params: Parameters for model training

    Returns:
        ModelTrainingResult with model stored in state and metrics

    Raises:
        ImportError: If scikit-learn is not installed
        ValueError: If target or feature columns don't exist or data is insufficient
    """
    # Prepare data
    X, y, feature_cols, source_name = prepare_ml_data(
        state, params.target_column, params.feature_columns, params.dataframe_name
    )

    # Train model
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params.test_size, random_state=params.random_state
        )
        
        # Capture indices for predictions DataFrame
        train_indices = X_train.index
        test_indices = X_test.index

        model = GradientBoostingClassifier(
            n_estimators=params.n_estimators,
            learning_rate=params.learning_rate,
            random_state=params.random_state
        )
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test) if len(X_test) > 0 else None

    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for ML tools. "
            "Install with: pip install stats-compass-core[ml]"
        ) from e

    return create_training_result(
        state=state,
        model=model,
        model_type="gradient_boosting_classifier",
        target_column=params.target_column,
        feature_cols=feature_cols,
        train_score=train_score,
        test_score=test_score,
        train_size=len(X_train),
        test_size=len(X_test) if params.test_size > 0 else None,
        source_name=source_name,
        hyperparameters={
            "test_size": params.test_size,
            "random_state": params.random_state,
            "n_estimators": params.n_estimators,
            "learning_rate": params.learning_rate,
        },
        save_path=params.save_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_indices=train_indices,
        test_indices=test_indices,
        is_classifier=True,
    )
