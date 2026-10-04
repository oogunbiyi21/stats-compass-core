"""
Tool for training a random forest classifier.
"""

from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.ml.common import create_training_result, prepare_ml_data
from stats_compass_core.registry import registry
from stats_compass_core.results import ModelTrainingResult
from stats_compass_core.state import DataFrameState


class TrainRandomForestClassifierInput(StrictToolInput):
    """Input schema for train_random_forest_classifier tool."""

    dataframe_name: str | None = Field(
        default=None, description="Name of DataFrame to train on. Uses active if not specified."
    )
    target_column: str = Field(description="Name of the target column to predict")
    feature_columns: list[str] | None = Field(
        default=None,
        description=(
            "List of feature columns. "
            "If None, uses all numeric columns except target and the result "
            "carries a FEATURES_INFERRED warning naming them. Declare them when "
            "any column may have been recorded after the outcome."
        ),
    )
    test_size: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Fraction of data to use for testing"
    )
    random_state: int | None = Field(
        default=42, description="Random seed for reproducibility"
    )
    n_estimators: int = Field(
        default=100, ge=1, description="Number of trees in the forest"
    )
    max_depth: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum depth of each tree. Unlimited by default, which lets the "
            "forest memorise the training set — a train score of 1.00 alongside a "
            "far lower test score is the sign. Values around 4-8 usually help."
        ),
    )
    min_samples_leaf: int = Field(
        default=1,
        ge=1,
        description=(
            "Minimum samples required at a leaf. Raising it (5-20) is the other "
            "main lever against memorising small datasets."
        ),
    )
    min_samples_split: int = Field(
        default=2, ge=2, description="Minimum samples required to split a node"
    )
    class_weight: str | None = Field(
        default=None,
        description=(
            "Set to 'balanced' when classes are imbalanced. Without it a model "
            "trained on, say, 20% positives can score well by predicting the "
            "majority class for everyone and never catching the cases you care about."
        ),
    )
    save_path: str | None = Field(
        default=None, description="Path to save the trained model (e.g., 'model.joblib')"
    )


@registry.register(
    category="ml",
    input_schema=TrainRandomForestClassifierInput,
    description="Train a random forest classifier",
)
def train_random_forest_classifier(
    state: DataFrameState, params: TrainRandomForestClassifierInput
) -> ModelTrainingResult:
    """
    Train a random forest classifier on DataFrame data.

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
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params.test_size, random_state=params.random_state
        )

        # Capture indices for predictions DataFrame
        train_indices = X_train.index
        test_indices = X_test.index

        model = RandomForestClassifier(
            n_estimators=params.n_estimators,
            random_state=params.random_state,
            max_depth=params.max_depth,
            min_samples_leaf=params.min_samples_leaf,
            min_samples_split=params.min_samples_split,
            class_weight=params.class_weight,
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
        model_type="random_forest_classifier",
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
        },
        save_path=params.save_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_indices=train_indices,
        test_indices=test_indices,
        is_classifier=True,
        features_declared=bool(params.feature_columns),
    )
