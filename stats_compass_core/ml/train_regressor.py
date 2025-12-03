"""
Tool for training a regression model on DataFrame data.
"""
import pandas as pd
from pydantic import BaseModel, Field

from stats_compass_core.registry import registry


class TrainedRegressorResult(BaseModel):
    """Result of regressor training."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    model: object
    feature_columns: list[str]
    target_column: str
    train_score: float
    n_samples: int


class TrainRegressorInput(BaseModel):
    """Input schema for train_regressor tool."""
    
    target_column: str = Field(
        description="Name of the target column to predict"
    )
    feature_columns: list[str] | None = Field(
        default=None,
        description="List of feature columns. If None, uses all numeric columns except target"
    )
    model_type: str = Field(
        default="random_forest",
        pattern="^(linear_regression|random_forest|gradient_boosting)$",
        description="Type of regressor: 'linear_regression', 'random_forest', 'gradient_boosting'"
    )
    test_size: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Fraction of data to use for testing"
    )
    random_state: int | None = Field(
        default=42,
        description="Random seed for reproducibility"
    )


@registry.register(
    category="ml",
    input_schema=TrainRegressorInput,
    description="Train a regression model on DataFrame data"
)
def train_regressor(df: pd.DataFrame, params: TrainRegressorInput) -> TrainedRegressorResult:
    """
    Train a regression model on DataFrame data.
    
    Note: Requires scikit-learn to be installed (install with 'ml' extra).
    
    Args:
        df: Input DataFrame with features and target
        params: Parameters for model training
    
    Returns:
        TrainedRegressorResult containing trained model and metadata
    
    Raises:
        ImportError: If scikit-learn is not installed
        ValueError: If target or feature columns don't exist or data is insufficient
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for ML tools. "
            "Install with: pip install stats-compass-core[ml]"
        ) from e
    
    # Validate target column
    if params.target_column not in df.columns:
        raise ValueError(f"Target column '{params.target_column}' not found in DataFrame")
    
    # Determine feature columns
    if params.feature_columns:
        feature_cols = params.feature_columns
        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Feature columns not found: {missing_cols}")
    else:
        # Use all numeric columns except target
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != params.target_column]
        if not feature_cols:
            raise ValueError("No numeric feature columns available")
    
    # Prepare data
    X = df[feature_cols]
    y = df[params.target_column]
    
    # Check for sufficient data
    if len(df) < 2:
        raise ValueError("Insufficient data: need at least 2 samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params.test_size, random_state=params.random_state
    )
    
    # Select and train model
    if params.model_type == "linear_regression":
        model = LinearRegression()
    elif params.model_type == "random_forest":
        model = RandomForestRegressor(random_state=params.random_state)
    else:  # gradient_boosting
        model = GradientBoostingRegressor(random_state=params.random_state)
    
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    
    return TrainedRegressorResult(
        model=model,
        feature_columns=feature_cols,
        target_column=params.target_column,
        train_score=train_score,
        n_samples=len(X_train)
    )
