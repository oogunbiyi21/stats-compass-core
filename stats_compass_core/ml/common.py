"""Common models and utilities for ML tools."""

from pydantic import BaseModel


class TrainedModelResult(BaseModel):
    """Result of model training."""

    model_config = {"arbitrary_types_allowed": True}

    model: object
    feature_columns: list[str]
    target_column: str
    train_score: float
    n_samples: int
