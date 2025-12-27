"""
Supervised machine learning tools.

This module provides training and evaluation tools for supervised learning:

Classification:
- train_logistic_regression: Train logistic regression classifier
- train_random_forest_classifier: Train random forest classifier
- train_gradient_boosting_classifier: Train gradient boosting classifier
- evaluate_classification_model: Evaluate classifier performance

Regression:
- train_linear_regression: Train linear regression model
- train_random_forest_regressor: Train random forest regressor
- train_gradient_boosting_regressor: Train gradient boosting regressor
- evaluate_regression_model: Evaluate regressor performance

Note: Requires scikit-learn. Install with: pip install stats-compass-core[ml]
"""

# Re-export all tools for convenience
from stats_compass_core.ml.supervised.train_logistic_regression import (
    TrainLogisticRegressionInput,
    train_logistic_regression,
)
from stats_compass_core.ml.supervised.train_random_forest_classifier import (
    TrainRandomForestClassifierInput,
    train_random_forest_classifier,
)
from stats_compass_core.ml.supervised.train_gradient_boosting_classifier import (
    TrainGradientBoostingClassifierInput,
    train_gradient_boosting_classifier,
)
from stats_compass_core.ml.supervised.train_linear_regression import (
    TrainLinearRegressionInput,
    train_linear_regression,
)
from stats_compass_core.ml.supervised.train_random_forest_regressor import (
    TrainRandomForestRegressorInput,
    train_random_forest_regressor,
)
from stats_compass_core.ml.supervised.train_gradient_boosting_regressor import (
    TrainGradientBoostingRegressorInput,
    train_gradient_boosting_regressor,
)
from stats_compass_core.ml.supervised.evaluate_classification_model import (
    EvaluateClassificationInput,
    evaluate_classification_model,
)
from stats_compass_core.ml.supervised.evaluate_regression_model import (
    EvaluateRegressionInput,
    evaluate_regression_model,
)

__all__ = [
    # Classification training
    "TrainLogisticRegressionInput",
    "train_logistic_regression",
    "TrainRandomForestClassifierInput",
    "train_random_forest_classifier",
    "TrainGradientBoostingClassifierInput",
    "train_gradient_boosting_classifier",
    # Regression training
    "TrainLinearRegressionInput",
    "train_linear_regression",
    "TrainRandomForestRegressorInput",
    "train_random_forest_regressor",
    "TrainGradientBoostingRegressorInput",
    "train_gradient_boosting_regressor",
    # Evaluation
    "EvaluateClassificationInput",
    "evaluate_classification_model",
    "EvaluateRegressionInput",
    "evaluate_regression_model",
]
