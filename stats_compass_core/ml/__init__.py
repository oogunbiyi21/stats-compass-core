"""
Machine learning tools.

This module provides atomic ML training tools following the single-responsibility
principle. Each algorithm is exposed as a separate tool for better MCP compatibility
and easier testing/debugging.

Available tools (one per algorithm):
- train_logistic_regression: Train logistic regression classifier
- train_random_forest_classifier: Train random forest classifier
- train_gradient_boosting_classifier: Train gradient boosting classifier
- train_linear_regression: Train linear regression model
- train_random_forest_regressor: Train random forest regressor
- train_gradient_boosting_regressor: Train gradient boosting regressor

Legacy files (deprecated, kept for backward compatibility):
- _deprecated_train_classifier.py: Old multi-algorithm classifier
- _deprecated_train_regressor.py: Old multi-algorithm regressor

Note: These tools are automatically discovered by the registry.
Import scikit-learn separately with: pip install stats-compass-core[ml]
"""
