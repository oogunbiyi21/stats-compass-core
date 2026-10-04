"""Leakage must not pass silently when nobody is checking the output.

The reproduced failure: a churn frame with a cancellation_reason column that is
filled in only for customers who churned. Feature engineering encoded every
non-target categorical, the trainer fell back to "all numeric except target",
and the model scored 100% on the holdout with no warning. Excluding the target
is not the fix; the target was already excluded. The column that leaks is a
different column that was only ever written after the outcome happened.
"""

import numpy as np
import pandas as pd
import pytest

from stats_compass_core.ml.supervised.train_linear_regression import (
    TrainLinearRegressionInput,
    train_linear_regression,
)
from stats_compass_core.ml.supervised.train_random_forest_classifier import (
    TrainRandomForestClassifierInput,
    train_random_forest_classifier,
)
from stats_compass_core.state import DataFrameState
from stats_compass_core.workflows import ClassificationConfig, run_classification
from stats_compass_core.workflows.classification import RunClassificationInput


def _churn_frame(n: int = 400, with_leak: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    churn = rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7])
    frame = {
        "tenure_months": rng.integers(1, 60, n),
        "monthly_spend": rng.normal(50, 15, n).round(2),
        "plan": rng.choice(["basic", "pro", "team"], n),
    }
    if with_leak:
        # Recorded at cancellation, so it exists only for churners.
        frame["cancellation_reason"] = np.where(
            churn == "Yes", rng.choice(["price", "service", "moved"], n), None
        )
    frame["churn"] = churn
    return pd.DataFrame(frame)


def _codes(warnings) -> list[str]:
    return [w["code"] if isinstance(w, dict) else w.code for w in warnings]


def _with_code(warnings, code):
    return [w for w in warnings if _codes([w]) == [code]]


def _classify(frame, features=None):
    state = DataFrameState()
    state.set_dataframe(frame, "churn", operation="test")
    result = run_classification(state, RunClassificationInput(
        dataframe_name="churn",
        target_column="churn",
        feature_columns=features,
        config=ClassificationConfig(generate_plots=False, save_model=False),
    ))
    return state, result


def _step(result, name):
    return next(s for s in result.steps if s.step_name == name)


class TestChurnFrameAcceptance:
    """T6.6 AC: no longer 100% holdout accuracy *silently*."""

    def test_undeclared_features_carry_leakage_suspected(self):
        _, result = _classify(_churn_frame())
        leaks = _with_code(result.warnings, "LEAKAGE_SUSPECTED")
        assert leaks, f"no LEAKAGE_SUSPECTED; warnings were {_codes(result.warnings)}"
        # Named as the user knows it, not as the encoder renamed it.
        assert any("cancellation_reason" in w.columns for w in leaks)

    def test_leakage_is_in_the_op_log(self):
        state, _ = _classify(_churn_frame())
        logged = [
            h for h in state.get_history()
            if h.operation == "warning" and h.details.get("code") == "LEAKAGE_SUSPECTED"
        ]
        assert logged

    def test_trainer_result_carries_the_warning_too(self):
        _, result = _classify(_churn_frame())
        train = _step(result, "train_model")
        assert "LEAKAGE_SUSPECTED" in _codes(train.result["warnings"])

    def test_declared_features_do_not_encode_the_undeclared_column(self):
        _, result = _classify(
            _churn_frame(), features=["tenure_months", "monthly_spend", "plan"]
        )
        encode = _step(result, "target_encode")
        assert encode.status == "success", encode.error
        assert "cancellation_reason" not in encode.result["original_columns"]
        assert "LEAKAGE_SUSPECTED" not in _codes(result.warnings)

    def test_inferred_features_are_named(self):
        """The fallback stays, but it no longer happens quietly."""
        _, result = _classify(_churn_frame(with_leak=False))
        inferred = _with_code(result.warnings, "FEATURES_INFERRED")
        assert inferred
        assert "tenure_months" in inferred[0].columns

    def test_declared_features_are_not_reported_as_inferred(self):
        _, result = _classify(
            _churn_frame(with_leak=False), features=["tenure_months", "plan"]
        )
        assert "FEATURES_INFERRED" not in _codes(result.warnings)


class TestNoFalseAlarms:
    def test_clean_churn_frame_raises_no_leakage(self):
        _, result = _classify(_churn_frame(with_leak=False))
        assert "LEAKAGE_SUSPECTED" not in _codes(result.warnings)

    def test_unique_id_column_is_not_perfect_on_its_own(self):
        """A unique value per row is trivially 'predictive' to an unbounded tree.

        The detector caps the tree at one leaf per class, so an ID column with
        no relationship to the target must not fire.
        """
        frame = _churn_frame(with_leak=False)
        frame["row_id"] = np.arange(len(frame))
        state = DataFrameState()
        state.set_dataframe(frame, "churn", operation="test")
        result = train_random_forest_classifier(state, TrainRandomForestClassifierInput(
            dataframe_name="churn", target_column="churn",
            feature_columns=["tenure_months", "monthly_spend", "row_id"],
        ))
        assert "LEAKAGE_SUSPECTED" not in _codes(result.warnings)


class TestTrainerDetector:
    """The detector lives in the trainers, so a direct call is covered too."""

    def test_null_pattern_that_tracks_the_target(self):
        frame = _churn_frame(with_leak=False)
        # Days from signup to cancellation: only exists once someone has left.
        days = np.random.default_rng(2).integers(1, 400, len(frame))
        frame["days_to_cancel"] = np.where(frame["churn"] == "Yes", days, np.nan)
        state = DataFrameState()
        state.set_dataframe(frame, "churn", operation="test")
        result = train_random_forest_classifier(state, TrainRandomForestClassifierInput(
            dataframe_name="churn", target_column="churn",
            feature_columns=["tenure_months", "monthly_spend", "days_to_cancel"],
        ))
        leaks = _with_code(result.warnings, "LEAKAGE_SUSPECTED")
        assert leaks
        assert leaks[0].columns == ["days_to_cancel"]

    def test_regression_feature_that_is_the_target_rescaled(self):
        rng = np.random.default_rng(3)
        n = 200
        frame = pd.DataFrame({"x": rng.normal(size=n), "noise": rng.normal(size=n)})
        frame["revenue"] = 3 * frame["x"] + rng.normal(size=n)
        frame["revenue_in_pence"] = frame["revenue"] * 100
        state = DataFrameState()
        state.set_dataframe(frame, "sales", operation="test")
        result = train_linear_regression(state, TrainLinearRegressionInput(
            dataframe_name="sales", target_column="revenue",
            feature_columns=["x", "noise", "revenue_in_pence"],
        ))
        leaks = _with_code(result.warnings, "LEAKAGE_SUSPECTED")
        assert [w.columns for w in leaks] == [["revenue_in_pence"]]

    @pytest.mark.parametrize("features", [None, ["x", "noise"]])
    def test_honest_regression_is_quiet(self, features):
        rng = np.random.default_rng(4)
        n = 200
        frame = pd.DataFrame({"x": rng.normal(size=n), "noise": rng.normal(size=n)})
        frame["y"] = 3 * frame["x"] + rng.normal(size=n)
        state = DataFrameState()
        state.set_dataframe(frame, "d", operation="test")
        result = train_linear_regression(state, TrainLinearRegressionInput(
            dataframe_name="d", target_column="y", feature_columns=features,
        ))
        assert "LEAKAGE_SUSPECTED" not in _codes(result.warnings)
