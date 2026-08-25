"""Classification and regression must stay in step with each other.

These two workflows are near-duplicates maintained in parallel: same shape, same
feature-engineering step, same evaluation step, same plotting block. Every fix
has to be applied twice and nothing enforced that it was.

That is not hypothetical. Three separate defects came from exactly this:

  - a feature-column fix applied to classification (keyword argument) silently
    missed regression (positional), so regression kept asking the trainer for
    columns that encoding had already renamed;
  - the holdout fix was applied to both evaluators but only classification's
    plots, so its charts described memorised data;
  - plot_df_name was referenced in regression without being assigned, which was
    dead code only because the branch reaching it was unregistered.

These tests exist so the next divergence fails here rather than in someone's
analysis.
"""

import numpy as np
import pandas as pd
import pytest

from stats_compass_core.state import DataFrameState
from stats_compass_core.workflows import (
    ClassificationConfig,
    RegressionConfig,
    run_classification,
    run_regression,
)
from stats_compass_core.workflows.classification import RunClassificationInput
from stats_compass_core.workflows.regression import RunRegressionInput


def _frame(target_is_categorical: bool, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    target = (
        rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7])
        if target_is_categorical
        else rng.integers(100_000, 900_000, size=n)
    )
    return pd.DataFrame({
        "category_a": rng.choice(["x", "y", "z"], size=n),
        "category_b": rng.choice(["p", "q"], size=n),
        "numeric": rng.normal(size=n),
        "target": target,
    })


def _run(kind: str, *, features=None, plots=False):
    state = DataFrameState()
    state.set_dataframe(_frame(kind == "classification"), "df", operation="parity")

    if kind == "classification":
        result = run_classification(state, RunClassificationInput(
            dataframe_name="df", target_column="target", feature_columns=features,
            config=ClassificationConfig(model_type="random_forest", generate_plots=plots),
        ))
    else:
        result = run_regression(state, RunRegressionInput(
            dataframe_name="df", target_column="target", feature_columns=features,
            config=RegressionConfig(model_type="random_forest", generate_plots=plots),
        ))
    return state, result


def _step(result, name):
    return next((s for s in result.steps if s.step_name == name), None)


BOTH = pytest.mark.parametrize("kind", ["classification", "regression"])


class TestBothWorkflowsTranslateFeatureColumns:
    @BOTH
    def test_named_categoricals_train_after_encoding(self, kind):
        """The defect that shipped: fixed in one workflow, missed in the other."""
        _, result = _run(kind, features=["category_a", "category_b", "numeric"])
        train = _step(result, "train_model")
        assert train.status == "success", train.error

    @BOTH
    def test_trainer_receives_the_encoded_names(self, kind):
        _, result = _run(kind, features=["category_a", "category_b", "numeric"])
        trained_on = set(_step(result, "train_model").result["feature_columns"])
        assert trained_on == {"category_a_encoded", "category_b_encoded", "numeric"}


class TestBothWorkflowsReportTheHoldout:
    @BOTH
    def test_metrics_describe_the_test_split(self, kind):
        _, result = _run(kind)
        evaluation = _step(result, "evaluate_model").result
        assert evaluation["evaluated_on"] == "test"

    @BOTH
    def test_train_metrics_are_included(self, kind):
        _, result = _run(kind)
        assert _step(result, "evaluate_model").result["train_metrics"] is not None

    @BOTH
    def test_evaluated_rows_are_fewer_than_the_dataset(self, kind):
        state, result = _run(kind)
        evaluation = _step(result, "evaluate_model").result
        assert evaluation["n_samples"] < len(state.get_dataframe("df"))


class TestBothWorkflowsPlotWithoutBlowingUp:
    @BOTH
    def test_plot_generation_does_not_raise(self, kind):
        """regression referenced plot_df_name without assigning it — harmless
        only because the branch reaching it was never registered."""
        _, result = _run(kind, plots=True)
        failed = [
            s for s in result.steps
            if s.step_name.startswith(("plot_", "confusion", "roc", "precision"))
            and s.status == "failed"
        ]
        assert not failed, [(s.step_name, s.error) for s in failed]

    @BOTH
    def test_any_plot_reading_a_frame_reads_the_holdout(self, kind):
        """Asserting a holdout frame merely *exists* is not enough — it is built
        either way, so a plot ignoring it still passes. The evidence is which
        frame the plot reports having used."""
        _, result = _run(kind, plots=True)

        # Charts that score rows. feature_importance is excluded on purpose:
        # it describes the model's learned structure and reports the original
        # source frame, so it is neither train nor test data.
        row_scoring = {
            "confusion_matrix", "roc", "roc_curve", "precision_recall",
            "precision_recall_curve", "residuals", "predicted_vs_actual",
        }
        read_frames = [
            s.result["dataframe_name"]
            for s in result.steps
            if s.status == "success" and isinstance(s.result, dict)
            and s.result.get("chart_type") in row_scoring
            and s.result.get("dataframe_name")
        ]
        offenders = [f for f in read_frames if not f.endswith("_holdout")]
        assert not offenders, (
            f"{kind} plotted {offenders}, which include training rows — the "
            "chart would contradict the accuracy printed beside it"
        )
