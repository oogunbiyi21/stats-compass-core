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


def _run(kind: str, *, features=None, plots=False, **config_overrides):
    state = DataFrameState()
    state.set_dataframe(_frame(kind == "classification"), "df", operation="parity")

    settings = {"model_type": "random_forest", "generate_plots": plots, **config_overrides}

    if kind == "classification":
        result = run_classification(state, RunClassificationInput(
            dataframe_name="df", target_column="target", feature_columns=features,
            config=ClassificationConfig(**settings),
        ))
    else:
        result = run_regression(state, RunRegressionInput(
            dataframe_name="df", target_column="target", feature_columns=features,
            config=RegressionConfig(**settings),
        ))
    return state, result


def _step(result, name):
    return next((s for s in result.steps if s.step_name == name), None)


# Everything a supervised workflow emits that is not a chart. Anything else in
# the step list is a chart step, under either naming convention — classification
# names them for the chart type, regression prefixes them "plot_" (D6).
NON_CHART_STEPS = {
    "bin_rare_categories", "target_encode", "train_model", "evaluate_model",
}


def _chart_steps(result):
    return [s for s in result.steps if s.step_name not in NON_CHART_STEPS]


def _shared_prefix(result):
    """(index, name) pairs up to and including evaluate_model.

    The plot block legitimately differs in length between the two workflows, so
    parity on step numbering is asserted over the part that should not differ.
    """
    prefix = []
    for step in result.steps:
        prefix.append((step.step_index, step.step_name))
        if step.step_name == "evaluate_model":
            break
    return prefix


BOTH = pytest.mark.parametrize("kind", ["classification", "regression"])

# Scenarios both workflows must handle identically. Named so a failure report
# says which configuration broke.
SCENARIOS = {
    "defaults": {},
    "gradient_boosting": {"model_type": "gradient_boosting"},
    "no_feature_engineering": {"feature_engineering": None},
    "drop_columns": {"drop_columns": ["category_b"]},
}
EVERY_SCENARIO = pytest.mark.parametrize("scenario", list(SCENARIOS))


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


# =============================================================================
# Below here: added to cover what the refactor can break.
#
# The suite above asserts behaviour that three shipped defects violated. It did
# not assert step numbering, artifact ordering, or result shape — which are
# exactly the three things merging the two workflows behind one skeleton, and
# handing step numbering to an accumulator, are most likely to disturb. Without
# these, "the parity suite still passes" would have meant very little.
# =============================================================================


class TestStepNumbering:
    """step_index is 1-based and contiguous. An accumulator that owns the
    counter must not skip, repeat or reorder — users read these numbers, and
    ARCHITECTURE.md documents them as 1-indexed."""

    @BOTH
    @EVERY_SCENARIO
    def test_indices_are_contiguous_from_one(self, kind, scenario):
        _, result = _run(kind, **SCENARIOS[scenario])
        indices = [s.step_index for s in result.steps]
        assert indices == list(range(1, len(indices) + 1)), (
            f"{kind}/{scenario} numbered its steps {indices}"
        )

    @BOTH
    def test_plot_steps_continue_the_sequence(self, kind):
        """Plots are numbered by the same counter, not restarted."""
        _, result = _run(kind, plots=True)
        indices = [s.step_index for s in result.steps]
        assert indices == list(range(1, len(indices) + 1))

    @EVERY_SCENARIO
    def test_both_workflows_number_the_shared_steps_identically(self, scenario):
        _, classification = _run("classification", **SCENARIOS[scenario])
        _, regression = _run("regression", **SCENARIOS[scenario])
        assert _shared_prefix(classification) == _shared_prefix(regression)


class TestArtifactOrdering:
    """dataframes_created and models_created feed the result the assistant
    reads. Order is meaningful: the predictions frame is the one to carry
    forward, and it is identified by position."""

    @BOTH
    @EVERY_SCENARIO
    def test_predictions_frame_is_recorded_last(self, kind, scenario):
        _, result = _run(kind, **SCENARIOS[scenario])
        created = result.artifacts.dataframes_created
        assert created, f"{kind}/{scenario} recorded no dataframes"
        assert created[-1] == result.artifacts.final_dataframe

    @BOTH
    @EVERY_SCENARIO
    def test_created_frames_all_exist_in_state(self, kind, scenario):
        state, result = _run(kind, **SCENARIOS[scenario])
        missing = [
            name for name in result.artifacts.dataframes_created
            if state.get_dataframe(name) is None
        ]
        assert not missing, f"{kind}/{scenario} claims frames that do not exist: {missing}"

    @EVERY_SCENARIO
    def test_both_workflows_create_the_same_frames_in_the_same_order(self, scenario):
        _, classification = _run("classification", **SCENARIOS[scenario])
        _, regression = _run("regression", **SCENARIOS[scenario])
        assert (
            classification.artifacts.dataframes_created
            == regression.artifacts.dataframes_created
        )

    @BOTH
    def test_the_recorded_model_is_the_one_that_was_trained(self, kind):
        _, result = _run(kind)
        assert result.artifacts.models_created == [
            _step(result, "train_model").result["model_id"]
        ]

    @BOTH
    def test_feature_engineering_frames_precede_the_predictions_frame(self, kind):
        _, result = _run(kind)
        created = result.artifacts.dataframes_created
        assert created == ["df_binned", "df_encoded", "df_encoded_predictions"], created


class TestResultShape:
    """The fields the MCP summariser forwards."""

    @BOTH
    @EVERY_SCENARIO
    def test_workflow_reports_its_own_name(self, kind, scenario):
        _, result = _run(kind, **SCENARIOS[scenario])
        assert result.workflow_name == f"run_{kind}"

    @BOTH
    @EVERY_SCENARIO
    def test_a_clean_run_reports_success(self, kind, scenario):
        _, result = _run(kind, **SCENARIOS[scenario])
        assert result.status == "success", [
            (s.step_name, s.error) for s in result.steps if s.status == "failed"
        ]

    @BOTH
    def test_notes_point_at_the_predictions_frame(self, kind):
        _, result = _run(kind)
        assert any(
            result.artifacts.final_dataframe in note for note in result.notes
        ), result.notes

    @BOTH
    def test_chart_count_matches_the_chart_steps_that_succeeded(self, kind):
        _, result = _run(kind, plots=True)
        succeeded = [s for s in _chart_steps(result) if s.status == "success"]
        assert result.artifacts.charts_generated == len(succeeded)


class TestFailuresExplainThemselves:
    """D1. A workflow that fails outright should say why and what to try.

    Regression populated error_summary and suggestion; classification left both
    None, so an assistant handed a failed classification had nothing to relay —
    the MCP summariser forwards both fields. The shared skeleton now assembles
    them for both.
    """

    @BOTH
    def test_a_failed_run_carries_a_diagnosis(self, kind):
        _, result = _run(kind, features=["no_such_column"], feature_engineering=None)
        assert result.status == "failed", "scenario stopped reproducing a failure"
        assert result.error_summary, f"{kind} failed without an error_summary"
        assert result.suggestion, f"{kind} failed without a suggestion"


class TestConfiguredPlotsAreAccountedFor:
    """D2. RegressionConfig.plots defaults to residuals, predicted_vs_actual and
    feature_importance, but only feature_importance has a registered tool. The
    other two used to be dropped by a bare `continue` — no step, no skip, no
    note — so the config advertised charts the workflow could not produce and
    said nothing when they failed to arrive.

    Every configured plot now produces a step in both workflows, successful or
    skipped.
    """

    @BOTH
    def test_every_configured_plot_produces_a_step(self, kind):
        _, result = _run(kind, plots=True)
        configured = (
            ["confusion_matrix", "roc", "precision_recall", "feature_importance"]
            if kind == "classification"
            else ["residuals", "predicted_vs_actual", "feature_importance"]
        )
        assert len(_chart_steps(result)) == len(configured), (
            f"{kind} configured {len(configured)} plots but recorded "
            f"{len(_chart_steps(result))} steps: "
            f"{[s.step_name for s in _chart_steps(result)]}"
        )


class TestBothWorkflowsScopeFeatureEngineering:
    """T6.6: encode only what was declared, and say so when nothing was."""

    @BOTH
    def test_undeclared_categorical_is_not_encoded(self, kind):
        _, result = _run(kind, features=["category_a", "numeric"])
        encode = _step(result, "target_encode")
        assert encode.status == "success", encode.error
        assert encode.result["original_columns"] == ["category_a"]

    @BOTH
    def test_inferred_features_surface_as_a_workflow_warning(self, kind):
        _, result = _run(kind)
        codes = [w.code for w in result.warnings]
        assert "FEATURES_INFERRED" in codes

    @BOTH
    def test_declared_features_raise_no_inferred_warning(self, kind):
        _, result = _run(kind, features=["category_a", "category_b", "numeric"])
        assert "FEATURES_INFERRED" not in [w.code for w in result.warnings]

    @BOTH
    def test_parity_frame_raises_no_leakage(self, kind):
        """The shared fixture is honest data; a detector that fires here is noise."""
        _, result = _run(kind)
        assert "LEAKAGE_SUSPECTED" not in [w.code for w in result.warnings]


def _run_with_customer_id(kind: str, *, features=None, **config_overrides):
    """The parity frame plus a string ID unique to every row."""
    frame = _frame(kind == "classification")
    frame.insert(0, "customer_id", [f"C{i:05d}" for i in range(len(frame))])
    state = DataFrameState()
    state.set_dataframe(frame, "df", operation="parity")
    settings = {
        "model_type": "random_forest", "generate_plots": False, **config_overrides
    }
    if kind == "classification":
        return run_classification(state, RunClassificationInput(
            dataframe_name="df", target_column="target", feature_columns=features,
            config=ClassificationConfig(**settings),
        ))
    return run_regression(state, RunRegressionInput(
        dataframe_name="df", target_column="target", feature_columns=features,
        config=RegressionConfig(**settings),
    ))


class TestBothWorkflowsKeepIdColumnsOutOfTheModel:
    """T6.7: an ID column is rejected with a named warning and never reaches the
    model, whether or not binning is enabled and whether or not it was declared.
    """

    def _assert_excluded(self, result):
        train = _step(result, "train_model")
        assert train.status == "success", train.error
        features = train.result["feature_columns"]
        assert not any(f.startswith("customer_id") for f in features), features
        flagged = [w for w in result.warnings if w.code == "HIGH_CARDINALITY"]
        assert [w.columns for w in flagged] == [["customer_id"]]

    @BOTH
    def test_undeclared(self, kind):
        self._assert_excluded(_run_with_customer_id(kind))

    @BOTH
    def test_binning_disabled(self, kind):
        from stats_compass_core.workflows.configs import FeatureEngineeringConfig

        self._assert_excluded(_run_with_customer_id(
            kind,
            feature_engineering=FeatureEngineeringConfig(bin_rare_categories=False),
        ))

    @BOTH
    def test_declared(self, kind):
        self._assert_excluded(_run_with_customer_id(
            kind, features=["customer_id", "category_a", "numeric"]
        ))

    @BOTH
    def test_declaring_only_an_id_fails_rather_than_inferring(self, kind):
        """An emptied feature list must not fall back to every numeric column."""
        result = _run_with_customer_id(kind, features=["customer_id"])
        train = _step(result, "train_model")
        assert train.status == "failed"
        assert "customer_id" in train.error
        assert result.artifacts.models_created == []
