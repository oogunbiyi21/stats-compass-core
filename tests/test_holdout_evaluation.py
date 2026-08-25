"""Reported model metrics must describe held-out data, not memorised data.

create_predictions_dataframe predicts every row and marks each one train or test
in a `<target>_split` column. The evaluators ignored that column and scored the
whole frame, so roughly 80% of every reported metric was the model being graded
on rows it had trained on.

That failure is silent and flatters: a random forest reporting 94% accuracy when
its true holdout performance was 69%, on data whose majority class alone gets
80%. For a tool whose purpose is to stop people fooling themselves with
statistics, producing exactly that is the worst available bug, so these tests are
deliberately built around a model guaranteed to memorise.
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


@pytest.fixture
def unlearnable_classification():
    """Pure noise: no feature carries signal about the label.

    An unconstrained forest memorises the training rows perfectly and cannot do
    better than chance on held-out rows, so the gap between an honest metric and
    an inflated one is unmissable.
    """
    state = DataFrameState()
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "f3": rng.normal(size=n),
        "label": rng.integers(0, 2, size=n),
    })
    state.set_dataframe(df, "noise", operation="test_fixture")
    return state


@pytest.fixture
def unlearnable_regression():
    state = DataFrameState()
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "f3": rng.normal(size=n),
        "target": rng.normal(size=n) * 10,
    })
    state.set_dataframe(df, "noise", operation="test_fixture")
    return state


def _step(result, name):
    for step in result.steps:
        if step.step_name == name:
            return step
    raise AssertionError(f"no {name!r} step in {[s.step_name for s in result.steps]}")


def _recompute(state, predictions_df_name, target, prediction, split):
    """Score the predictions frame by hand, split by split."""
    df = state.get_dataframe(predictions_df_name)
    rows = df[df[f"{target}_split"] == split].dropna(subset=[target, prediction])
    return float((rows[target] == rows[prediction]).mean()), len(rows)


class TestClassificationReportsHoldout:
    def test_reported_accuracy_is_not_the_memorised_score(self, unlearnable_classification):
        """The headline number must not be dominated by training rows."""
        result = run_classification(
            state=unlearnable_classification,
            params=RunClassificationInput(
                dataframe_name="noise",
                target_column="label",
                config=ClassificationConfig(model_type="random_forest", generate_plots=False),
            ),
        )
        reported = _step(result, "evaluate_model").result["accuracy"]

        train_acc, n_train = _recompute(
            unlearnable_classification, "noise_predictions", "label", "pred_label", "train"
        )
        test_acc, n_test = _recompute(
            unlearnable_classification, "noise_predictions", "label", "pred_label", "test"
        )

        assert n_train > n_test, "fixture should produce a majority-train split"
        # The bug: reported tracked the train score because train rows dominate.
        assert reported == pytest.approx(test_acc, abs=1e-6), (
            f"reported {reported:.3f} should equal the holdout {test_acc:.3f}, "
            f"not the memorised {train_acc:.3f}"
        )

    def test_evaluated_row_count_is_the_holdout(self, unlearnable_classification):
        result = run_classification(
            state=unlearnable_classification,
            params=RunClassificationInput(
                dataframe_name="noise",
                target_column="label",
                config=ClassificationConfig(model_type="random_forest", generate_plots=False),
            ),
        )
        evaluation = _step(result, "evaluate_model").result
        _, n_test = _recompute(
            unlearnable_classification, "noise_predictions", "label", "pred_label", "test"
        )
        assert evaluation["n_samples"] == n_test

    def test_result_states_what_it_evaluated(self, unlearnable_classification):
        """A metric that does not say what it describes is how this went unnoticed."""
        result = run_classification(
            state=unlearnable_classification,
            params=RunClassificationInput(
                dataframe_name="noise",
                target_column="label",
                config=ClassificationConfig(model_type="random_forest", generate_plots=False),
            ),
        )
        assert _step(result, "evaluate_model").result["evaluated_on"] == "test"

    def test_train_metrics_are_reported_too(self, unlearnable_classification):
        """The train/test gap is the signal that a model has memorised. Hiding it
        is what let a 100% training score pass without comment."""
        result = run_classification(
            state=unlearnable_classification,
            params=RunClassificationInput(
                dataframe_name="noise",
                target_column="label",
                config=ClassificationConfig(model_type="random_forest", generate_plots=False),
            ),
        )
        evaluation = _step(result, "evaluate_model").result
        assert evaluation["train_metrics"] is not None
        assert evaluation["train_metrics"]["accuracy"] > evaluation["accuracy"], (
            "an unconstrained forest on noise should score far better on train"
        )


class TestRegressionReportsHoldout:
    def test_reported_r2_is_not_the_memorised_score(self, unlearnable_regression):
        result = run_regression(
            state=unlearnable_regression,
            params=RunRegressionInput(
                dataframe_name="noise",
                target_column="target",
                config=RegressionConfig(model_type="random_forest", generate_plots=False),
            ),
        )
        evaluation = _step(result, "evaluate_model").result

        df = unlearnable_regression.get_dataframe("noise_predictions")
        test_rows = df[df["target_split"] == "test"]
        assert evaluation["n_samples"] == len(test_rows)
        # Noise cannot be predicted; a memorised score would look far better.
        assert evaluation["r2"] < 0.5, (
            f"r2 of {evaluation['r2']:.3f} on pure noise indicates training rows "
            "are being scored"
        )

    def test_regression_states_what_it_evaluated(self, unlearnable_regression):
        result = run_regression(
            state=unlearnable_regression,
            params=RunRegressionInput(
                dataframe_name="noise",
                target_column="target",
                config=RegressionConfig(model_type="random_forest", generate_plots=False),
            ),
        )
        assert _step(result, "evaluate_model").result["evaluated_on"] == "test"


@pytest.fixture
def categorical_classification():
    """Categorical features, so target encoding runs and renames columns."""
    state = DataFrameState()
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "plan_type": rng.choice(["Basic", "Pro", "Enterprise"], size=n),
        "region": rng.choice(["North", "South", "East", "West"], size=n),
        "tenure_months": rng.integers(1, 25, size=n),
        "churned": rng.choice(["Yes", "No"], size=n, p=[0.2, 0.8]),
    })
    state.set_dataframe(df, "churn", operation="test_fixture")
    return state


class TestFeatureColumnsSurviveEncoding:
    """Target encoding replaces plan_type with plan_type_encoded and drops the
    original, so feature names supplied by the caller stopped existing before
    training ran — failing with "Feature columns not found" immediately after
    the encoding step reported success.
    """

    def test_named_categorical_features_still_train(self, categorical_classification):
        result = run_classification(
            state=categorical_classification,
            params=RunClassificationInput(
                dataframe_name="churn",
                target_column="churned",
                feature_columns=["plan_type", "region", "tenure_months"],
                config=ClassificationConfig(model_type="random_forest", generate_plots=False),
            ),
        )
        train_step = _step(result, "train_model")
        assert train_step.status == "success", train_step.error

    def test_untouched_features_are_not_renamed(self, categorical_classification):
        """Only encoded columns should be translated; numerics pass through."""
        from stats_compass_core.workflows.feature_engineering import map_feature_columns

        mapping = {"plan_type": "plan_type_encoded", "region": "region_encoded"}
        assert map_feature_columns(
            ["plan_type", "tenure_months", "region"], mapping
        ) == ["plan_type_encoded", "tenure_months", "region_encoded"]

    def test_no_mapping_leaves_features_alone(self):
        from stats_compass_core.workflows.feature_engineering import map_feature_columns

        assert map_feature_columns(["a", "b"], {}) == ["a", "b"]
        assert map_feature_columns(None, {"a": "a_encoded"}) is None


class TestTargetEncodingUsesCrossFitting:
    """sklearn's fit_transform cross-fits so a row's own target cannot influence
    its own encoding. Replacing it with fit().transform() would look equivalent
    and silently reintroduce the leakage it exists to prevent.
    """

    def test_encoder_is_cross_fitted_not_naively_fitted(self):
        import inspect
        from stats_compass_core.transforms import mean_target_encoding as mte

        source = inspect.getsource(mte)
        assert "fit_transform" in source, (
            "target encoding must use fit_transform, which cross-fits; "
            "fit().transform() lets each row's own target set its encoding"
        )
        assert ".fit(" not in source.replace("fit_transform", ""), (
            "a plain fit() on the full data would defeat the cross-fitting"
        )

    def test_encoding_differs_from_naive_group_means(self):
        """Cross-fitted encodings should not equal the raw category means; if
        they do, no cross-fitting happened."""
        from sklearn.preprocessing import TargetEncoder

        rng = np.random.default_rng(0)
        n = 200
        cats = rng.choice(["a", "b", "c"], size=n)
        y = rng.integers(0, 2, size=n).astype(float)
        X = pd.DataFrame({"cat": cats})

        cross_fitted = TargetEncoder(cv=5, shuffle=True, random_state=42).fit_transform(X, y)
        naive = pd.DataFrame({"cat": cats, "y": y}).groupby("cat")["y"].transform("mean")

        assert not np.allclose(cross_fitted.ravel(), naive.to_numpy()), (
            "encodings identical to raw group means indicate no cross-fitting"
        )


class TestChartsDescribeTheSameRowsAsTheMetrics:
    """Fixing the metric while leaving the charts on the full dataset was worse
    than fixing neither: a confusion matrix over memorised rows sat next to a
    correctly held-out accuracy, and the chart is what gets screenshotted.

    In the reported case the matrix summed to 494 rows and implied 93.7%
    accuracy, while the honest figure was 68.7% on 99 rows.
    """

    def test_plots_receive_only_holdout_rows(self, unlearnable_classification):
        result = run_classification(
            state=unlearnable_classification,
            params=RunClassificationInput(
                dataframe_name="noise",
                target_column="label",
                config=ClassificationConfig(
                    model_type="random_forest",
                    generate_plots=True,
                    plots=["confusion_matrix"],
                ),
            ),
        )
        state = unlearnable_classification
        full = state.get_dataframe("noise_predictions")
        holdout = state.get_dataframe("noise_predictions_holdout")
        assert len(holdout) < len(full), "the plotted frame must exclude training rows"

        # The matrix itself is the evidence: summing to the full row count means
        # the chart described training data whatever frame name it reported.
        plot = _step(result, "confusion_matrix").result
        plotted_rows = sum(sum(row) for row in plot["data"]["confusion_matrix"])
        assert plotted_rows == len(holdout), (
            f"confusion matrix covers {plotted_rows} rows but the holdout is "
            f"{len(holdout)}; the chart is describing memorised data"
        )

        evaluation = _step(result, "evaluate_model").result
        assert evaluation["n_samples"] == plotted_rows, (
            "charts and metrics must describe the same rows, or they contradict "
            "each other in the same response"
        )

    def test_holdout_frame_is_not_built_without_a_split(self, unlearnable_classification):
        """With nothing to filter on, plotting the whole frame is correct — but
        the metric must then also say it covered everything."""
        from stats_compass_core.workflows.utils import build_holdout_predictions

        state = unlearnable_classification
        df = pd.DataFrame({"y": [0, 1, 0], "pred_y": [0, 1, 1]})
        state.set_dataframe(df, "no_split", operation="test_fixture")

        name, evaluated_on = build_holdout_predictions(state, "no_split", "y")
        assert name == "no_split"
        assert evaluated_on == "all"
