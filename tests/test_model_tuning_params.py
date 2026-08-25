"""Hyperparameters that let a user do something about a bad model.

The workflows already forwarded a `hyperparameters` dict, but it was filtered
against the trainer schemas — and those declared only n_estimators, test_size
and random_state. So on discovering a forest with a train score of 1.00 and a
test score of 0.69, there was nothing to turn: no depth limit, no leaf minimum,
and no class weighting for an imbalanced target.
"""

import numpy as np
import pandas as pd
import pytest

from stats_compass_core.state import DataFrameState
from stats_compass_core.workflows import ClassificationConfig, run_classification
from stats_compass_core.workflows.classification import RunClassificationInput


@pytest.fixture
def imbalanced_noise():
    """20% positives and no real signal — the shape of the churn dataset."""
    state = DataFrameState()
    rng = np.random.default_rng(0)
    n = 400
    state.set_dataframe(pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "f3": rng.normal(size=n),
        "churned": rng.choice(["Yes", "No"], size=n, p=[0.2, 0.8]),
    }), "churn", operation="test_fixture")
    return state


def _train_step(result):
    return next(s for s in result.steps if s.step_name == "train_model")


def _run(state, **hyperparameters):
    return run_classification(
        state=state,
        params=RunClassificationInput(
            dataframe_name="churn",
            target_column="churned",
            config=ClassificationConfig(
                model_type="random_forest",
                generate_plots=False,
                hyperparameters=hyperparameters or None,
            ),
        ),
    )


class TestDepthLimitReachesTheModel:
    def test_unconstrained_forest_memorises(self, imbalanced_noise):
        """Baseline: this is the behaviour users were stuck with."""
        train = _train_step(_run(imbalanced_noise))
        assert train.result["metrics"]["train_score"] == pytest.approx(1.0, abs=0.02), (
            "an unconstrained forest on noise should memorise completely"
        )

    def test_max_depth_stops_the_memorising(self, imbalanced_noise):
        train = _train_step(_run(imbalanced_noise, max_depth=3))
        assert train.result["metrics"]["train_score"] < 0.95, (
            "max_depth did not reach the model — check it is declared on the "
            "trainer schema, since _build_training_params filters unknown keys"
        )

    def test_min_samples_leaf_stops_the_memorising(self, imbalanced_noise):
        train = _train_step(_run(imbalanced_noise, min_samples_leaf=20))
        assert train.result["metrics"]["train_score"] < 0.95


class TestClassWeightReachesTheModel:
    def test_balanced_changes_the_predictions(self, imbalanced_noise):
        """Without class weighting, a model on a 20% positive rate scores well by
        predicting the majority for everyone and catching nobody."""
        plain = _run(imbalanced_noise, max_depth=3)
        state2 = DataFrameState()
        state2.set_dataframe(
            imbalanced_noise.get_dataframe("churn").copy(), "churn", operation="t"
        )
        weighted = _run(state2, max_depth=3, class_weight="balanced")

        def positive_rate(state, name):
            df = state.get_dataframe(name)
            return (df["pred_churned"] == "Yes").mean()

        plain_rate = positive_rate(imbalanced_noise, "churn_encoded_predictions"
                                   if "churn_encoded_predictions" in
                                   [d.name for d in imbalanced_noise.list_dataframes()]
                                   else "churn_predictions")
        weighted_rate = positive_rate(state2, "churn_encoded_predictions"
                                      if "churn_encoded_predictions" in
                                      [d.name for d in state2.list_dataframes()]
                                      else "churn_predictions")

        assert weighted_rate > plain_rate, (
            "class_weight='balanced' should make the model predict the minority "
            "class more often; if the rates match it never reached the model"
        )


class TestUnknownHyperparametersAreStillIgnored:
    def test_nonsense_key_does_not_crash(self, imbalanced_noise):
        """The passthrough filters to schema fields, which must stay true."""
        train = _train_step(_run(imbalanced_noise, not_a_real_parameter=123))
        assert train.status == "success"
