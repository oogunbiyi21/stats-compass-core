"""Model files go through the same checks as every other write (security scan F4, F5, F7).

The trainers, fit_arima and the classification/regression workflows used to call
joblib.dump on a caller's path directly: no denylist, no overwrite protection, no
session confinement, and a ``save_path`` hyperparameter could set the path even
with ``save_model=False``. Now every model write goes through ``safe_save`` with
the session's write root, ``save_model=False`` writes nothing, and the keys the
workflow sets itself cannot be overridden through hyperparameters.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stats_compass_core.ml.supervised.train_linear_regression import (
    TrainLinearRegressionInput,
    train_linear_regression,
)
from stats_compass_core.state import DataFrameState
from stats_compass_core.utils.file_safety import FilePolicy, UnsafePathError
from stats_compass_core.workflows.classification import RunClassificationInput, run_classification
from stats_compass_core.workflows.configs import ClassificationConfig

PACKAGE = Path(__file__).resolve().parent.parent / "stats_compass_core"


@pytest.fixture
def dirs(tmp_path):
    out = {name: tmp_path / name for name in ("exports", "elsewhere")}
    for path in out.values():
        path.mkdir()
    return out


def _frame():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    return pd.DataFrame({
        "x": x,
        "z": rng.normal(size=200),
        "y": 2 * x + rng.normal(size=200),
        "churned": (x + rng.normal(size=200) > 0).astype(int),
    })


def _state(policy=None):
    state = DataFrameState(file_policy=policy or FilePolicy())
    state.set_dataframe(_frame(), "t", "test")
    return state


def _files(root):
    return sorted(p.name for p in Path(root).rglob("*") if p.is_file())


class TestTrainers:
    def test_save_path_lands_in_the_root(self, dirs):
        state = _state(FilePolicy(write_root=dirs["exports"]))
        train_linear_regression(state, TrainLinearRegressionInput(
            dataframe_name="t", target_column="y", feature_columns=["x"],
            save_path=str(dirs["elsewhere"] / "model.joblib"),
        ))
        assert _files(dirs["elsewhere"]) == []
        assert _files(dirs["exports"]) == ["model.joblib"]

    def test_a_source_file_is_never_written(self, dirs):
        state = _state()
        target = dirs["elsewhere"] / "server.py"
        with pytest.raises(UnsafePathError):
            train_linear_regression(state, TrainLinearRegressionInput(
                dataframe_name="t", target_column="y", feature_columns=["x"],
                save_path=str(target),
            ))
        assert not target.exists()

    def test_an_existing_file_is_not_overwritten(self, dirs):
        state = _state()
        target = dirs["elsewhere"] / "model.joblib"
        target.write_text("someone else's")
        train_linear_regression(state, TrainLinearRegressionInput(
            dataframe_name="t", target_column="y", feature_columns=["x"], save_path=str(target),
        ))
        assert target.read_text() == "someone else's"
        assert _files(dirs["elsewhere"]) == ["model.joblib", "model_1.joblib"]


class TestArima:
    def test_save_path_lands_in_the_root(self, dirs):
        pytest.importorskip("statsmodels")
        from stats_compass_core.ml.timeseries.arima import FitARIMAInput, fit_arima

        state = DataFrameState(file_policy=FilePolicy(write_root=dirs["exports"]))
        state.set_dataframe(
            pd.DataFrame({"v": np.sin(np.arange(60) / 4) + np.arange(60) * 0.1}), "ts", "test"
        )
        fit_arima(state, FitARIMAInput(
            dataframe_name="ts", target_column="v", p=1, d=0, q=0,
            save_path=str(dirs["elsewhere"] / "arima.joblib"),
        ))
        assert _files(dirs["elsewhere"]) == []
        assert _files(dirs["exports"]) == ["arima.joblib"]


def _classify(state, **config):
    result = run_classification(state, RunClassificationInput(
        dataframe_name="t", target_column="churned", feature_columns=["x", "z"],
        config=ClassificationConfig(model_type="logistic", generate_plots=False, **config),
    ))
    return next(s for s in result.steps if s.step_name == "train_model")


class TestWorkflows:
    def test_save_model_false_writes_nothing(self, dirs):
        state = _state()
        step = _classify(
            state, save_model=False,
            hyperparameters={"save_path": str(dirs["elsewhere"] / "planted.joblib")},
        )
        assert _files(dirs["elsewhere"]) == []
        assert step.status == "failed" and "save_path" in step.error

    def test_model_save_path_lands_in_the_root(self, dirs):
        state = _state(FilePolicy(write_root=dirs["exports"]))
        step = _classify(state, model_save_path=str(dirs["elsewhere"] / "clf.joblib"))
        assert step.status == "success"
        assert _files(dirs["elsewhere"]) == []
        assert _files(dirs["exports"]) == ["clf.joblib"]

    def test_the_default_path_lands_in_the_root(self, dirs):
        state = _state(FilePolicy(write_root=dirs["exports"]))
        _classify(state)
        assert len(_files(dirs["exports"])) == 1
        assert _files(dirs["exports"])[0].startswith("logistic_churned_")

    @pytest.mark.parametrize(
        "key", ["save_path", "target_column", "feature_columns", "dataframe_name"]
    )
    def test_hyperparameters_cannot_set_what_the_workflow_sets(self, key):
        step = _classify(_state(), hyperparameters={key: "x"})
        assert step.status == "failed" and key in step.error

    def test_model_hyperparameters_still_reach_the_model(self):
        step = _classify(_state(), save_model=False, hyperparameters={"max_iter": 250})
        assert step.status == "success"
        assert step.result["hyperparameters"]["max_iter"] == 250


def test_only_file_safety_dumps_models():
    """The F4/F5 pattern cannot come back: one place writes model files."""
    offenders = [
        str(path.relative_to(PACKAGE))
        for path in PACKAGE.rglob("*.py")
        if "joblib.dump(" in path.read_text() and path.name != "file_safety.py"
    ]
    assert offenders == []
