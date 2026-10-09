"""Every tool that writes a file honours the session's write root (re-scan F1–F5, F7, 9 Oct 2026).

Six plot tools called safe_save without the session's write root, so in a
confined session a caller's save_path still wrote figures anywhere writable,
including another session's export folder. This test calls every registered
tool that takes a save path, under a write root, with a path pointing outside
it, and checks the file lands inside. A new writer without an entry below
fails test_every_writer_is_listed, so the gap cannot reopen unnoticed.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState
from stats_compass_core.utils.file_safety import FilePolicy

PACKAGE = Path(__file__).resolve().parent.parent / "stats_compass_core"
PATH_FIELDS = {"save_path", "filepath"}
READERS = {"load_csv", "load_excel"}  # covered by test_file_policy


def _frame():
    rng = np.random.default_rng(0)
    n = 120
    x1 = rng.normal(size=n)
    label = (x1 + rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame({
        "x1": x1,
        "x2": rng.normal(size=n),
        "y": 2 * x1 + rng.normal(size=n),
        "label": label,
        "pred": label,
        "prob": np.clip(0.5 + 0.3 * x1, 0, 1),
        "cat": rng.choice(["a", "b", "c"], size=n),
        "v": np.sin(np.arange(n) / 4) + np.arange(n) * 0.05,
    })


def _call(state, name, **params):
    meta = next(m for m in registry.list_tools() if m.name == name)
    return meta.function(state, meta.input_schema(**params))


@pytest.fixture(scope="module")
def models():
    """Model ids for the writers that need one, trained in an unconfined state."""
    pytest.importorskip("sklearn")
    pytest.importorskip("statsmodels")
    state = DataFrameState(file_policy=FilePolicy())
    state.set_dataframe(_frame(), "t", "test")
    regression = _call(state, "train_linear_regression", dataframe_name="t", target_column="y",
                       feature_columns=["x1", "x2"])
    arima = _call(state, "fit_arima", dataframe_name="t", target_column="v", p=1, d=0, q=0)
    return state, regression.model_id, arima.model_id


TRAIN = {"dataframe_name": "t", "feature_columns": ["x1", "x2"]}

# Each writer: its parameters (without the path) and the extension it writes.
WRITERS = {
    "save_csv": (lambda ids: {"dataframe_name": "t"}, "filepath", ".csv"),
    "save_model": (lambda ids: {"model_id": ids["regression"]}, "filepath", ".joblib"),
    "fit_arima": (lambda ids: {"dataframe_name": "t", "target_column": "v", "p": 1, "d": 0, "q": 0}, "save_path", ".joblib"),
    "train_linear_regression": (lambda ids: {**TRAIN, "target_column": "y"}, "save_path", ".joblib"),
    "train_gradient_boosting_regressor": (lambda ids: {**TRAIN, "target_column": "y"}, "save_path", ".joblib"),
    "train_random_forest_regressor": (lambda ids: {**TRAIN, "target_column": "y"}, "save_path", ".joblib"),
    "train_logistic_regression": (lambda ids: {**TRAIN, "target_column": "label"}, "save_path", ".joblib"),
    "train_gradient_boosting_classifier": (lambda ids: {**TRAIN, "target_column": "label"}, "save_path", ".joblib"),
    "train_random_forest_classifier": (lambda ids: {**TRAIN, "target_column": "label"}, "save_path", ".joblib"),
    "bar_chart": (lambda ids: {"dataframe_name": "t", "column": "cat"}, "save_path", ".png"),
    "histogram": (lambda ids: {"dataframe_name": "t", "column": "x1"}, "save_path", ".png"),
    "lineplot": (lambda ids: {"dataframe_name": "t", "y_column": "x1"}, "save_path", ".png"),
    "scatter_plot": (lambda ids: {"dataframe_name": "t", "x": "x1", "y": "x2"}, "save_path", ".png"),
    "confusion_matrix_plot": (lambda ids: {"dataframe_name": "t", "true_column": "label", "pred_column": "pred"}, "save_path", ".png"),
    "roc_curve_plot": (lambda ids: {"dataframe_name": "t", "true_column": "label", "prob_column": "prob"}, "save_path", ".png"),
    "precision_recall_curve_plot": (lambda ids: {"dataframe_name": "t", "true_column": "label", "prob_column": "prob"}, "save_path", ".png"),
    "feature_importance": (lambda ids: {"model_id": ids["regression"]}, "save_path", ".png"),
    "forecast_plot": (lambda ids: {"model_id": ids["arima"]}, "save_path", ".png"),
}


def test_every_writer_is_listed():
    registry.auto_discover()
    writers = {
        m.name for m in registry.list_tools()
        if PATH_FIELDS & set(m.input_schema.model_fields) and m.name not in READERS
    }
    assert writers == set(WRITERS)


@pytest.mark.parametrize("name", sorted(WRITERS))
def test_the_file_lands_inside_the_write_root(name, models, tmp_path):
    trained, regression_id, arima_id = models
    exports, elsewhere = tmp_path / "exports", tmp_path / "elsewhere"
    elsewhere.mkdir()
    state = DataFrameState(file_policy=FilePolicy(write_root=exports))
    state.set_dataframe(_frame(), "t", "test")
    for model_id in (regression_id, arima_id):  # carry the trained models across
        state._models[model_id] = trained._models[model_id]
        state._model_metadata[model_id] = trained._model_metadata[model_id]
    params_for, field, ext = WRITERS[name]
    params = params_for({"regression": regression_id, "arima": arima_id})
    params[field] = str(elsewhere / f"{name}{ext}")
    _call(state, name, **params)
    assert list(elsewhere.iterdir()) == []
    written = [p for p in exports.rglob("*") if p.is_file()]
    assert [p.name for p in written] == [f"{name}{ext}"]


def test_every_save_call_names_its_root():
    """safe_save, safe_write_path and safe_save_figure take the root explicitly at
    every call in the package; root=None is a decision, not an omission."""
    missing = []
    for path in PACKAGE.rglob("*.py"):
        if path.name == "file_safety.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name in ("safe_save", "safe_write_path", "safe_save_figure"):
                    if not any(k.arg == "root" for k in node.keywords):
                        missing.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
    assert missing == []


def test_without_a_root_argument_the_environment_decides(tmp_path, monkeypatch):
    """A caller that passes no root at all gets STATS_COMPASS_WRITE_ROOT, not anywhere."""
    from stats_compass_core.utils.file_safety import safe_write_path

    root = tmp_path / "env-root"
    monkeypatch.setenv("STATS_COMPASS_WRITE_ROOT", str(root))
    path = safe_write_path(str(tmp_path / "elsewhere" / "x.csv"))
    assert Path(path).parent == root.resolve()
