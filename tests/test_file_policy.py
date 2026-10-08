"""Where tools may read and write files (security scan F6, F8, 8 Oct 2026).

On a laptop a caller's path is the user's own choice, and that stays the
default. A server sets a FilePolicy on each session's state, or the
STATS_COMPASS_WRITE_ROOT / STATS_COMPASS_READ_ROOTS environment variables for
every state: writes then land, by base name only, inside the write root, with
an extension that fits the file; reads and listings must resolve inside a read
root. Containment is checked on the real path, so a symlink cannot lead out.
"""

import os

import pandas as pd
import pytest

from stats_compass_core.data.list_files import ListFilesInput, list_files
from stats_compass_core.data.load_csv import LoadCSVInput, load_csv
from stats_compass_core.data.load_dataset import LoadDatasetInput, load_dataset
from stats_compass_core.data.save_csv import SaveCSVInput, save_csv
from stats_compass_core.plots.classification_curves import ROCCurveInput, roc_curve_plot
from stats_compass_core.state import DataFrameState
from stats_compass_core.utils.file_safety import FilePolicy, UnsafePathError


@pytest.fixture
def dirs(tmp_path):
    out = {name: tmp_path / name for name in ("exports", "uploads", "elsewhere")}
    for path in out.values():
        path.mkdir()
    (out["uploads"] / "orders.csv").write_text("a,b\n1,2\n")
    (out["elsewhere"] / "secret.csv").write_text("key\nhunter2\n")
    return out


@pytest.fixture
def confined(dirs):
    state = DataFrameState(
        file_policy=FilePolicy(write_root=dirs["exports"], read_roots=(dirs["uploads"],))
    )
    state.set_dataframe(pd.DataFrame({"y": [0, 1, 0, 1], "p": [0.1, 0.9, 0.4, 0.6]}), "scores", "test")
    return state


def _everything_under(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


class TestWritesStayInTheRoot:
    @pytest.mark.parametrize(
        "requested",
        ["out.csv", "sub/dir/out.csv", "../out.csv", "/tmp/out.csv", "~/out.csv"],
    )
    def test_save_csv_lands_in_the_root_by_base_name(self, confined, dirs, requested):
        result = save_csv(confined, SaveCSVInput(dataframe_name="scores", filepath=requested))
        assert os.path.dirname(result["filepath"]) == os.path.realpath(dirs["exports"])
        assert os.path.basename(result["filepath"]) == "out.csv"
        assert _everything_under(dirs["elsewhere"]) == ["secret.csv"]

    def test_plot_save_path_lands_in_the_root(self, confined, dirs):
        roc_curve_plot(
            confined,
            ROCCurveInput(true_column="y", prob_column="p", save_path="../../../evil/roc.png"),
        )
        assert _everything_under(dirs["exports"]) == ["roc.png"]
        assert not (dirs["exports"].parent.parent / "evil").exists()

    @pytest.mark.parametrize("name", ["roc.py", "roc.sh", "roc.exe", "roc"])
    def test_a_figure_must_have_an_image_extension(self, confined, name):
        with pytest.raises(UnsafePathError):
            roc_curve_plot(confined, ROCCurveInput(true_column="y", prob_column="p", save_path=name))

    @pytest.mark.parametrize("name", ["data.joblib", "data.png", "data.py"])
    def test_a_csv_must_have_a_csv_extension(self, confined, name):
        with pytest.raises(UnsafePathError):
            save_csv(confined, SaveCSVInput(dataframe_name="scores", filepath=name))

    def test_a_symlink_in_the_root_cannot_lead_out(self, confined, dirs):
        target = dirs["elsewhere"] / "planted.csv"
        (dirs["exports"] / "out.csv").symlink_to(target)  # dangling: target missing
        result = save_csv(confined, SaveCSVInput(dataframe_name="scores", filepath="out.csv"))
        assert not target.exists()
        assert os.path.realpath(result["filepath"]).startswith(os.path.realpath(dirs["exports"]))

    def test_the_root_is_created_if_missing(self, dirs):
        root = dirs["exports"] / "session-1"
        state = DataFrameState(file_policy=FilePolicy(write_root=root))
        state.set_dataframe(pd.DataFrame({"a": [1]}), "t", "test")
        save_csv(state, SaveCSVInput(dataframe_name="t", filepath="t.csv"))
        assert (root / "t.csv").exists()


class TestReadsStayInTheRoots:
    def test_a_file_inside_loads(self, confined, dirs):
        load_csv(confined, LoadCSVInput(file_path=str(dirs["uploads"] / "orders.csv")))
        assert confined.get_dataframe("orders").shape == (1, 2)

    @pytest.mark.parametrize(
        "path",
        ["{elsewhere}/secret.csv", "{uploads}/../elsewhere/secret.csv", "/etc/hosts"],
    )
    def test_a_file_outside_is_refused(self, confined, dirs, path):
        with pytest.raises(UnsafePathError):
            load_csv(confined, LoadCSVInput(file_path=path.format(**dirs)))

    def test_a_symlink_inside_cannot_lead_out(self, confined, dirs):
        (dirs["uploads"] / "link.csv").symlink_to(dirs["elsewhere"] / "secret.csv")
        with pytest.raises(UnsafePathError):
            load_csv(confined, LoadCSVInput(file_path=str(dirs["uploads"] / "link.csv")))

    def test_listing_outside_is_refused(self, confined, dirs):
        with pytest.raises(UnsafePathError):
            list_files(confined, ListFilesInput(directory=str(dirs["elsewhere"])))
        with pytest.raises(UnsafePathError):
            list_files(confined, ListFilesInput(directory="/"))

    def test_listing_inside_works(self, confined, dirs):
        result = list_files(confined, ListFilesInput(directory=str(dirs["uploads"])))
        assert result.files == ["orders.csv"]

    def test_a_relative_path_resolves_against_the_first_root(self, confined):
        load_csv(confined, LoadCSVInput(file_path="orders.csv"))
        assert confined.get_dataframe("orders").shape == (1, 2)


class TestEnvironmentSetsTheDefault:
    def test_both_variables(self, dirs, monkeypatch):
        monkeypatch.setenv("STATS_COMPASS_WRITE_ROOT", str(dirs["exports"]))
        monkeypatch.setenv(
            "STATS_COMPASS_READ_ROOTS", os.pathsep.join([str(dirs["uploads"]), str(dirs["exports"])])
        )
        policy = DataFrameState().file_policy
        assert policy.write_root == dirs["exports"]
        assert policy.read_roots == (dirs["uploads"], dirs["exports"])

    def test_unset_means_unconfined(self, monkeypatch):
        monkeypatch.delenv("STATS_COMPASS_WRITE_ROOT", raising=False)
        monkeypatch.delenv("STATS_COMPASS_READ_ROOTS", raising=False)
        policy = DataFrameState().file_policy
        assert policy.write_root is None and policy.read_roots is None


class TestUnconfinedIsUnchanged:
    """A laptop user's own paths still work when no policy is set."""

    def test_save_and_load_anywhere(self, dirs, monkeypatch):
        monkeypatch.delenv("STATS_COMPASS_WRITE_ROOT", raising=False)
        monkeypatch.delenv("STATS_COMPASS_READ_ROOTS", raising=False)
        state = DataFrameState()
        state.set_dataframe(pd.DataFrame({"a": [1]}), "t", "test")
        target = dirs["elsewhere"] / "nested" / "t.csv"
        save_csv(state, SaveCSVInput(dataframe_name="t", filepath=str(target)))
        assert target.exists()
        load_csv(state, LoadCSVInput(file_path=str(dirs["elsewhere"] / "secret.csv")))


class TestBuiltInDatasets:
    @pytest.mark.parametrize("name", ["../../pyproject", "../datasets/Housing", "/etc/hosts", "a/b"])
    def test_a_name_cannot_leave_the_datasets_folder(self, name):
        with pytest.raises((ValueError, FileNotFoundError)):
            load_dataset(DataFrameState(), LoadDatasetInput(name=name))

    def test_a_listed_name_loads(self):
        state = DataFrameState()
        load_dataset(state, LoadDatasetInput(name="Housing"))
        assert len(state.get_dataframe("Housing")) > 0


def test_the_denylist_reads_the_real_path(tmp_path):
    """A symlink to a system folder is judged by where it leads."""
    from stats_compass_core.utils.file_safety import is_path_safe

    link = tmp_path / "etc-link"
    link.symlink_to("/etc")
    ok, _ = is_path_safe(str(link / "passwd.csv"))
    assert not ok


def test_unsafe_path_error_is_a_value_error():
    """Callers that catch ValueError for bad input keep catching this."""
    assert issubclass(UnsafePathError, ValueError)
