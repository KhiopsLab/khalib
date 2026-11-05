import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from khiops.sklearn import KhiopsClassifier
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix, dok_matrix, lil_matrix
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from khalib import (
    Histogram,
    KhalibClassifier,
    build_reliability_diagram,
    calibrate_binary,
    calibration_error,
)
from khalib import main as khalib_main
from khalib.main import compute_dirac_indexes


@pytest.mark.parametrize(
    "scores",
    [
        [0.1 + 100j, 0.9 - 100j],
        np.array([0.1 + 100j, 0.9 - 100j]),
        np.array([0.1 + 100j, 0.9 - 100j], dtype=object),
        np.array([0.1 + 0j, 0.9 + 0j]),
        np.array([[0.1 + 100j], [0.9 - 100j]]),
    ],
)
@pytest.mark.parametrize(
    "api", ["histogram", "manual", "calibrate", "ece", "diagram", "vfind"]
)
def test_score_apis_reject_complex_values(scores, api):
    histogram = Histogram([0.0, 1.0], [2], [(1, 1)], [0, 1])
    function, args = {
        "histogram": (Histogram.from_data, ([0, 1],)),
        "manual": (Histogram.from_data_and_breakpoints, ([0.0, 1.0], [0, 1])),
        "calibrate": (calibrate_binary, (histogram,)),
        "ece": (calibration_error, ([0, 1],)),
        "diagram": (build_reliability_diagram, ([0, 1],)),
        "vfind": (histogram.vfind, ()),
    }[api]
    with pytest.raises(ValueError, match="must contain only real values"):
        function(scores, *args)


@pytest.mark.parametrize(
    "scores",
    [[True, False], np.array([True, False]), np.array([0.1, True], dtype=object)],
)
@pytest.mark.parametrize(
    "api", ["histogram", "manual", "calibrate", "ece", "diagram", "vfind"]
)
def test_score_apis_reject_boolean_values(scores, api):
    histogram = Histogram([0.0, 1.0], [2], [(1, 1)], [0, 1])
    function, args = {
        "histogram": (Histogram.from_data, ([0, 1],)),
        "manual": (Histogram.from_data_and_breakpoints, ([0.0, 1.0], [0, 1])),
        "calibrate": (calibrate_binary, (histogram,)),
        "ece": (calibration_error, ([0, 1],)),
        "diagram": (build_reliability_diagram, ([0, 1],)),
        "vfind": (histogram.vfind, ()),
    }[api]
    with pytest.raises(ValueError, match="must not contain boolean values"):
        function(scores, *args)


@pytest.mark.parametrize(
    "scores",
    [
        ["0.1", "0.9"],
        np.array(["0.1", "0.9"]),
        np.array([0.1, "0.9"], dtype=object),
    ],
)
@pytest.mark.parametrize(
    "api", ["histogram", "manual", "calibrate", "ece", "diagram", "vfind"]
)
def test_score_apis_reject_numeric_strings(scores, api):
    histogram = Histogram([0.0, 1.0], [2], [(1, 1)], [0, 1])
    function, args = {
        "histogram": (Histogram.from_data, ([0, 1],)),
        "manual": (Histogram.from_data_and_breakpoints, ([0.0, 1.0], [0, 1])),
        "calibrate": (calibrate_binary, (histogram,)),
        "ece": (calibration_error, ([0, 1],)),
        "diagram": (build_reliability_diagram, ([0, 1],)),
        "vfind": (histogram.vfind, ()),
    }[api]
    with pytest.raises(ValueError, match="must contain only numeric values"):
        function(scores, *args)


@pytest.mark.parametrize(
    "api", ["histogram", "manual", "calibrate", "ece", "diagram", "vfind"]
)
def test_score_apis_reject_missing_values(api):
    scores = np.array([0.1, pd.NA], dtype=object)
    histogram = Histogram([0.0, 1.0], [2], [(1, 1)], [0, 1])
    function, args = {
        "histogram": (Histogram.from_data, ([0, 1],)),
        "manual": (Histogram.from_data_and_breakpoints, ([0.0, 1.0], [0, 1])),
        "calibrate": (calibrate_binary, (histogram,)),
        "ece": (calibration_error, ([0, 1],)),
        "diagram": (build_reliability_diagram, ([0, 1],)),
        "vfind": (histogram.vfind, ()),
    }[api]
    with pytest.raises(ValueError, match="must not contain NaN or missing values"):
        function(scores, *args)


@pytest.mark.parametrize("dtype", [complex, np.complex64, np.complex128])
@pytest.mark.parametrize("imaginary", [0, 1])
@pytest.mark.parametrize("as_object", [False, True])
@pytest.mark.parametrize("field", ["breakpoints", "freqs", "target_freqs", "classes"])
def test_histogram_rejects_complex_fields(dtype, imaginary, as_object, field):
    kwargs = {
        "breakpoints": [0.0, 1.0],
        "freqs": [2],
        "target_freqs": [(1, 1)],
        "classes": [0, 1],
    }
    values = np.array(kwargs[field], dtype=dtype)
    values.flat[0] += imaginary * 1j
    kwargs[field] = list(values.astype(object) if as_object else values)
    with pytest.raises(ValueError, match=rf"'{field}' must contain only real values"):
        Histogram(**kwargs)


@pytest.mark.parametrize("breakpoints", [[0j, 1], np.array([0j, 1], dtype=object)])
def test_manual_histogram_rejects_complex_breakpoints(breakpoints):
    with pytest.raises(ValueError, match="'breakpoints' must contain only real values"):
        Histogram.from_data_and_breakpoints([0.1, 0.9], breakpoints)


@pytest.mark.parametrize(
    "matrix_type", [coo_matrix, csr_matrix, dok_matrix, lil_matrix]
)
@pytest.mark.parametrize(
    "api", ["histogram", "manual", "calibrate", "ece", "diagram", "vfind"]
)
def test_sparse_scores_are_rejected(matrix_type, api):
    scores = matrix_type(np.array([[0.1], [0.9]]))
    histogram = Histogram([0.0, 0.5, 1.0], [1, 1], [(1, 0), (0, 1)], [0, 1])
    function, args = {
        "histogram": (Histogram.from_data, ([0, 1],)),
        "manual": (Histogram.from_data_and_breakpoints, ([0.0, 1.0], [0, 1])),
        "calibrate": (calibrate_binary, (histogram,)),
        "ece": (calibration_error, ([0, 1],)),
        "diagram": (build_reliability_diagram, ([0, 1],)),
        "vfind": (histogram.vfind, ()),
    }[api]
    with pytest.raises(ValueError, match="must not be a sparse matrix"):
        function(scores, *args)


@pytest.mark.parametrize(
    "matrix_type", [coo_matrix, csr_matrix, dok_matrix, lil_matrix]
)
def test_sparse_multiclass_scores_are_rejected(matrix_type):
    scores = matrix_type([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    with pytest.raises(ValueError, match="must not be a sparse matrix"):
        calibration_error(scores, [0, 1, 2])


@pytest.mark.parametrize(
    "matrix_type", [coo_matrix, csr_matrix, dok_matrix, lil_matrix]
)
def test_sparse_labels_are_rejected(matrix_type):
    labels = matrix_type([[0], [1]])
    with pytest.raises(ValueError, match="must not be a sparse matrix"):
        Histogram.from_data([0.1, 0.9], labels)


def test_sparse_histogram_fields_are_rejected():
    with pytest.raises(ValueError, match="must not be a sparse matrix"):
        Histogram(breakpoints=csr_matrix([[0.0, 1.0]]), freqs=[2])

    histogram = Histogram([0.0, 1.0], [2])
    with pytest.raises(ValueError, match="must not be a sparse matrix"):
        histogram.find(csr_matrix([[0.5]]))


@pytest.mark.parametrize(
    "function",
    [Histogram.from_data, calibration_error, build_reliability_diagram],
)
@pytest.mark.parametrize("labels", [[0j, 1], np.array([0, 1j], dtype=object)])
def test_score_apis_reject_complex_labels(function, labels):
    with pytest.raises(ValueError, match="'y' must contain only real values"):
        function([0.1, 0.9], labels)


@pytest.mark.parametrize(
    "labels",
    [
        [0, np.nan],
        np.array(["class", np.nan], dtype=object),
        np.array(["class", pd.NA], dtype=object),
    ],
)
def test_histogram_and_score_apis_reject_missing_labels(labels):
    with pytest.raises(ValueError, match="'y' must not contain NaN or missing values"):
        Histogram.from_data([0.1, 0.9], labels)
    with pytest.raises(ValueError, match="'y' must not contain NaN or missing values"):
        Histogram.from_data([0.1, 0.9], labels, method="eq-width")
    with pytest.raises(ValueError, match="'y' must not contain NaN or missing values"):
        Histogram.from_data_and_breakpoints([0.1, 0.9], [0.0, 1.0], labels)
    for method in ("eq-width", "eq-freq"):
        with pytest.raises(
            ValueError, match="'y' must not contain NaN or missing values"
        ):
            calibration_error([0.1, 0.9], labels, histogram_method=method)
    with pytest.raises(ValueError, match="'y' must not contain NaN or missing values"):
        build_reliability_diagram([0.1, 0.9], labels)


@pytest.mark.parametrize("missing", [np.nan, pd.NA])
def test_histogram_rejects_missing_classes(missing):
    with pytest.raises(
        ValueError, match="'classes' must not contain NaN or missing values"
    ):
        Histogram(
            [0.0, 1.0],
            [2],
            target_freqs=[(1, 1)],
            classes=[0, missing],
        )


@pytest.mark.parametrize(
    ("function", "parameter"),
    [
        (Histogram.from_data, "max_bins"),
        (calibration_error, "max_bins"),
        (build_reliability_diagram, "dirac_threshold"),
        (build_reliability_diagram, "log_plot_threshold"),
        (build_reliability_diagram, "min_density_bar_width"),
    ],
)
@pytest.mark.parametrize("value", [1 + 0j, np.complex128(1 + 2j)])
def test_score_apis_reject_complex_options(function, parameter, value):
    with pytest.raises(
        ValueError, match=rf"'{parameter}' must contain only real values"
    ):
        function([0.1, 0.9], [0, 1], **{parameter: value})


@pytest.mark.parametrize(
    "value", [1.5, 1.0, np.inf, np.nan, True, np.bool_(False), "2"]
)
def test_histogram_rejects_noninteger_max_bins(value):
    with pytest.raises(ValueError, match="'max_bins' must be a nonnegative integer"):
        Histogram.from_data([0.1, 0.9], max_bins=value)


def test_histogram_rejects_negative_max_bins():
    with pytest.raises(ValueError, match="'max_bins' must be nonnegative"):
        Histogram.from_data([0.1, 0.9], max_bins=-1)


@pytest.mark.parametrize("value", [1.5, 1.0, np.inf, np.nan, True, "2", -1])
def test_calibration_error_rejects_invalid_max_bins(value):
    with pytest.raises(ValueError, match="'max_bins'.*nonnegative"):
        calibration_error([0.1, 0.9], [0, 1], max_bins=value)


@pytest.mark.parametrize("max_bins", [1.5, np.inf, np.nan, True, "2", -1])
def test_calibration_error_ignores_max_bins_with_histogram(max_bins):
    histogram = Histogram([0.0, 1.0], [2], [(1, 1)], [0, 1])
    expected = calibration_error([0.1, 0.9], [0, 1], histogram=histogram)
    assert (
        calibration_error([0.1, 0.9], [0, 1], histogram=histogram, max_bins=max_bins)
        == expected
    )


def test_histogram_accepts_numpy_integer_max_bins():
    histogram = Histogram.from_data([0.1, 0.9], method="eq-width", max_bins=np.int64(2))
    assert histogram.n_bins <= 2


@pytest.mark.parametrize("use_finest", ["yes", 1, None])
def test_histogram_rejects_nonboolean_use_finest(use_finest):
    with pytest.raises(TypeError, match="'use_finest' must be a boolean"):
        Histogram.from_data([0.1, 0.9], method="eq-width", use_finest=use_finest)


def test_histogram_accepts_numpy_boolean_use_finest():
    histogram = Histogram.from_data(
        [0.1, 0.9], method="eq-width", use_finest=np.bool_(True)
    )
    assert histogram.n_bins > 0


@pytest.mark.parametrize("reuse_stats", ["yes", 1, None])
def test_calibration_error_rejects_nonboolean_reuse_stats(reuse_stats):
    histogram = Histogram([0.0, 1.0], [2], [(1, 1)], [0, 1])
    with pytest.raises(TypeError, match="'reuse_stats' must be a boolean"):
        calibration_error(
            [0.1, 0.9], [0, 1], histogram=histogram, reuse_stats=reuse_stats
        )


def test_calibration_error_accepts_numpy_boolean_reuse_stats():
    histogram = Histogram([0.0, 1.0], [2], [(1, 1)], [0, 1])
    assert calibration_error(
        [0.1, 0.9], [0, 1], histogram=histogram, reuse_stats=np.bool_(True)
    ) == pytest.approx(0.4)


def test_dirac_rejects_complex_threshold():
    histogram = Histogram([0.0, 1.0], [2])
    with pytest.raises(
        ValueError, match="'dirac_threshold' must contain only real values"
    ):
        compute_dirac_indexes(histogram, np.complex128(1e-6))


@pytest.fixture(name="data_root_dir")
def fixture_data_root_dir():
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(name="adult_scores_df")
def fixture_adult_scores_df(data_root_dir):
    return pd.read_csv(f"{data_root_dir}/tables/adult_scores_sample.tsv", sep="\t")


@pytest.fixture(name="y_fixtures")
def fixture_y_fixtures(adult_scores_df):
    rng = np.random.default_rng(seed=1234567)
    return {
        "int": adult_scores_df.y,
        "str": np.where(adult_scores_df.y == 0, "zero", "one"),
        "intnl": np.where(adult_scores_df.y == 0, 10, 2),
        "bool": np.where(adult_scores_df.y == 0, False, True),
        "float": np.where(adult_scores_df.y == 0, 10.1, 2.1),
        "random": rng.integers(low=0, high=2, size=adult_scores_df.shape[0]),
    }


@pytest.fixture(name="y_scores_fixtures")
def fixture_y_scores_fixtures(adult_scores_df):
    y_scores = adult_scores_df.y_score
    return {
        "original": y_scores,
        "original-2d": np.transpose(np.vstack([1 - y_scores, y_scores])),
        "original-fake-2d": np.array(y_scores).reshape(y_scores.shape[0], 1),
        "constant": np.full(y_scores.shape[0], 0.5),
    }


@pytest.fixture(name="vehicles_scores_df")
def fixture_vehicles_scores_df(data_root_dir):
    return pd.read_csv(f"{data_root_dir}/tables/vehicles_scores.tsv", sep="\t")


@pytest.fixture(name="short_test_id")
def fixture_data_df(request):
    trans_table = str.maketrans("[", "_", "]")
    return request.node.nodeid.split("::")[-1].translate(trans_table)


@pytest.fixture(name="ref_histogram")
def fixture_ref_histogram(data_root_dir, short_test_id):
    with open(f"{data_root_dir}/histogram/ref/{short_test_id}.json") as ref_json_file:
        yield read_histogram_from_json_data(json.load(ref_json_file))


def read_histogram_from_json_data(json_data):
    return Histogram(
        breakpoints=json_data["breakpoints"],
        freqs=json_data["freqs"],
        target_freqs=[tuple(cur_freqs) for cur_freqs in json_data["target_freqs"]],
        classes=json_data["classes"],
    )


def is_target_inverted(y_fixture):
    return y_fixture in ["float", "intnl", "str"]


class TestSingleCoreRunner:
    def test_context_serializes_runner_changes(self, monkeypatch):
        default_runner = object()
        single_core_runner = object()
        active_runner = [default_runner]
        first_entered = threading.Event()
        release_first = threading.Event()
        second_attempting = threading.Event()
        second_entered = threading.Event()

        monkeypatch.setattr(
            khalib_main, "khiops_single_core_runner", single_core_runner
        )
        monkeypatch.setattr(khalib_main.kh, "get_runner", lambda: active_runner[0])
        monkeypatch.setattr(
            khalib_main.kh,
            "set_runner",
            lambda runner: active_runner.__setitem__(0, runner),
        )

        def first_call():
            with khalib_main.single_core_khiops_runner():
                first_entered.set()
                assert release_first.wait(timeout=5)

        def second_call():
            second_attempting.set()
            with khalib_main.single_core_khiops_runner():
                second_entered.set()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(first_call)
            assert first_entered.wait(timeout=5)
            second = executor.submit(second_call)
            assert second_attempting.wait(timeout=5)
            assert not second_entered.wait(timeout=0.1)
            release_first.set()
            first.result(timeout=5)
            second.result(timeout=5)

        assert second_entered.is_set()
        assert active_runner[0] is default_runner

    @pytest.mark.parametrize("core_limit", [None, "8"])
    def test_initialization_failure_restores_environment(self, monkeypatch, core_limit):
        # Preserve both an absent core limit and an explicitly configured one
        if core_limit is None:
            monkeypatch.delenv("KHIOPS_PROC_NUMBER", raising=False)
        else:
            monkeypatch.setenv("KHIOPS_PROC_NUMBER", core_limit)
        monkeypatch.setattr(khalib_main, "khiops_single_core_runner", None)
        error = RuntimeError("Runner initialization failed")

        def failing_runner():
            assert os.environ["KHIOPS_PROC_NUMBER"] == "1"
            raise error

        monkeypatch.setattr(khalib_main, "KhiopsLocalRunner", failing_runner)
        with pytest.raises(RuntimeError, match="Runner initialization failed") as exc:
            khalib_main.build_single_core_khiops_runner()
        assert exc.value is error
        assert os.environ.get("KHIOPS_PROC_NUMBER") == core_limit
        assert khalib_main.khiops_single_core_runner is None

    @pytest.mark.parametrize("core_limit", [None, "8"])
    def test_initialization_restores_environment(self, monkeypatch, core_limit):
        # Build a real runner while keeping the cached runner isolated
        if core_limit is None:
            monkeypatch.delenv("KHIOPS_PROC_NUMBER", raising=False)
        else:
            monkeypatch.setenv("KHIOPS_PROC_NUMBER", core_limit)
        monkeypatch.setattr(khalib_main, "khiops_single_core_runner", None)
        khalib_main.build_single_core_khiops_runner()
        assert khalib_main.khiops_single_core_runner is not None
        assert os.environ.get("KHIOPS_PROC_NUMBER") == core_limit


class TestHistogram:
    @pytest.mark.parametrize("classes", [["cat", "cat"], [0, 0]])
    def test_rejects_duplicate_classes_with_target_statistics(self, classes):
        with pytest.raises(
            ValueError,
            match="`classes` must contain unique labels",
        ):
            Histogram(
                breakpoints=[0.0, 1.0],
                freqs=[2],
                target_freqs=[(1, 1)],
                classes=classes,
            )

    @pytest.mark.parametrize("method", ["khiops", "eq-freq", "eq-width"])
    @pytest.mark.parametrize("supervised", [False, True])
    def test_from_data_rejects_empty_samples(self, method, supervised):
        with pytest.raises(ValueError, match="'x' must contain at least one sample"):
            Histogram.from_data([], y=[] if supervised else None, method=method)

    @pytest.mark.parametrize("supervised", [False, True])
    def test_manual_histogram_rejects_empty_samples(self, supervised):
        with pytest.raises(ValueError, match="'x' must contain at least one sample"):
            Histogram.from_data_and_breakpoints(
                [], [0.0, 1.0], y=[] if supervised else None
            )

    @pytest.mark.parametrize("input_type", ["list", "array", "object", "column"])
    @pytest.mark.parametrize("base", [2**60, 2**63])
    def test_large_integer_scores_retain_bin_assignments(self, input_type, base):
        scores = [base, base + 2]
        if input_type == "array":
            scores = np.array(scores)
        elif input_type == "object":
            scores = np.array(scores, dtype=object)
        elif input_type == "column":
            scores = np.array(scores).reshape(-1, 1)
        histogram = Histogram.from_data_and_breakpoints(
            scores, [base - 1, base + 1, base + 3], [0, 1]
        )
        assert histogram.freqs == [1, 1]
        assert histogram.target_freqs == [(1, 0), (0, 1)]
        np.testing.assert_array_equal(histogram.vfind(scores).reshape(-1), [0, 1])
        np.testing.assert_array_equal(
            calibrate_binary(scores, histogram), [[1.0, 0.0], [0.0, 1.0]]
        )

    def test_mixed_signed_unsigned_integer_lookup(self):
        base = 2**63
        scores = [np.int64(-1), np.uint64(base), np.uint64(base + 2)]
        histogram = Histogram.from_data_and_breakpoints(
            scores, [-2, 0, base + 1, base + 3], [0, 0, 1]
        )
        assert histogram.freqs == [1, 1, 1]
        np.testing.assert_array_equal(histogram.vfind(scores), [0, 1, 2])
        assert [histogram.find(value) for value in scores] == [0, 1, 2]

    def test_exact_large_integer_boundaries_and_clamping(self):
        base = 2**60
        histogram = Histogram([base, base + 2, base + 4], [1, 1])
        scores = [base - 1, base, base + 1, base + 2, base + 3, base + 4, base + 5]
        expected = [0, 0, 0, 0, 1, 1, 1]
        np.testing.assert_array_equal(histogram.vfind(scores), expected)
        assert [histogram.find(value) for value in scores] == expected

    @pytest.mark.parametrize("breakpoints", [[], [0.0]])
    def test_constructor_rejects_insufficient_breakpoints(self, breakpoints):
        with pytest.raises(ValueError, match="`breakpoints` must contain at least two"):
            Histogram(breakpoints=breakpoints, freqs=[])

    @pytest.mark.parametrize("breakpoints", [[], [0.0]])
    @pytest.mark.parametrize("supervised", [False, True])
    def test_manual_histogram_rejects_insufficient_breakpoints(
        self, breakpoints, supervised
    ):
        with pytest.raises(ValueError, match="`breakpoints` must contain at least two"):
            Histogram.from_data_and_breakpoints(
                [0.1, 0.9], breakpoints, y=[0, 1] if supervised else None
            )

    @pytest.mark.parametrize("breakpoint_index", [0, 1, 2])
    @pytest.mark.parametrize("supervised", [False, True])
    def test_constructor_rejects_nan_breakpoints(self, breakpoint_index, supervised):
        breakpoints = [0.0, 0.5, 1.0]
        breakpoints[breakpoint_index] = np.nan
        with pytest.raises(
            ValueError, match=rf"`breakpoints` contains NaN at index {breakpoint_index}"
        ):
            Histogram(
                breakpoints=breakpoints,
                freqs=[1, 1],
                target_freqs=[(1, 0), (0, 1)] if supervised else [],
                classes=[0, 1] if supervised else [],
            )

    @pytest.mark.parametrize("breakpoint_index", [0, 1, 2])
    @pytest.mark.parametrize("supervised", [False, True])
    def test_manual_histogram_rejects_nan_breakpoints(
        self, breakpoint_index, supervised
    ):
        breakpoints = [0.0, 0.5, 1.0]
        breakpoints[breakpoint_index] = np.nan
        with pytest.raises(
            ValueError, match=rf"`breakpoints` contains NaN at index {breakpoint_index}"
        ):
            Histogram.from_data_and_breakpoints(
                [0.1, 0.9], breakpoints, y=[0, 1] if supervised else None
            )

    @pytest.mark.parametrize("value", [-np.inf, np.inf])
    @pytest.mark.parametrize("manual", [False, True])
    def test_rejects_infinite_breakpoints(self, value, manual):
        breakpoints = [0.0, value] if value > 0 else [value, 0.0]
        factory = Histogram.from_data_and_breakpoints if manual else Histogram
        args = ([0.1, 0.9], breakpoints) if manual else ()
        kwargs = {} if manual else {"breakpoints": breakpoints, "freqs": [2]}
        with pytest.raises(ValueError, match=r"`breakpoints` contains infinity"):
            factory(*args, **kwargs)

    @pytest.mark.parametrize("frequency", [np.inf, -np.inf, np.nan])
    @pytest.mark.parametrize("bin_index", [0, 1])
    @pytest.mark.parametrize("supervised", [False, True])
    def test_rejects_nonfinite_bin_frequencies(self, frequency, bin_index, supervised):
        freqs = [2, 2]
        freqs[bin_index] = frequency
        with pytest.raises(
            ValueError, match=rf"`freqs` must be finite.*bin index {bin_index}"
        ):
            Histogram(
                breakpoints=[0.0, 0.5, 1.0],
                freqs=freqs,
                target_freqs=[(0, freq) for freq in freqs] if supervised else [],
                classes=[0, 1] if supervised else [],
            )

    @pytest.mark.parametrize("frequency", [np.inf, -np.inf, np.nan])
    @pytest.mark.parametrize("bin_index", [0, 1])
    @pytest.mark.parametrize("class_index", [0, 1])
    def test_rejects_nonfinite_target_frequencies(
        self, frequency, bin_index, class_index
    ):
        target_freqs = [(1, 1), (1, 1)]
        target_freqs[bin_index] = (frequency, 1) if class_index == 0 else (1, frequency)
        with pytest.raises(
            ValueError,
            match=(
                "`target_freqs` must be finite.*"
                rf"bin index {bin_index}, class index {class_index}"
            ),
        ):
            Histogram(
                breakpoints=[0.0, 0.5, 1.0],
                freqs=[2, 2],
                target_freqs=target_freqs,
                classes=[0, 1],
            )

    @pytest.mark.parametrize("bin_index", [0, 1])
    @pytest.mark.parametrize("supervised", [False, True])
    def test_rejects_negative_bin_frequencies(self, bin_index, supervised):
        # Matching target sums must not allow a negative bin frequency
        freqs = [2, 2]
        freqs[bin_index] = -1
        target_freqs = [(0, freq) for freq in freqs] if supervised else []
        with pytest.raises(
            ValueError, match=rf"`freqs` must be nonnegative.*bin index {bin_index}"
        ):
            Histogram(
                breakpoints=[0.0, 0.5, 1.0],
                freqs=freqs,
                target_freqs=target_freqs,
                classes=[0, 1] if supervised else [],
            )

    @pytest.mark.parametrize("bin_index", [0, 1])
    @pytest.mark.parametrize("class_index", [0, 1])
    def test_rejects_negative_target_frequencies(self, bin_index, class_index):
        # Negative counts can still sum to a valid positive bin frequency
        target_freqs = [(1, 1), (1, 1)]
        target_freqs[bin_index] = (-1, 3) if class_index == 0 else (3, -1)
        with pytest.raises(
            ValueError,
            match=(
                "`target_freqs` must be nonnegative.*"
                rf"bin index {bin_index}, class index {class_index}"
            ),
        ):
            Histogram(
                breakpoints=[0.0, 0.5, 1.0],
                freqs=[2, 2],
                target_freqs=target_freqs,
                classes=[0, 1],
            )

    @pytest.mark.parametrize("n_bins", [1, 2])
    @pytest.mark.parametrize("supervised", [False, True])
    def test_rejects_zero_total_frequency(self, n_bins, supervised):
        with pytest.raises(ValueError, match="`freqs` must have a positive total"):
            Histogram(
                breakpoints=list(range(n_bins + 1)),
                freqs=[0] * n_bins,
                target_freqs=[(0, 0)] * n_bins if supervised else [],
                classes=[0, 1] if supervised else [],
            )

    @pytest.mark.parametrize("supervised", [False, True])
    def test_rejects_overflowing_total_frequency(self, supervised):
        with pytest.raises(ValueError, match="`freqs` must have a finite total"):
            Histogram(
                breakpoints=[0.0, 0.5, 1.0],
                freqs=[1e308, 1e308],
                target_freqs=[(1e308, 0), (0, 1e308)] if supervised else [],
                classes=[0, 1] if supervised else [],
            )

    def test_large_finite_total_frequency(self):
        histogram = Histogram(
            breakpoints=[0.0, 0.5, 1.0],
            freqs=[5e307, 5e307],
            target_freqs=[(5e307, 0), (0, 5e307)],
            classes=[0, 1],
        )
        assert histogram.densities == [1.0, 1.0]
        assert histogram.target_probas == [(1.0, 0.0), (0.0, 1.0)]

    def test_arbitrary_precision_integer_frequencies(self):
        frequency = 10**400
        histogram = Histogram(
            breakpoints=[0.0, 0.5, 1.0],
            freqs=[frequency, frequency],
            target_freqs=[(frequency // 2, frequency // 2)] * 2,
            classes=[0, 1],
        )
        assert histogram.densities == [1.0, 1.0]
        assert histogram.target_probas == [(0.5, 0.5), (0.5, 0.5)]

    @pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint64])
    @pytest.mark.parametrize("manual", [False, True])
    def test_numpy_integer_breakpoint_widths(self, dtype, manual):
        limits = np.iinfo(dtype)
        breakpoints = [dtype(limits.min), dtype(limits.max)]
        if manual:
            histogram = Histogram.from_data_and_breakpoints([0], breakpoints)
        else:
            histogram = Histogram(breakpoints=breakpoints, freqs=[1])
        expected_width = int(limits.max) - int(limits.min)
        assert histogram.bins[0][1] - histogram.bins[0][0] == expected_width
        assert histogram.densities[0] == pytest.approx(
            1 / expected_width, rel=1e-12, abs=0
        )

    def test_adjacent_large_integer_breakpoints_remain_distinct(self):
        right = np.iinfo(np.int64).max
        histogram = Histogram(
            breakpoints=[np.int64(right - 1), np.int64(right)], freqs=[1]
        )
        assert histogram.bins == [(right - 1, right)]
        assert histogram.densities == [1.0]

    def test_arbitrary_precision_integer_breakpoints(self):
        base = 2**1024
        histogram = Histogram([base, base + 1], [1])
        assert histogram.bins == [(base, base + 1)]
        assert histogram.densities == [1.0]
        with pytest.raises(ValueError, match="'value' must contain values within"):
            histogram.find(base)
        with pytest.raises(ValueError, match="'x' must contain values within"):
            Histogram.from_data_and_breakpoints(
                [base, base + 1], [base, base + 1], [0, 1]
            )

    @pytest.mark.parametrize("scores", [[10**400, 10**400 + 1], [10**400, 0.5]])
    def test_rejects_scores_outside_finite_float_range(self, scores):
        histogram = Histogram([0.0, 1.0], [2], target_freqs=[(1, 1)], classes=[0, 1])
        with pytest.raises(ValueError, match="within the finite floating-point range"):
            Histogram.from_data(scores, method="eq-width")
        with pytest.raises(ValueError, match="within the finite floating-point range"):
            Histogram.from_data_and_breakpoints(scores, [0.0, 1.0])
        with pytest.raises(ValueError, match="within the finite floating-point range"):
            histogram.vfind(scores)
        with pytest.raises(ValueError, match="within the finite floating-point range"):
            calibration_error(scores, [0, 1], histogram=histogram)
        with pytest.raises(ValueError, match="within the finite floating-point range"):
            calibrate_binary(scores, histogram)
        with pytest.raises(ValueError, match="within the finite floating-point range"):
            build_reliability_diagram(scores, [0, 1])

    def test_density_for_arbitrary_precision_integer_width(self):
        endpoint = 2**1024
        histogram = Histogram([-endpoint, endpoint], [1])
        assert histogram.densities == [1 / (2 * endpoint)]

    @pytest.mark.parametrize(
        ("dtype", "frequency", "n_bins"),
        [(np.int64, 2**62, 2), (np.int64, 2**62, 5), (np.uint64, 2**63, 3)],
    )
    def test_numpy_integer_frequency_totals(self, dtype, frequency, n_bins):
        histogram = Histogram(
            breakpoints=list(range(n_bins + 1)),
            freqs=[dtype(frequency)] * n_bins,
        )
        assert histogram.densities == pytest.approx([1 / n_bins] * n_bins)

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_numpy_integer_target_frequency_totals(self, dtype):
        frequency = 2**62
        histogram = Histogram(
            breakpoints=[0.0, 1.0],
            freqs=[2 * frequency],
            target_freqs=[(dtype(frequency), dtype(frequency))],
            classes=[0, 1],
        )
        assert histogram.densities == [1.0]
        assert histogram.target_probas == [(0.5, 0.5)]

    def test_rejects_wrapped_target_frequency_sum(self):
        with pytest.raises(ValueError, match="sums different from the bin frequency"):
            Histogram(
                breakpoints=[0.0, 1.0],
                freqs=[2**62],
                target_freqs=[(np.int64(2**62),) * 5],
                classes=list(range(5)),
            )

    def test_nonnegative_frequencies_allow_empty_bins(self):
        histogram = Histogram(
            breakpoints=[0.0, 0.5, 1.0],
            freqs=[0, 2],
            target_freqs=[(0, 0), (0, 2)],
            classes=[0, 1],
        )
        assert histogram.densities == [0.0, 2.0]
        assert histogram.target_probas == [(0, 0), (0.0, 1.0)]

    @pytest.mark.parametrize("method", ["khiops", "eq-freq", "eq-width", "manual"])
    @pytest.mark.parametrize("supervised", [False, True])
    @pytest.mark.parametrize("scores", [[0.1, np.nan], [[0.1], [np.nan]]])
    @pytest.mark.parametrize("dtype", [float, object])
    def test_rejects_nan_scores(self, method, supervised, scores, dtype):
        scores = np.array(scores, dtype=dtype)
        y = [0, 1] if supervised else None
        if method == "manual":
            factory = Histogram.from_data_and_breakpoints
            kwargs = {"breakpoints": [0.0, 0.5, 1.0]}
        else:
            factory = Histogram.from_data
            kwargs = {"method": method}
        with pytest.raises(ValueError, match="'x' must not contain NaN"):
            factory(scores, y=y, **kwargs)

    @pytest.mark.parametrize("method", ["khiops", "eq-freq", "eq-width"])
    @pytest.mark.parametrize("supervised", [False, True])
    @pytest.mark.parametrize("value", [-np.inf, np.inf])
    def test_from_data_rejects_infinite_scores(self, method, supervised, value):
        x = [0.1, value]
        y = [0, 1] if supervised else None
        with pytest.raises(ValueError, match="'x' must not contain infinite values"):
            Histogram.from_data(x, y=y, method=method)

    @pytest.mark.parametrize("dtype", [float, object])
    @pytest.mark.parametrize("value", [-np.inf, np.inf])
    def test_manual_histogram_rejects_infinite_scores(self, dtype, value):
        x = np.array([0.1, value], dtype=dtype)
        with pytest.raises(ValueError, match="'x' must not contain infinite values"):
            Histogram.from_data_and_breakpoints(x, [0.0, 0.5, 1.0], [0, 1])

    def test_missing_full_information_rate_selects_last_histogram(self, monkeypatch):
        # Capture the last candidate from a real report with information rates [0, 0]
        read_report = khalib_main.kh.read_analysis_results_file
        candidates = []

        def capture_report(path):
            report = read_report(path)
            histograms = report.preparation_report.variables_statistics[
                0
            ].modl_histograms
            assert histograms.information_rates == [0, 0]
            candidates.extend(histograms.histograms)
            return report

        monkeypatch.setattr(
            khalib_main.kh, "read_analysis_results_file", capture_report
        )
        histogram = Histogram.from_data(
            np.repeat([0.1, 0.9], 50), method="khiops", max_bins=10
        )
        assert histogram.breakpoints == candidates[-1].bounds
        assert histogram.freqs == candidates[-1].frequencies

    @pytest.mark.parametrize("method", ["khiops", "eq-freq", "eq-width", "manual"])
    @pytest.mark.parametrize(
        ("x", "y"),
        [([0.1], [0, 1]), ([0.1, 0.9], [0]), ([0.1, 0.5, 0.9], [0, 1])],
    )
    def test_rejects_mismatched_sample_counts(self, method, x, y):
        if method == "manual":
            factory = Histogram.from_data_and_breakpoints
            kwargs = {"breakpoints": [0.0, 1.0]}
        else:
            factory = Histogram.from_data
            kwargs = {"method": method}
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            factory(x, y=y, **kwargs)

    @pytest.mark.parametrize("constant_scores", [False, True])
    def test_numeric_multiclass_target_order(self, constant_scores):
        # Use labels whose numeric and alphanumeric orders differ
        y = np.repeat([2, 3, 10], [30, 50, 80])
        x = (
            np.full(len(y), 0.5)
            if constant_scores
            else np.repeat([0.1, 0.5, 0.9], [30, 50, 80])
        )
        histogram = Histogram.from_data(x, y)

        # Compare report statistics with counts computed in numeric class order
        expected = Histogram.from_data_and_breakpoints(x, histogram.breakpoints, y)
        assert histogram.classes == [2, 3, 10]
        assert histogram == expected

    @pytest.mark.parametrize("method", ["khiops", "eq-freq", "eq-width", "manual"])
    @pytest.mark.parametrize(
        "input_type", ["list", "array", "series", "column", "object", "object-column"]
    )
    @pytest.mark.parametrize("use_y", [False, True])
    def test_array_like_inputs(self, method, input_type, use_y):
        # Compare supported representations with a one-dimensional array
        x = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        y = np.array(["a", "a", "b", "a", "b", "b"])
        if method == "manual":
            factory = Histogram.from_data_and_breakpoints
            kwargs = {"breakpoints": [0.0, 0.3, 0.7, 1.0]}
        else:
            factory = Histogram.from_data
            kwargs = {"method": method}
        expected = factory(x, y=y if use_y else None, **kwargs)
        if input_type == "list":
            x, y = x.tolist(), y.tolist()
        elif input_type == "series":
            x, y = pd.Series(x), pd.Series(y)
        elif input_type == "column":
            x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        elif input_type == "object":
            x = x.astype(object)
        elif input_type == "object-column":
            x = x.astype(object).reshape(-1, 1)

        # Check frequencies, target statistics and class labels remain unchanged
        assert factory(x, y=y if use_y else None, **kwargs) == expected

    @pytest.mark.parametrize("manual", [False, True])
    @pytest.mark.parametrize("argument", ["x", "y"])
    @pytest.mark.parametrize(
        "invalid",
        [np.array(0.5), np.empty((2, 0)), np.ones((2, 2)), np.ones((2, 1, 1))],
    )
    def test_invalid_input_shapes(self, manual, argument, invalid):
        inputs = {"x": np.array([0.2, 0.8]), "y": np.array([0, 1])}
        inputs[argument] = invalid
        factory = Histogram.from_data
        if manual:
            factory = Histogram.from_data_and_breakpoints
            inputs["breakpoints"] = [0.0, 0.5, 1.0]
        with pytest.raises(ValueError, match=f"{argument} must be 1-D"):
            factory(**inputs)

    all_cases = [
        ("eq-freq", "bool", True),
        ("eq-freq", "float", True),
        ("eq-freq", "int", False),
        ("eq-freq", "int", True),
        ("eq-freq", "intnl", True),
        ("eq-width", "bool", True),
        ("eq-width", "float", True),
        ("eq-width", "int", False),
        ("eq-width", "int", True),
        ("eq-width", "intnl", True),
        ("khiops", "bool", True),
        ("khiops", "float", True),
        ("khiops", "int", False),
        ("khiops", "int", True),
        ("khiops", "intnl", True),
    ]

    @pytest.mark.parametrize(("method", "y_fixture", "use_y"), all_cases)
    def test_happy_path(
        self, y_fixtures, y_scores_fixtures, ref_histogram, method, y_fixture, use_y
    ):
        for y_scores_fixture in ["original", "original-fake-2d"]:
            # Prepare the input data for the histogram
            y = y_fixtures[y_fixture] if use_y else None
            y_scores = y_scores_fixtures[y_scores_fixture]

            # Compute the histogram with the test settings, check it against the
            # reference
            histogram = Histogram.from_data(y_scores, y=y, method=method)
            assert histogram == ref_histogram, f"Fixture '{y_scores_fixture}'"

    @pytest.mark.parametrize(("method", "y_fixture", "use_y"), all_cases)
    def test_single_value_score(
        self, y_fixtures, y_scores_fixtures, ref_histogram, method, y_fixture, use_y
    ):
        # Prepare the input data for the histogram
        y = y_fixtures[y_fixture] if use_y else None
        y_scores = y_scores_fixtures["constant"]

        # Compute the histogram with the test settings, check it against the reference
        histogram = Histogram.from_data(y_scores, y=y, method=method)
        assert histogram == ref_histogram

    @pytest.mark.parametrize("method", ["khiops", "eq-freq", "eq-width"])
    @pytest.mark.parametrize("value", [-1.0e8, 1.0e8])
    @pytest.mark.parametrize("supervised", [False, True])
    def test_large_constant_scores(self, method, value, supervised):
        # Large constants need a representable interval with unchanged statistics
        x = np.full(4, value)
        y = np.array([0, 0, 0, 1]) if supervised else None
        histogram = Histogram.from_data(x, y, method=method)
        lower, upper = histogram.breakpoints
        assert np.isfinite([lower, upper]).all()
        assert lower < upper
        assert lower <= value <= upper
        assert histogram.freqs == [4]
        np.testing.assert_array_equal(histogram.vfind(x), [0, 0, 0, 0])
        if supervised:
            assert histogram.classes == [0, 1]
            assert histogram.target_freqs == [(3, 1)]
            np.testing.assert_allclose(histogram.target_probas, [[0.75, 0.25]])

    @pytest.mark.parametrize("method", ["khiops", "eq-freq", "eq-width"])
    def test_constant_zero_density_is_capped(self, method):
        histogram = Histogram.from_data([0.0, 0.0], [0, 1], method=method)
        assert histogram.densities == [float(np.finfo(float).max)]
        assert histogram.breakpoints == [0.0, float(np.nextafter(0.0, np.inf))]
        assert histogram.freqs == [2]
        assert histogram.target_probas == [(0.5, 0.5)]

    def test_only_overflowing_densities_are_capped(self):
        # Keep empty and ordinary bins unchanged beside an overflowing density
        tiny = float(np.nextafter(0.0, np.inf))
        histogram = Histogram(breakpoints=[0.0, tiny, 2 * tiny, 1.0], freqs=[1, 0, 1])
        assert histogram.densities == [float(np.finfo(float).max), 0.0, 0.5]

    @pytest.mark.parametrize(
        ("left", "right"),
        [(-1e308, 1e308), (-1e308, 1.7e308), (-1.7e308, 1e308)],
    )
    @pytest.mark.parametrize("dtype", [float, np.float64])
    def test_overflowing_bin_width_preserves_density(self, left, right, dtype):
        histogram = Histogram([dtype(left), dtype(right)], [1])
        expected = 0.5 / (right / 2 - left / 2)
        assert histogram.densities[0] > 0
        assert histogram.densities[0] == pytest.approx(expected, rel=1e-12, abs=0)

    def test_overflowing_width_with_empty_and_ordinary_bins(self):
        histogram = Histogram([-1e308, 1e308, 1.1e308, 1.2e308], [1, 0, 1])
        assert histogram.densities == pytest.approx(
            [0.25 / 1e308, 0.0, 0.5 / (1.2e308 - 1.1e308)],
            rel=1e-12,
            abs=0,
        )

    def test_constant_report_at_float_limit(self, monkeypatch):
        # Exercise the float limit independently of Khiops' narrower numeric range
        read_report = khalib_main.kh.read_analysis_results_file
        value = float(np.finfo(float).max)

        def read_report_at_float_limit(path):
            report = read_report(path)
            stats = report.preparation_report.variables_statistics[0]
            stats.min = stats.max = value
            return report

        monkeypatch.setattr(
            khalib_main.kh, "read_analysis_results_file", read_report_at_float_limit
        )
        histogram = Histogram.from_data([0.5, 0.5], method="eq-width")
        lower, upper = histogram.breakpoints
        assert np.isfinite([lower, upper]).all()
        assert lower < upper == value
        assert histogram.freqs == [2]

    @pytest.mark.parametrize("method", ["eq-freq", "eq-width", "khiops"])
    def test_no_info_target(self, y_fixtures, y_scores_fixtures, ref_histogram, method):
        # Prepare the input data for the histogram
        y = y_fixtures["random"]
        y_scores = y_scores_fixtures["original"]

        # Compute the histogram with the test settings, check it against the reference
        histogram = Histogram.from_data(y_scores, y=y, method=method)
        assert histogram == ref_histogram

    @pytest.mark.parametrize(
        "value", [float("nan"), np.float32("nan"), np.float64("nan"), pd.NA, None]
    )
    def test_find_rejects_missing_values(self, value):
        histogram = Histogram([0.0, 0.5, 1.0], [1, 1])
        with pytest.raises(
            ValueError, match="'value' must not contain NaN or missing values"
        ):
            histogram.find(value)

    def test_find_rejects_boolean_values(self):
        histogram = Histogram([0.0, 0.5, 1.0], [1, 1])
        with pytest.raises(ValueError, match="'value' must not contain boolean values"):
            histogram.find(True)

    @pytest.mark.parametrize("value", [[0.1], [0.1, 0.9], np.array([[0.1]])])
    def test_find_rejects_nonscalar_values(self, value):
        histogram = Histogram([0.0, 0.5, 1.0], [1, 1])
        with pytest.raises(ValueError, match="'value' must be a scalar"):
            histogram.find(value)

    @pytest.mark.parametrize("dtype", [complex, np.complex64, np.complex128])
    @pytest.mark.parametrize("value", [0.1 + 100j, 0.9 - 100j, 0.5 + 0j])
    def test_find_rejects_complex(self, dtype, value):
        histogram = Histogram([0.0, 0.5, 1.0], [1, 1])
        with pytest.raises(ValueError, match="'value' must contain only real values"):
            histogram.find(dtype(value))

    @pytest.mark.parametrize("value", [-np.inf, np.inf])
    def test_find_rejects_infinities(self, value):
        histogram = Histogram([0.0, 0.5, 1.0], [1, 1])
        with pytest.raises(ValueError, match="'value' must not contain infinite"):
            histogram.find(value)
        with pytest.raises(ValueError, match="'values' must not contain infinite"):
            histogram.vfind([0.1, value, 0.9])

    def test_find_vfind_coherence(self, y_fixtures, y_scores_fixtures):
        # Prepare the input data for the histogram
        y = y_fixtures["int"]
        y_scores = y_scores_fixtures["original"]

        # Create an equal width histogram
        histogram = Histogram.from_data(y_scores, y=y, method="eq-width")

        # Check the coherence for some test points
        test_scores = [i / 10 for i in range(-1, 12)]
        np.testing.assert_array_equal(
            histogram.vfind(test_scores),
            [histogram.find(score) for score in test_scores],
        )

        # Check the coherence for the histogram breakpoints
        np.testing.assert_array_equal(
            histogram.vfind(histogram.breakpoints),
            [histogram.find(score) for score in histogram.breakpoints],
        )

    def test_find_vfind_boundary_coherence(self):
        """Test find/vfind agree on all boundary edge cases"""
        histogram = Histogram(
            breakpoints=[0.0, 0.3, 0.7, 1.0],
            freqs=[10, 20, 15],
        )
        # Test: below min, at each breakpoint, between bins, above max
        edge_values = [-0.5, 0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0, 1.5]
        expected_bins = [0, 0, 0, 0, 1, 1, 2, 2, 2]
        np.testing.assert_array_equal(
            histogram.vfind(edge_values),
            expected_bins,
        )
        assert [histogram.find(value) for value in edge_values] == expected_bins

    def test_exact_breakpoints_match_histogram_construction(self):
        x = np.array([0.3, 0.7])
        y = np.array([0, 1])
        histogram = Histogram.from_data_and_breakpoints(
            x, breakpoints=[0.0, 0.3, 0.7, 1.0], y=y
        )

        assert histogram.freqs == [1, 1, 0]
        assert histogram.target_freqs == [(1, 0), (0, 1), (0, 0)]
        np.testing.assert_array_equal(
            calibrate_binary(x, histogram, only_positive=True), [0, 1]
        )

    @pytest.mark.parametrize("y_scores_fixture", ["original", "constant"])
    @pytest.mark.parametrize("method", ["eq-freq", "eq-width", "khiops"])
    def test_manual_vs_khiops_coherence(
        self, y_fixtures, y_scores_fixture, y_scores_fixtures, method
    ):
        # Prepare the input data for the histogram
        y = y_fixtures["int"]
        y_scores = y_scores_fixtures[y_scores_fixture]

        # Obtain the histogram first with Khiops
        histogram_ref = Histogram.from_data(y_scores, y=y, method=method)

        # Rebuild it with the `from_data_and_breakpoints` method and check
        histogram = Histogram.from_data_and_breakpoints(
            y_scores, histogram_ref.breakpoints, y=y
        )
        assert histogram == histogram_ref


def sigmoid_stair(x):
    return (
        0.2 / (1 + np.exp(-400 * (x - 0.11)))
        + 0.3 / (1 + np.exp(-400 * (x - 0.19)))
        + 0.1 / (1 + np.exp(-400 * (x - 0.39)))
        + 0.25 / (1 + np.exp(-400 * (x - 0.75)))
        + 0.15 / (1 + np.exp(-400 * (x - 0.91)))
    )


def sine_cube_root(x):
    return 0.5 * (np.power(x, 1 / 3) * (1 + np.cos(16 * np.pi * x)))


def identity(x):
    return x


@pytest.fixture(name="distorted_scores")
def fixture_distorted_scores():
    # Create the random scores
    rng = np.random.default_rng(seed=1234567)
    y_scores = rng.beta(a=0.4, b=0.6, size=100_000)
    y_scores_fixtures = {
        "1d": y_scores,
        "2d": np.transpose(np.vstack([1 - y_scores, y_scores])),
        "fake-2d": np.array(y_scores).reshape(y_scores.shape[0], 1),
    }
    # Build the fixture data dictionary
    data = {}
    data["y_scores_fixtures"] = y_scores_fixtures
    for distortion_fun in (sigmoid_stair, sine_cube_root, identity):
        # Sample a binary distribution according to the distorted probabilities
        distortion_data = {}
        y_cond_probas = distortion_fun(y_scores)
        y = rng.binomial(1, y_cond_probas)

        # Store the distorted sample and its variants
        distortion_data["y_fixtures"] = {
            "int": (y, False),
            "str": (np.where(y == 0, "zero", "one"), True),
        }

        # Store the expecte ECE and error margin for each distortion function
        # Note: These were calculated with a grid of 1e-8 intervals
        if distortion_fun.__name__ == "sine_cube_root":
            distortion_data |= {
                "theoretical_ece": 0.2381850813972994,
                "approx_kwargs": {"abs": 1.0e-3},
            }
        elif distortion_fun.__name__ == "sigmoid_stair":
            distortion_data |= {
                "theoretical_ece": 0.07390433812747241,
                "approx_kwargs": {"abs": 1.0e-3},
            }
        else:
            assert distortion_fun.__name__ == "identity"
            distortion_data |= {"theoretical_ece": 0, "approx_kwargs": {"abs": 2.5e-2}}
        data[distortion_fun.__name__] = distortion_data

    return data


class TestECE:
    @pytest.mark.parametrize("multi_class_method", ["top-label", "classwise"])
    @pytest.mark.parametrize("supply_histogram", [False, True])
    def test_rejects_reused_statistics_for_multiclass_scores(
        self, multi_class_method, supply_histogram
    ):
        scores = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        histogram = Histogram(breakpoints=[0.0, 1.0], freqs=[3])
        with pytest.raises(
            ValueError, match="'reuse_stats' is not supported for multiclass scores"
        ):
            calibration_error(
                scores,
                [0, 1, 2],
                multi_class_method=multi_class_method,
                histogram=histogram if supply_histogram else None,
                reuse_stats=True,
            )

    @pytest.mark.parametrize("method", ["bin", "label-bin"])
    @pytest.mark.parametrize("labels", [[0, 1], ["no", "yes"]])
    def test_rejects_scores_in_empty_bins(self, method, labels):
        histogram = Histogram(
            breakpoints=[0.0, 0.3, 0.7, 1.0],
            freqs=[1, 0, 1],
            target_freqs=[(1, 0), (0, 0), (0, 1)],
            classes=[0, 1],
        )
        with pytest.raises(
            ValueError,
            match=r"sample indices \[1\], bin indices \[1\]",
        ):
            calibration_error(
                [0.1, 0.5], labels, method=method, histogram=histogram, reuse_stats=True
            )

    @pytest.mark.parametrize("method", ["bin", "label-bin"])
    @pytest.mark.parametrize("reuse_stats", [False, True])
    def test_rejects_empty_supplied_bins_before_ece_refresh(self, method, reuse_stats):
        histogram = Histogram(
            breakpoints=[0.0, 0.3, 0.7, 1.0],
            freqs=[1, 0, 1],
            target_freqs=[(1, 0), (0, 0), (0, 1)],
            classes=[0, 1],
        )
        with pytest.raises(
            ValueError,
            match=r"sample indices \[1\], bin indices \[1\]",
        ):
            calibration_error(
                [0.1, 0.5],
                [0, 1],
                method=method,
                histogram=histogram,
                reuse_stats=reuse_stats,
            )

    @pytest.mark.parametrize("method", ["bin", "label-bin"])
    def test_large_integer_scores_do_not_overflow_accumulation(self, method):
        score = 2**62
        histogram = Histogram([score - 1, score + 1], [5])
        assert calibration_error(
            np.full(5, score, dtype=np.int64),
            [0, 1, 0, 1, 0],
            method=method,
            histogram=histogram,
        ) == pytest.approx(float(score))

    @pytest.mark.parametrize("parameter", ["method", "multi_class_method"])
    @pytest.mark.parametrize("invalid", ["typo", "", None])
    @pytest.mark.parametrize(
        ("scores", "labels"),
        [
            ([0.1, 0.9], [0, 1]),
            ([[0.4, 0.4, 0.2], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1]], [0, 1, 2]),
        ],
    )
    @pytest.mark.parametrize("supply_histogram", [False, True])
    def test_rejects_invalid_estimation_methods(
        self, parameter, invalid, scores, labels, supply_histogram
    ):
        histogram = Histogram(breakpoints=[0.0, 1.0], freqs=[2])
        with pytest.raises(ValueError, match=rf"'{parameter}' must be"):
            calibration_error(
                scores,
                labels,
                histogram=histogram if supply_histogram else None,
                **{parameter: invalid},
            )

    @pytest.mark.parametrize("method", ["bin", "label-bin"])
    @pytest.mark.parametrize("classes", [[], [0, 1]])
    def test_rejects_reused_histogram_without_target_statistics(self, method, classes):
        histogram = Histogram(breakpoints=[0.0, 1.0], freqs=[2], classes=classes)
        with pytest.raises(ValueError, match="'histogram' must have target statistics"):
            calibration_error(
                [0.1, 0.9], [0, 1], method=method, histogram=histogram, reuse_stats=True
            )

    @pytest.mark.parametrize("method", ["bin", "label-bin"])
    @pytest.mark.parametrize("classes", [[], [0, 1]])
    def test_rebuilds_missing_histogram_target_statistics(self, method, classes):
        histogram = Histogram(breakpoints=[0.0, 1.0], freqs=[2], classes=classes)
        expected = 0.0 if method == "bin" else 0.4
        assert calibration_error(
            [0.1, 0.9], [0, 1], method=method, histogram=histogram
        ) == pytest.approx(expected)
        assert histogram.target_freqs == []

    @pytest.mark.parametrize("method", ["bin", "label-bin"])
    @pytest.mark.parametrize("multi_class_method", ["top-label", "classwise"])
    @pytest.mark.parametrize("reuse_stats", [False, True])
    @pytest.mark.parametrize(
        ("scores", "labels"),
        [
            ([np.nan, 0.9], [0, 1]),
            ([[np.nan], [0.9]], [0, 1]),
            ([[np.nan, 0.1], [0.1, 0.9]], [0, 1]),
            ([[0.8, 0.1, np.nan], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], [0, 1, 2]),
        ],
    )
    @pytest.mark.parametrize("dtype", [float, object])
    def test_rejects_nan_scores(
        self, method, multi_class_method, reuse_stats, scores, labels, dtype
    ):
        # Neither unused score columns nor reused statistics may hide NaN inputs
        scores = np.array(scores, dtype=dtype)
        histogram = Histogram(
            breakpoints=[0.0, 1.0], freqs=[2], target_freqs=[(1, 1)], classes=[0, 1]
        )
        with pytest.raises(ValueError, match="'y_scores' must not contain NaN"):
            calibration_error(
                scores,
                labels,
                method=method,
                multi_class_method=multi_class_method,
                histogram=histogram,
                reuse_stats=reuse_stats,
            )

    @pytest.mark.parametrize(
        "scores",
        [
            [-np.inf, 0.9],
            [[0.9, -np.inf, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
        ],
    )
    def test_rejects_infinite_scores(self, scores):
        labels = [0, 1] if np.asarray(scores).ndim == 1 else [0, 1, 2]
        with pytest.raises(ValueError, match="'y_scores' must not contain infinite"):
            calibration_error(scores, labels)

    @pytest.mark.parametrize("method", ["bin", "label-bin"])
    @pytest.mark.parametrize("multi_class_method", ["top-label", "classwise"])
    @pytest.mark.parametrize(
        ("scores", "labels", "reuse_stats"),
        [
            ([0.1, 0.9], [0, 1], False),
            ([0.1, 0.9], [0, 1], True),
            ([[0.1], [0.9]], [0, 1], False),
            ([[0.1], [0.9]], [0, 1], True),
            ([[0.9, 0.1], [0.1, 0.9]], [0, 1], False),
            ([[0.9, 0.1], [0.1, 0.9]], [0, 1], True),
            ([[0.4, 0.4, 0.2], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1]], [0, 1, 2], False),
        ],
    )
    def test_numeric_object_scores(
        self, method, multi_class_method, reuse_stats, scores, labels
    ):
        # Object scores must behave like floating-point scores in every ECE mode
        histogram = Histogram(
            breakpoints=[0.0, 1.0], freqs=[2], target_freqs=[(1, 1)], classes=[0, 1]
        )
        kwargs = {
            "method": method,
            "multi_class_method": multi_class_method,
            "histogram": histogram,
            "reuse_stats": reuse_stats,
        }
        expected = calibration_error(np.array(scores, dtype=float), labels, **kwargs)
        assert calibration_error(
            np.array(scores, dtype=object), labels, **kwargs
        ) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("method", "expected"), [("bin", 1 / 30), ("label-bin", 13 / 90)]
    )
    @pytest.mark.parametrize("labels", [[0, 1, 2], ["bird", "cat", "dog"]])
    def test_top_label_ties_select_first_class(self, method, expected, labels):
        # The tied first row is correct only when the first maximum is selected
        scores = [[0.4, 0.4, 0.2], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]
        histogram = Histogram(breakpoints=[0.0, 1.0], freqs=[3])
        assert calibration_error(
            scores, labels, method=method, histogram=histogram
        ) == pytest.approx(expected)

    @pytest.mark.parametrize("multi_class_method", ["top-label", "classwise"])
    @pytest.mark.parametrize("reuse_stats", [False, True])
    @pytest.mark.parametrize(
        ("scores", "y"),
        [
            ([0.1], [0, 1]),
            ([0.1, 0.9], [0]),
            ([[0.9, 0.1]], [0, 1]),
            ([[0.1, 0.2, 0.7]], [0, 1, 2]),
            ([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]], [0]),
            ([0.1, 0.5, 0.9], [0, 1]),
        ],
    )
    def test_rejects_mismatched_sample_counts(
        self, scores, y, multi_class_method, reuse_stats
    ):
        # Supplied statistics must not bypass sample-count validation
        histogram = Histogram(
            breakpoints=[0.0, 1.0],
            freqs=[2],
            target_freqs=[(1, 1)],
            classes=[0, 1],
        )
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            calibration_error(
                scores,
                y,
                multi_class_method=multi_class_method,
                histogram=histogram,
                reuse_stats=reuse_stats,
            )

    @pytest.mark.parametrize("method", ["bin", "label-bin"])
    @pytest.mark.parametrize("multi_class_method", ["top-label", "classwise"])
    @pytest.mark.parametrize(
        "y",
        [
            [0, 0, 0, 0],
            [0, 2, 0, 2],
            ["cat", "dog", "cat", "dog"],
            [0, 1, 2, 3],
        ],
    )
    def test_multiclass_rejects_mismatched_classes(self, method, multi_class_method, y):
        # Reject missing or extra labels before mapping score columns to classes
        scores = np.array(
            [[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.3, 0.3, 0.4]]
        )
        with pytest.raises(
            ValueError,
            match="Number of distinct labels in 'y' must match.*"
            rf"{len(set(y))} != 3",
        ):
            calibration_error(
                scores, y, method=method, multi_class_method=multi_class_method
            )

    @pytest.mark.parametrize("method", ["bin", "label-bin"])
    @pytest.mark.parametrize("all_correct", [False, True])
    @pytest.mark.parametrize("histogram_source", ["internal", "boundaries"])
    def test_top_label_rejects_single_correctness_outcome(
        self, method, all_correct, histogram_source
    ):
        # Vary confidence while keeping every prediction correct or incorrect
        scores = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6]])
        y = np.array([0, 1, 2]) if all_correct else np.array([1, 2, 0])
        confidence = scores.max(axis=1)
        correctness = np.full(3, int(all_correct))
        histogram = None
        if histogram_source == "boundaries":
            histogram = Histogram.from_data_and_breakpoints(
                confidence, [0.0, 0.65, 0.75, 1.0], correctness
            )

        # Reject single-outcome data regardless of the histogram source
        with pytest.raises(
            ValueError, match="Top-label ECE requires both correct and incorrect"
        ):
            calibration_error(
                scores,
                y,
                method=method,
                histogram=histogram,
            )

    @pytest.mark.parametrize(
        "scores", [[0.8, 0.9], [[0.8], [0.9]], [[0.2, 0.8], [0.1, 0.9]]]
    )
    @pytest.mark.parametrize("reuse_stats", [False, True])
    def test_binary_rejects_single_class_target(self, scores, reuse_stats):
        # A supplied two-class histogram must not bypass target validation
        histogram = None
        if reuse_stats:
            histogram = Histogram(
                breakpoints=[0.0, 1.0],
                freqs=[2],
                target_freqs=[(1, 1)],
                classes=["cat", "dog"],
            )
        with pytest.raises(ValueError, match="Binary ECE requires exactly two"):
            calibration_error(
                np.array(scores),
                np.array(["cat", "cat"]),
                histogram=histogram,
                reuse_stats=reuse_stats,
            )

    @pytest.mark.parametrize("method", ["bin", "label-bin"])
    def test_reused_binary_histogram_requires_matching_class_order(self, method):
        histogram = Histogram(
            breakpoints=[0.0, 1.0],
            freqs=[3],
            target_freqs=[(2, 1)],
            classes=[1, 0],
        )
        with pytest.raises(
            ValueError, match="Reused binary histogram classes must match"
        ):
            calibration_error(
                [0.1, 0.3, 0.8],
                [0, 1, 1],
                method=method,
                histogram=histogram,
                reuse_stats=True,
            )

        assert calibration_error(
            [0.1, 0.3, 0.8], [0, 1, 1], method=method, histogram=histogram
        ) == pytest.approx(
            0.2666666666666666 if method == "bin" else 0.3555555555555556
        )

    def test_classwise_target_error_has_context(self, monkeypatch):
        # Simulate a one-vs-rest target that lacks one outcome
        monkeypatch.setattr(
            khalib_main, "label_binarize", lambda *args, **kwargs: np.zeros((3, 3))
        )
        with pytest.raises(
            ValueError, match="Classwise ECE requires both positive and negative"
        ):
            calibration_error(
                np.eye(3), np.array(["a", "b", "c"]), multi_class_method="classwise"
            )

    @pytest.mark.parametrize("method", ["bin", "label-bin"])
    @pytest.mark.parametrize("as_list", [False, True])
    def test_binary_vector_inputs(self, method, as_list):
        # Compare list and single-column inputs with vector inputs
        scores = np.array([0.1, 0.9])
        y = np.array([0, 1])
        histogram = Histogram.from_data_and_breakpoints(scores, [0.0, 1.0], y)
        expected = calibration_error(scores, y, method=method, histogram=histogram)
        scores = scores.tolist() if as_list else scores[:, np.newaxis]
        y = y.tolist() if as_list else y[:, np.newaxis]

        assert calibration_error(
            scores, y, method=method, histogram=histogram
        ) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("method", "expected"), [("bin", 0.1), ("label-bin", 2 / 15)]
    )
    @pytest.mark.parametrize("score_shape", ["1d", "column", "binary"])
    def test_supplied_histogram_uses_current_data(self, method, expected, score_shape):
        # Recompute statistics from current data using the supplied boundaries
        histogram = Histogram(
            breakpoints=[0.0, 0.3, 0.7, 1.0],
            freqs=[10, 10, 20],
            target_freqs=[(0, 10), (0, 10), (20, 0)],
            classes=[0, 1],
        )
        scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        y = np.array([0, 0, 1, 0, 1, 1])
        if score_shape == "column":
            scores = scores.reshape(-1, 1)
        elif score_shape == "binary":
            scores = np.column_stack([1 - scores, scores])

        # Check current-data statistics and preserve the supplied histogram
        assert calibration_error(
            scores, y, method=method, histogram=histogram
        ) == pytest.approx(expected)
        assert histogram.freqs == [10, 10, 20]
        assert histogram.target_freqs == [(0, 10), (0, 10), (20, 0)]

    @pytest.mark.parametrize(
        ("method", "expected"), [("bin", 0.1), ("label-bin", 2 / 15)]
    )
    @pytest.mark.parametrize("supply_histogram", [False, True])
    def test_histogram_statistics_are_not_recomputed(
        self, method, expected, supply_histogram, monkeypatch
    ):
        # Provide same-data statistics through either histogram source
        scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        y = np.array([0, 0, 1, 0, 1, 1])
        histogram = Histogram.from_data_and_breakpoints(scores, [0.0, 0.3, 0.7, 1.0], y)
        monkeypatch.setattr(Histogram, "from_data", lambda *args, **kwargs: histogram)

        # Reject a second statistics pass
        def unexpected_recomputation(*args, **kwargs):
            pytest.fail("Histogram statistics must not be recomputed")

        monkeypatch.setattr(
            Histogram, "from_data_and_breakpoints", unexpected_recomputation
        )

        # Reuse supplied statistics or use the internally built histogram directly
        assert calibration_error(
            scores,
            y,
            method=method,
            histogram=histogram if supply_histogram else None,
            reuse_stats=supply_histogram,
        ) == pytest.approx(expected)

    @pytest.mark.parametrize("y_fixture", ["bool", "float", "int", "intnl", "str"])
    @pytest.mark.parametrize("y_scores_fixture", ["original", "original-2d"])
    @pytest.mark.parametrize(
        ("method", "expected_ece"), [("bin", 0.036162213), ("label-bin", 0.086438357)]
    )
    def test_fast_binary_ece(
        self,
        method,
        expected_ece,
        y_fixture,
        y_fixtures,
        y_scores_fixture,
        y_scores_fixtures,
    ):
        # Prepare the input data for the ECE estimation
        y = y_fixtures[y_fixture]
        y_scores = y_scores_fixtures[y_scores_fixture].copy()
        if is_target_inverted(y_fixture):
            if len(y_scores.shape) == 1:
                y_scores = 1 - y_scores
            else:
                y_scores[:, [0, 1]] = y_scores[:, [1, 0]]

        # Estimate the ECE
        ece = calibration_error(y_scores, y, method=method)
        assert ece == pytest.approx(expected_ece)

    @pytest.mark.parametrize(
        ("multi_class_method", "expected_ece"),
        [("top-label", 0.0723944), ("classwise", 0.04642129)],
    )
    @pytest.mark.parametrize("as_list", [False, True])
    def test_fast_multi_class_ece(
        self, multi_class_method, expected_ece, as_list, vehicles_scores_df
    ):
        # Preserve multiclass score matrices while normalizing target vectors
        scores = vehicles_scores_df.drop("y", axis=1).to_numpy()
        y = vehicles_scores_df.y.to_numpy()
        if as_list:
            scores, y = scores.tolist(), y.tolist()
        else:
            y = y[:, np.newaxis]
        ece = calibration_error(
            scores,
            y,
            multi_class_method=multi_class_method,
        )
        assert ece == pytest.approx(expected_ece)

    @pytest.mark.slow
    @pytest.mark.parametrize("y_fixture_name", ["int", "str"])
    @pytest.mark.parametrize("y_scores_fixture_name", ["1d", "2d", "fake-2d"])
    @pytest.mark.parametrize(
        "distortion_fun_name", ["sine_cube_root", "sigmoid_stair", "identity"]
    )
    @pytest.mark.parametrize(
        ("method", "expected_ece"), [("label-bin", 0)]
    )  # [("bin", 0.036162213), ("label-bin", 0.086438357)]
    # )
    def test_accuracy_binary_ece(
        self,
        method,
        expected_ece,
        distorted_scores,
        distortion_fun_name,
        y_scores_fixture_name,
        y_fixture_name,
    ):
        # Prepare the input data for the ECE estimation
        y_scores = distorted_scores["y_scores_fixtures"][y_scores_fixture_name].copy()
        y, inverted_target = distorted_scores[distortion_fun_name]["y_fixtures"][
            y_fixture_name
        ]
        if inverted_target:
            if len(y_scores.shape) == 1 or y_scores.shape[1] == 1:
                y_scores = 1 - y_scores
            else:
                y_scores[:, [0, 1]] = y_scores[:, [1, 0]]

        # Estimate the ECE
        ece = calibration_error(y_scores, y, method=method)
        assert ece == pytest.approx(
            distorted_scores[distortion_fun_name]["theoretical_ece"],
            **distorted_scores[distortion_fun_name]["approx_kwargs"],
        )


def _clf_data_builder(n_train, n_calib, n_test, n_classes, n_features, n_informative):
    # Note, we build this way to ensure reproducibility for the not_fitted/None case
    x, y = make_classification(
        n_samples=n_train + n_calib + n_test,
        n_classes=n_classes,
        n_features=n_features,
        n_informative=n_informative,
        random_state=1,
        return_X_y=True,
    )
    x_train, x_not_train, y_train, y_not_train = train_test_split(
        x, y, train_size=n_train, random_state=2
    )
    x_calib, x_test, y_calib, y_test = train_test_split(
        x_not_train, y_not_train, train_size=n_calib, random_state=3
    )

    return x_train, x_calib, x_test, y_train, y_calib, y_test


@pytest.fixture(name="clf_data", scope="session")
def fixture_clf_data():
    return {
        "binary": {
            "slow": _clf_data_builder(
                n_train=10_000,
                n_calib=10_000,
                n_test=30_000,
                n_classes=2,
                n_features=10,
                n_informative=2,
            ),
            "fast": _clf_data_builder(
                n_train=1_000,
                n_calib=1_000,
                n_test=1_000,
                n_classes=2,
                n_features=10,
                n_informative=2,
            ),
        },
        "multi_class": {
            "slow": _clf_data_builder(
                n_train=10_000,
                n_calib=10_000,
                n_test=30_000,
                n_classes=4,
                n_features=10,
                n_informative=4,
            ),
            "fast": _clf_data_builder(
                n_train=1_000,
                n_calib=1_000,
                n_test=1_000,
                n_classes=4,
                n_features=10,
                n_informative=4,
            ),
        },
    }


class TestKhalibClassifier:
    @pytest.mark.parametrize("random_state", [True, np.bool_(True)])
    def test_fit_rejects_boolean_random_state(self, random_state):
        x = np.arange(60).reshape(-1, 1)
        y = np.tile([0, 1, 2], 20)
        calibrator = KhalibClassifier(LogisticRegression(), random_state=random_state)
        with pytest.raises(TypeError, match="'random_state' must not be a boolean"):
            calibrator.fit(x, y)
        with pytest.raises(NotFittedError):
            check_is_fitted(calibrator)

    def test_classifier_accepts_sparse_feature_matrices(self):
        x = np.arange(60).reshape(-1, 1)
        y = np.tile([0, 1, 2], 20)
        x_sparse = csr_matrix(x)
        calibrator = KhalibClassifier(LogisticRegression()).fit(x_sparse, y)
        assert calibrator.predict_proba(x_sparse[:5]).shape == (5, 3)

    def test_fit_clones_unfitted_estimator(self):
        x = np.arange(60).reshape(-1, 1)
        y = np.tile([0, 1, 2], 20)
        estimator = LogisticRegression()
        calibrator = KhalibClassifier(estimator, random_state=0).fit(x, y)

        assert not hasattr(estimator, "classes_")
        assert calibrator.fitted_estimator_ is not estimator
        assert calibrator.predict_proba(x[:5]).shape == (5, 3)

    def test_fit_copies_prefitted_estimator(self):
        x = np.arange(60).reshape(-1, 1)
        y = np.tile([0, 1, 2], 20)
        estimator = LogisticRegression().fit(x, y)
        calibrator = KhalibClassifier(estimator).fit(x, y)
        probabilities = calibrator.predict_proba(x[:5])

        estimator.fit(x, np.tile([0, 1], 30))

        assert calibrator.fitted_estimator_ is not estimator
        np.testing.assert_array_equal(calibrator.classes_, [0, 1, 2])
        np.testing.assert_allclose(calibrator.predict_proba(x[:5]), probabilities)

    @pytest.mark.parametrize("missing", [np.nan, pd.NA])
    def test_fit_rejects_missing_labels(self, missing):
        x = np.arange(60).reshape(-1, 1)
        y = np.tile([0, 1, 2], 20).astype(object)
        y[0] = missing
        calibrator = KhalibClassifier(LogisticRegression())
        with pytest.raises(
            ValueError, match="'y' must not contain NaN or missing values"
        ):
            calibrator.fit(x, y)
        with pytest.raises(NotFittedError):
            check_is_fitted(calibrator)

    @pytest.mark.parametrize("parameter", ["X", "y", "train_size", "random_state"])
    @pytest.mark.parametrize("as_object", [False, True])
    def test_fit_rejects_complex_inputs(self, parameter, as_object):
        x = np.arange(60).reshape(-1, 1)
        y = np.tile([0, 1, 2], 20)
        inputs = {"X": x, "y": y, "train_size": 0.75, "random_state": 0}
        value = np.asarray(inputs[parameter], dtype=complex)
        inputs[parameter] = value.astype(object) if as_object else value
        calibrator = KhalibClassifier(
            LogisticRegression(),
            train_size=inputs["train_size"],
            random_state=inputs["random_state"],
        )
        with pytest.raises(
            ValueError, match=rf"'{parameter}' must contain only real values"
        ):
            calibrator.fit(inputs["X"], inputs["y"])
        with pytest.raises(NotFittedError):
            check_is_fitted(calibrator)

    @pytest.mark.parametrize("sparse", [False, True])
    def test_predict_rejects_complex_features(self, sparse):
        x = np.arange(60).reshape(-1, 1)
        y = np.tile([0, 1, 2], 20)
        calibrator = KhalibClassifier(LogisticRegression().fit(x, y)).fit(x, y)
        features = x[:2].astype(complex)
        if sparse:
            features = csr_matrix(features)
        with pytest.raises(ValueError, match="'X' must contain only real values"):
            calibrator.predict_proba(features)

    @pytest.mark.parametrize("n_classes", [2, 3])
    @pytest.mark.parametrize("during_fit", [False, True])
    def test_rejects_complex_estimator_scores(self, monkeypatch, n_classes, during_fit):
        x = np.arange(60).reshape(-1, 1)
        y = np.tile(np.arange(n_classes), 60 // n_classes)
        estimator = LogisticRegression().fit(x, y)
        calibrator = KhalibClassifier(estimator)
        if not during_fit:
            calibrator.fit(x, y)
        target = estimator if during_fit else calibrator.fitted_estimator_
        predict_proba = target.predict_proba

        def complex_proba(features):
            return predict_proba(features).astype(complex)

        monkeypatch.setattr(target, "predict_proba", complex_proba)
        operation = calibrator.fit if during_fit else calibrator.predict_proba
        args = (x, y) if during_fit else (x[:2],)
        with pytest.raises(ValueError, match="'scores' must contain only real values"):
            operation(*args)

    @pytest.mark.parametrize("search_type", [GridSearchCV, RandomizedSearchCV])
    @pytest.mark.parametrize("n_classes", [3, 4])
    @pytest.mark.parametrize("prefitted", [False, True])
    @pytest.mark.parametrize("pipeline_outside", [False, True])
    def test_rejects_search_ovo_decision_scores(
        self, search_type, n_classes, prefitted, pipeline_outside
    ):
        # Exercise delegation in both directions between pipelines and searches
        x = np.arange(120).reshape(-1, 1)
        y = np.tile(np.arange(n_classes), 120 // n_classes)
        estimator = search_type(
            make_pipeline(StandardScaler(), SVC(decision_function_shape="ovo")),
            {"svc__C": [1.0]},
            cv=2,
            **({"n_iter": 1} if search_type is RandomizedSearchCV else {}),
        )
        if pipeline_outside:
            estimator = make_pipeline(StandardScaler(), estimator)
        if prefitted:
            estimator.fit(x, y)
        calibrator = KhalibClassifier(estimator, random_state=0)
        with pytest.raises(ValueError, match='decision_function_shape="ovo"'):
            calibrator.fit(x, y)
        with pytest.raises(NotFittedError):
            check_is_fitted(calibrator)

    @pytest.mark.parametrize("search_type", [GridSearchCV, RandomizedSearchCV])
    @pytest.mark.parametrize(
        ("n_classes", "probability"), [(2, False), (3, True), (4, True)]
    )
    def test_search_ovo_setting_allows_binary_scores_and_probabilities(
        self, search_type, n_classes, probability
    ):
        x = np.arange(120).reshape(-1, 1)
        y = np.tile(np.arange(n_classes), 120 // n_classes)
        estimator = search_type(
            SVC(decision_function_shape="ovo", probability=probability, random_state=0),
            {"C": [1.0]},
            cv=2,
            **({"n_iter": 1} if search_type is RandomizedSearchCV else {}),
        ).fit(x, y)
        calibrator = KhalibClassifier(estimator).fit(x, y)
        probabilities = calibrator.predict_proba(x[:5])
        assert probabilities.shape == (5, n_classes)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    @pytest.mark.parametrize("search_type", [GridSearchCV, RandomizedSearchCV])
    @pytest.mark.parametrize("n_classes", [3, 4])
    def test_predict_uses_snapshot_after_search_mutation(self, search_type, n_classes):
        # Mutating the caller's fitted search does not alter calibrated predictions
        x = np.arange(120).reshape(-1, 1)
        y = np.tile(np.arange(n_classes), 120 // n_classes)
        estimator = search_type(
            SVC(decision_function_shape="ovr"),
            {"C": [1.0]},
            cv=2,
            **({"n_iter": 1} if search_type is RandomizedSearchCV else {}),
        ).fit(x, y)
        calibrator = KhalibClassifier(estimator).fit(x, y)
        probabilities = calibrator.predict_proba(x[:5])
        estimator.best_estimator_.set_params(decision_function_shape="ovo")
        np.testing.assert_allclose(calibrator.predict_proba(x[:5]), probabilities)

    @pytest.mark.parametrize("n_classes", [3, 4])
    @pytest.mark.parametrize("prefitted", [False, True])
    @pytest.mark.parametrize("pipeline_depth", [0, 1, 2])
    def test_rejects_multiclass_ovo_decision_scores(
        self, n_classes, prefitted, pipeline_depth
    ):
        # Three-class pairwise scores have the same shape as classwise scores
        x = np.arange(120).reshape(-1, 1)
        y = np.tile(np.arange(n_classes), 120 // n_classes)
        estimator = SVC(decision_function_shape="ovo")
        for _ in range(pipeline_depth):
            estimator = make_pipeline(StandardScaler(), estimator)
        if prefitted:
            estimator.fit(x, y)
        calibrator = KhalibClassifier(estimator, random_state=0)
        with pytest.raises(ValueError, match='decision_function_shape="ovo"'):
            calibrator.fit(x, y)
        with pytest.raises(NotFittedError):
            check_is_fitted(calibrator)

    @pytest.mark.parametrize(
        ("n_classes", "probability"), [(2, False), (3, True), (4, True)]
    )
    @pytest.mark.parametrize("pipeline_depth", [0, 1, 2])
    def test_ovo_setting_allows_binary_scores_and_probabilities(
        self, n_classes, probability, pipeline_depth
    ):
        # The OVO setting does not affect binary scores or predict_proba columns
        x = np.arange(120).reshape(-1, 1)
        y = np.tile(np.arange(n_classes), 120 // n_classes)
        estimator = SVC(
            decision_function_shape="ovo", probability=probability, random_state=0
        )
        for _ in range(pipeline_depth):
            estimator = make_pipeline(StandardScaler(), estimator)
        estimator.fit(x, y)
        calibrator = KhalibClassifier(estimator).fit(x, y)
        probabilities = calibrator.predict_proba(x[:5])
        assert probabilities.shape == (5, n_classes)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    @pytest.mark.parametrize("n_classes", [3, 4])
    @pytest.mark.parametrize("pipeline_depth", [0, 1, 2])
    def test_predict_uses_snapshot_after_estimator_mutation(
        self, n_classes, pipeline_depth
    ):
        # Mutating the caller's estimator does not alter calibrated predictions
        x = np.arange(60).reshape(-1, 1)
        y = np.repeat(np.arange(n_classes), 60 // n_classes)
        classifier = SVC(decision_function_shape="ovr")
        estimator = classifier
        for _ in range(pipeline_depth):
            estimator = make_pipeline(StandardScaler(), estimator)
        estimator.fit(x, y)
        calibrator = KhalibClassifier(estimator).fit(x, y)
        probabilities = calibrator.predict_proba(x[:5])
        classifier.set_params(decision_function_shape="ovo")
        np.testing.assert_allclose(calibrator.predict_proba(x[:5]), probabilities)

    @pytest.mark.parametrize(
        ("classes", "calibration_labels"),
        [
            ([10, 20], [10, 10]),
            ([10, 20], [20, 20]),
            (["a", "b", "c"], ["a", "c"]),
            (["a", "b", "c"], ["a", "b", "d"]),
        ],
    )
    def test_fit_rejects_mismatched_calibration_classes(
        self, classes, calibration_labels
    ):
        # Use a fitted estimator whose classes differ from the calibration labels
        x = np.arange(60).reshape(-1, 1)
        y = np.repeat(classes, 60 // len(classes))
        estimator = LogisticRegression().fit(x, y)
        calibrator = KhalibClassifier(estimator)
        with pytest.raises(ValueError, match="Calibration labels must.*exactly match"):
            calibrator.fit(x[: len(calibration_labels)], calibration_labels)
        assert not hasattr(calibrator, "histograms_")
        with pytest.raises(NotFittedError):
            check_is_fitted(calibrator)
        with pytest.raises(NotFittedError):
            calibrator.predict_proba(x[:1])

    @pytest.mark.parametrize("n_classes", [2, 3])
    def test_fit_rejects_split_missing_calibration_class(self, n_classes):
        # The fixed split puts the only example of the last class in training
        x = np.arange(12).reshape(-1, 1)
        y = np.zeros(12, dtype=int)
        y[0] = n_classes - 1
        if n_classes == 3:
            y[1:6] = 1
        calibrator = KhalibClassifier(LogisticRegression(), random_state=0)
        with pytest.raises(ValueError, match="Calibration labels must.*exactly match"):
            calibrator.fit(x, y)
        assert not hasattr(calibrator.estimator, "classes_")
        assert not hasattr(calibrator, "histograms_")
        with pytest.raises(NotFittedError):
            check_is_fitted(calibrator)

    @pytest.mark.parametrize("failure", ["labels", "histogram"])
    def test_failed_refit_preserves_state(self, monkeypatch, failure):
        # Retain a working model while fitting a replacement with different labels
        x = np.arange(60).reshape(-1, 1)
        y = np.repeat([0, 1, 2], 20)
        calibrator = KhalibClassifier(LogisticRegression().fit(x, y))
        calibrator.fit(x, y)
        old_estimator = calibrator.fitted_estimator_
        old_histograms = calibrator.histograms_
        old_probas = calibrator.predict_proba(x)
        new_y = np.repeat(["a", "b", "c"], 20)
        calibrator.set_params(estimator=LogisticRegression().fit(x, new_y))

        # Reject calibration labels missing two estimator classes
        if failure == "labels":
            new_y = np.repeat("a", len(x))
            message = "Calibration labels must"
        # Fail after the first class histogram has already been built
        else:
            build_histogram = Histogram.from_data
            calls = 0

            def failing_histogram(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ValueError("Histogram construction failed")
                return build_histogram(*args, **kwargs)

            monkeypatch.setattr(Histogram, "from_data", failing_histogram)
            message = "Histogram construction failed"
        with pytest.raises(ValueError, match=message):
            calibrator.fit(x, new_y)

        # Preserve the previous model's calibration and class mapping
        assert calibrator.fitted_estimator_ is old_estimator
        assert calibrator.histograms_ is old_histograms
        np.testing.assert_allclose(calibrator.predict_proba(x), old_probas)
        np.testing.assert_array_equal(calibrator.classes_, [0, 1, 2])

    @pytest.mark.parametrize("failure", ["labels", "histogram"])
    def test_failed_first_fit_remains_unfitted(self, monkeypatch, failure):
        # Start with a fitted base estimator but no calibration state
        x = np.arange(60).reshape(-1, 1)
        y = np.repeat(["a", "b", "c"], 20)
        calibrator = KhalibClassifier(LogisticRegression().fit(x, y))

        # Reject calibration labels missing two estimator classes
        if failure == "labels":
            y = np.repeat("a", len(x))
            message = "Calibration labels must"
        # Fail after the first class histogram has already been built
        else:
            build_histogram = Histogram.from_data
            calls = 0

            def failing_histogram(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ValueError("Histogram construction failed")
                return build_histogram(*args, **kwargs)

            monkeypatch.setattr(Histogram, "from_data", failing_histogram)
            message = "Histogram construction failed"
        with pytest.raises(ValueError, match=message):
            calibrator.fit(x, y)

        # No partial fitted state is exposed after the failure
        with pytest.raises(NotFittedError):
            check_is_fitted(calibrator)
        with pytest.raises(NotFittedError):
            calibrator.predict_proba(x)
        with pytest.raises(NotFittedError):
            _ = calibrator.classes_

    @pytest.mark.parametrize("classes", [[10, 20], ["a", "b", "c"]])
    def test_predict_uses_calibrated_probabilities(self, classes):
        # Fit the calibrator with non-positional class labels
        x = np.arange(60).reshape(-1, 1)
        y = np.repeat(classes, 60 // len(classes))
        estimator = LogisticRegression().fit(x, y)
        calibrator = KhalibClassifier(estimator).fit(x, y)
        np.testing.assert_array_equal(calibrator.classes_, estimator.classes_)
        assert calibrator.classes_ is not estimator.classes_

        # Force calibration to favor the last class
        n_histograms = 1 if len(classes) == 2 else len(classes)
        calibrator.histograms_ = [
            Histogram(
                breakpoints=[0.0, 1.0],
                freqs=[1],
                target_freqs=[(0, 1) if k == n_histograms - 1 else (1, 0)],
                classes=[0, 1],
            )
            for k in range(n_histograms)
        ]

        # Check class mapping and sklearn accuracy interfaces
        expected = np.repeat(classes[-1], 3)
        assert estimator.predict(x[:3])[0] != expected[0]
        np.testing.assert_array_equal(calibrator.predict(x[:3]), expected)
        assert calibrator.score(x[:3], expected) == 1
        assert get_scorer("accuracy")(calibrator, x[:3], expected) == 1

    def test_predict_requires_fit(self):
        with pytest.raises(NotFittedError):
            KhalibClassifier().predict(np.array([[0.0]]))

    def test_classes_requires_fit(self):
        with pytest.raises(NotFittedError):
            _ = KhalibClassifier().classes_

    def test_multiclass_zero_calibrated_probabilities_fall_back(self):
        # Prepare a fitted estimator and zero-positive calibration bins
        x = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 1, 2])
        estimator = LogisticRegression().fit(x, y)
        x_test = np.array([[0.5], [1.5]])
        calibrator = KhalibClassifier()
        calibrator.fitted_estimator_ = estimator
        histogram = Histogram(
            breakpoints=[0.0, 1.0], freqs=[1], target_freqs=[(1, 0)], classes=[0, 1]
        )
        calibrator.histograms_ = [histogram] * 3

        # Check the warning and fallback probabilities
        with pytest.warns(UserWarning, match=r"samples \[0, 1\]"):
            probabilities = calibrator.predict_proba(x_test)
        np.testing.assert_allclose(probabilities, estimator.predict_proba(x_test))

    def test_multiclass_zero_calibration_requires_predict_proba(self):
        # Prepare a decision-only estimator and zero-positive calibration bins
        x = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1, 2, 2])
        estimator = SVC().fit(x, y)
        calibrator = KhalibClassifier()
        calibrator.fitted_estimator_ = estimator
        histogram = Histogram(
            breakpoints=[0.0, 1.0], freqs=[1], target_freqs=[(1, 0)], classes=[0, 1]
        )
        calibrator.histograms_ = [histogram] * 3

        # Reject decision scores as fallback probabilities
        with pytest.raises(ValueError, match="does not provide predict_proba"):
            calibrator.predict_proba(np.array([[-1.5], [1.5]]))

    @pytest.mark.filterwarnings("ignore:'force_all_finite' was renamed")
    # Notes:
    # - SVC does not have predict_proba
    # - MC LR OVR+normalization worsens the calibration
    @pytest.mark.parametrize(
        ("clf", "class_mode", "expected_ece"),
        [
            (KhiopsClassifier(), "binary", 0.005826),
            (KhiopsClassifier(), "multi_class", 0.017015),
            (LogisticRegression(), "binary", 0.009168),
            (LogisticRegression(), "multi_class", 0.103714),
            (SVC(), "binary", 0.006882),
            (SVC(), "multi_class", 0.052011),
            (None, "binary", 0.005826),
            (None, "multi_class", 0.017015),
        ],
    )
    @pytest.mark.parametrize(
        ("train_mode", "train_size"),
        [pytest.param("slow", 10000, marks=pytest.mark.slow), ("fast", 1000)],
    )
    @pytest.mark.parametrize("train_clf", [False, True])
    def test_calibrator(
        self, train_clf, train_mode, train_size, clf, class_mode, expected_ece, clf_data
    ):
        # Explose the input data to variables
        x_train, x_calib, x_test, y_train, y_calib, y_test = clf_data[class_mode][
            train_mode
        ]

        # Create the calibrated classifier
        train_calib_split_state = 2
        if clf is not None and train_clf:
            # Control the train/calib split: This is only for reproducibility of the
            # case where the classifier is fitted or None versos the pre-fitted case
            x_not_test = np.vstack([x_train, x_calib])
            y_not_test = np.hstack([y_train, y_calib])
            x_train, x_calib, y_train, y_calib = train_test_split(
                x_not_test,
                y_not_test,
                test_size=x_train.shape[0],
                random_state=train_calib_split_state,
            )

            # Fit the classifier and calibrator
            clf.fit(x_train, y_train)
            cclf = KhalibClassifier(clf)
            cclf.fit(x_calib, y_calib)
        else:
            cclf = KhalibClassifier(
                clf, train_size=train_size, random_state=train_calib_split_state
            )
            cclf.fit(np.vstack([x_train, x_calib]), np.hstack([y_train, y_calib]))
        check_is_fitted(cclf)

        # We check the ECE only in "slow" mode because the train data is large enough
        # The "fast" mode is only to test the correct execution
        if train_mode == "slow":
            # Check the resulting ECE is in the nominal range
            y_scores_calib_test = cclf.predict_proba(x_test)
            ece = calibration_error(
                y_scores_calib_test, y_test, multi_class_method="top-label"
            )
            assert ece == pytest.approx(expected_ece, rel=1e-2)


class TestCalibrateBinary:
    @pytest.mark.parametrize("only_positive", ["yes", 1, None])
    def test_rejects_nonboolean_only_positive(self, only_positive):
        histogram = Histogram(
            breakpoints=[0.0, 1.0], freqs=[2], target_freqs=[(1, 1)], classes=[0, 1]
        )
        with pytest.raises(TypeError, match="'only_positive' must be a boolean"):
            calibrate_binary([0.1, 0.9], histogram, only_positive=only_positive)

    @pytest.mark.parametrize("only_positive", [False, True])
    @pytest.mark.parametrize("scores", [[], np.empty((0, 1))])
    def test_empty_scores_return_empty_probabilities(self, only_positive, scores):
        histogram = Histogram(
            breakpoints=[0.0, 1.0], freqs=[2], target_freqs=[(1, 1)], classes=[0, 1]
        )
        probabilities = calibrate_binary(scores, histogram, only_positive=only_positive)
        assert probabilities.shape == ((0,) if only_positive else (0, 2))

    @pytest.mark.parametrize("only_positive", [False, True])
    @pytest.mark.parametrize("classes", [[], [0, 1]])
    def test_rejects_missing_target_statistics(self, only_positive, classes):
        histogram = Histogram(breakpoints=[0.0, 1.0], freqs=[2], classes=classes)
        with pytest.raises(ValueError, match="'histogram' must have target statistics"):
            calibrate_binary([0.5], histogram, only_positive=only_positive)

    @pytest.mark.parametrize("only_positive", [False, True])
    @pytest.mark.parametrize("scores", [[np.nan], [0.1, np.nan, 0.9], [[np.nan]]])
    @pytest.mark.parametrize("dtype", [float, object])
    def test_rejects_nan_scores(self, only_positive, scores, dtype):
        scores = np.array(scores, dtype=dtype)
        histogram = Histogram.from_data_and_breakpoints(
            [0.1, 0.9], [0.0, 0.5, 1.0], [0, 1]
        )
        with pytest.raises(ValueError, match="'y_scores' must not contain NaN"):
            calibrate_binary(scores, histogram, only_positive=only_positive)

    @pytest.mark.parametrize("value", [-np.inf, np.inf])
    @pytest.mark.parametrize("dtype", [float, object])
    def test_rejects_infinite_scores(self, value, dtype):
        histogram = Histogram.from_data_and_breakpoints(
            [0.1, 0.9], [0.0, 0.5, 1.0], [0, 1]
        )
        scores = np.array([0.1, value], dtype=dtype)
        with pytest.raises(ValueError, match="'values' must not contain infinite"):
            calibrate_binary(scores, histogram)

    @pytest.mark.parametrize("only_positive", [False, True])
    @pytest.mark.parametrize("as_column", [False, True])
    def test_rejects_scores_in_empty_bins(self, only_positive, as_column):
        # Mix populated bins with two empty bins, including repeated hits
        histogram = Histogram.from_data_and_breakpoints(
            [0.1, 0.9], [0.0, 0.2, 0.4, 0.6, 1.0], [0, 1]
        )
        scores = np.array([0.1, 0.3, 0.5, 0.3, 0.9])
        if as_column:
            scores = scores[:, np.newaxis]
        with pytest.raises(
            ValueError,
            match=r"sample indices \[1, 2, 3\], bin indices \[1, 2, 1\]",
        ):
            calibrate_binary(scores, histogram, only_positive=only_positive)

    @pytest.mark.parametrize("only_positive", [False, True])
    @pytest.mark.parametrize("dtype", [float, object])
    def test_unused_empty_bins_allow_calibration(self, only_positive, dtype):
        # Empty bins are harmless when no input score maps to them
        histogram = Histogram.from_data_and_breakpoints(
            [0.1, 0.9], [0.0, 0.2, 0.4, 0.6, 1.0], [0, 1]
        )
        scores = np.array([-0.5, -0.1, 0.2, 0.9, 1.1, 1.5], dtype=dtype)
        expected = (
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
            if only_positive
            else [[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3
        )
        np.testing.assert_array_equal(
            calibrate_binary(scores, histogram, only_positive=only_positive), expected
        )

    @pytest.mark.parametrize("only_positive", [False, True])
    @pytest.mark.parametrize("as_list", [False, True])
    @pytest.mark.parametrize("dtype", [float, object])
    def test_vector_inputs(self, only_positive, as_list, dtype):
        # Compare lists and single-column inputs with vector calibration
        scores = np.array([0.1, 0.9])
        histogram = Histogram(
            breakpoints=[0.0, 0.5, 1.0],
            freqs=[1, 1],
            target_freqs=[(1, 0), (0, 1)],
            classes=[0, 1],
        )
        expected = calibrate_binary(scores, histogram, only_positive=only_positive)
        scores = scores.astype(dtype)
        scores = scores.tolist() if as_list else scores[:, np.newaxis]
        np.testing.assert_array_equal(
            calibrate_binary(scores, histogram, only_positive=only_positive),
            expected,
        )

    @pytest.mark.parametrize("scores", [np.array(0.5), np.ones((2, 2))])
    def test_invalid_score_shapes(self, scores):
        histogram = Histogram(
            breakpoints=[0.0, 1.0], freqs=[2], target_freqs=[(1, 1)], classes=[0, 1]
        )
        with pytest.raises(ValueError, match="y_scores must be 1-D"):
            calibrate_binary(scores, histogram)


class TestReliabilityDiagram:
    def test_closes_figure_when_histogram_construction_fails(self, monkeypatch):
        previous_figures = set(plt.get_fignums())

        def fail_histogram(*args, **kwargs):
            raise RuntimeError("Histogram construction failed")

        monkeypatch.setattr(Histogram, "from_data", fail_histogram)
        with pytest.raises(RuntimeError, match="Histogram construction failed"):
            build_reliability_diagram([0.1, 0.9], [0, 1])

        assert set(plt.get_fignums()) == previous_figures

    @pytest.mark.parametrize(
        "threshold", ["dirac_threshold", "log_plot_threshold", "min_density_bar_width"]
    )
    @pytest.mark.parametrize(
        "value", [np.inf, -np.inf, np.nan, 10**400, True, np.bool_(True)]
    )
    def test_rejects_invalid_plotting_thresholds(self, monkeypatch, threshold, value):
        def unexpected_plot(*args, **kwargs):
            pytest.fail("Threshold validation must precede plotting")

        monkeypatch.setattr(plt, "subplot_mosaic", unexpected_plot)
        with pytest.raises(
            ValueError, match=f"'{threshold}' must be finite and positive"
        ):
            build_reliability_diagram([0.1, 0.9], [0, 1], **{threshold: value})

    @pytest.mark.parametrize(
        "threshold", ["dirac_threshold", "log_plot_threshold", "min_density_bar_width"]
    )
    def test_rejects_nonpositive_plotting_thresholds(self, threshold):
        with pytest.raises(
            ValueError, match=f"'{threshold}' must be finite and positive"
        ):
            build_reliability_diagram([0.1, 0.9], [0, 1], **{threshold: 0.0})

    @pytest.mark.parametrize(
        "threshold", ["dirac_threshold", "log_plot_threshold", "min_density_bar_width"]
    )
    @pytest.mark.parametrize("value", ["0.1", None, [0.1, 0.2]])
    def test_rejects_non_numeric_plotting_thresholds(self, threshold, value):
        with pytest.raises(TypeError, match=f"'{threshold}' must be a real number"):
            build_reliability_diagram([0.1, 0.9], [0, 1], **{threshold: value})

    @pytest.mark.parametrize(
        "scores", [[np.nan, 0.9], [[np.nan], [0.9]], [[np.nan, 0.1], [0.1, 0.9]]]
    )
    @pytest.mark.parametrize("dtype", [float, object])
    def test_rejects_nan_scores_before_plotting(self, monkeypatch, scores, dtype):
        # Reject NaN even in the unused column before allocating a figure
        def unexpected_plot(*args, **kwargs):
            pytest.fail("NaN validation must precede plotting")

        monkeypatch.setattr(plt, "subplot_mosaic", unexpected_plot)
        with pytest.raises(ValueError, match="'y_scores' must not contain NaN"):
            build_reliability_diagram(np.array(scores, dtype=dtype), [0, 1])

    @pytest.mark.parametrize("scores", [[-np.inf, 0.9], [[0.9, -np.inf], [0.1, 0.9]]])
    def test_rejects_infinite_scores_before_plotting(self, monkeypatch, scores):
        def unexpected_plot(*args, **kwargs):
            pytest.fail("Infinite-score validation must precede plotting")

        monkeypatch.setattr(plt, "subplot_mosaic", unexpected_plot)
        with pytest.raises(ValueError, match="'y_scores' must not contain infinite"):
            build_reliability_diagram(scores, [0, 1])

    @pytest.mark.parametrize(
        "scores",
        [
            [-0.1, 0.9],
            [0.1, 1.1],
            [[1.1, 0.5], [0.1, 0.9]],
            [[0.9, 0.1], [0.1, -0.1]],
        ],
    )
    def test_rejects_scores_outside_unit_interval_before_plotting(
        self, monkeypatch, scores
    ):
        def unexpected_plot(*args, **kwargs):
            pytest.fail("Score-range validation must precede plotting")

        monkeypatch.setattr(plt, "subplot_mosaic", unexpected_plot)
        with pytest.raises(
            ValueError, match="'y_scores' must be in the range \\[0, 1\\]"
        ):
            build_reliability_diagram(scores, [0, 1])

    @pytest.mark.parametrize("labels", [[0, 0, 0], ["cat"] * 3, [0, 1, 2]])
    @pytest.mark.parametrize("as_column", [False, True])
    def test_rejects_nonbinary_targets_before_plotting(
        self, monkeypatch, labels, as_column
    ):
        # Invalid targets must not create figures or launch histogram construction
        def unexpected_work(*args, **kwargs):
            pytest.fail("Target validation must precede plotting and histograms")

        monkeypatch.setattr(plt, "subplot_mosaic", unexpected_work)
        monkeypatch.setattr(Histogram, "from_data", unexpected_work)
        y = np.array(labels)
        if as_column:
            y = y[:, np.newaxis]
        with pytest.raises(ValueError, match="exactly two distinct target labels"):
            build_reliability_diagram([0.1, 0.5, 0.9], y)

    @pytest.mark.parametrize(
        ("scores", "y"),
        [([0.1], [0, 1]), ([0.1, 0.9], [0]), ([[0.9, 0.1]], [0, 1])],
    )
    def test_rejects_mismatched_sample_counts(self, scores, y):
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            build_reliability_diagram(scores, y)

    @pytest.mark.parametrize("score_fixture", ["original", "constant"])
    @pytest.mark.parametrize("n_columns", [1, 2])
    @pytest.mark.parametrize("dtype", [float, object])
    def test_score_shapes_produce_same_diagram(
        self, score_fixture, n_columns, dtype, y_scores_fixtures, y_fixtures
    ):
        # Compare supported matrix inputs with positive-class scores
        scores = np.asarray(y_scores_fixtures[score_fixture])
        matrix = (
            scores[:, np.newaxis]
            if n_columns == 1
            else np.column_stack([1 - scores, scores])
        )
        figures = []
        try:
            for values, labels in (
                (scores, y_fixtures["int"].tolist()),
                (matrix.astype(dtype), np.asarray(y_fixtures["int"])[:, np.newaxis]),
            ):
                figure, _ = build_reliability_diagram(values, labels)
                figures.append(figure)
                figure.canvas.draw()
            np.testing.assert_array_equal(
                np.asarray(figures[0].canvas.buffer_rgba()),
                np.asarray(figures[1].canvas.buffer_rgba()),
            )
        finally:
            for figure in figures:
                plt.close(figure)

    @pytest.mark.parametrize(
        "scores", [np.array(0.5), np.empty((2, 0)), np.ones((2, 3)), np.ones((2, 1, 1))]
    )
    def test_invalid_score_shapes(self, scores):
        with pytest.raises(ValueError, match="one or two columns"):
            build_reliability_diagram(scores, np.array([0, 1]))


class TestDiracHeuristic:
    @pytest.mark.parametrize(
        ("freqs", "expected"),
        [
            ([0, 0, 1, 0, 0], [False, False, True, False, False]),
            ([1, 0, 0, 0, 1], [True, False, False, False, True]),
            ([1, 1, 0, 1, 1], [False, False, False, False, False]),
        ],
    )
    def test_only_populated_isolated_bins_are_diracs(self, freqs, expected):
        histogram = Histogram(
            breakpoints=[0.0, 1e-8, 2e-8, 3e-8, 4e-8, 5e-8], freqs=freqs
        )
        assert compute_dirac_indexes(histogram, 1e-6) == expected

    @pytest.mark.parametrize(
        ("width", "expected"), [(1e-8, True), (1e-6, False), (1e-4, False)]
    )
    def test_single_bin_width_threshold(self, width, expected):
        histogram = Histogram(breakpoints=[0.0, width], freqs=[1])
        assert compute_dirac_indexes(histogram, 1e-6) == [expected]

    def test_uniform(self):
        # The uniform distribution should have only one bin and no diracs
        rng = np.random.default_rng(seed=1234567)
        y_scores = rng.uniform(size=2000)
        uhist = Histogram.from_data(y_scores, use_finest=True)
        assert compute_dirac_indexes(uhist, 1e-06) == [False]

    def test_dirac(self):
        y_scores = np.array([0.2] * 250 + [0.5] * 250 + [0.9] * 500)
        y = np.array(
            [0] * 200 + [1] * 50 + [0] * 125 + [1] * 125 + [0] * 50 + [1] * 450
        )
        hist = Histogram.from_data(y_scores, y=y)
        y_scores_calib = calibrate_binary(y_scores, hist, only_positive=True)
        uhist = Histogram.from_data(y_scores_calib, use_finest=True)
        assert compute_dirac_indexes(uhist, 1e-06) == [True, False, True, False, True]

    def test_beta(self):
        # A sufficiently large continuous distribution sample should have no diracs
        rng = np.random.default_rng(seed=1234567)
        y_scores = rng.beta(a=0.5, b=0.5, size=2000)
        uhist = Histogram.from_data(y_scores, use_finest=True)
        assert not all(compute_dirac_indexes(uhist, 1e-06))
