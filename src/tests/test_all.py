import json
import os

import numpy as np
import pandas as pd
import pytest
from khiops.sklearn import KhiopsClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from khalib import Histogram, KhalibClassifier, calibration_error


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


class TestHistogram:
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

    @pytest.mark.parametrize("method", ["eq-freq", "eq-width", "khiops"])
    def test_no_info_target(self, y_fixtures, y_scores_fixtures, ref_histogram, method):
        # Prepare the input data for the histogram
        y = y_fixtures["random"]
        y_scores = y_scores_fixtures["original"]

        # Compute the histogram with the test settings, check it against the reference
        histogram = Histogram.from_data(y_scores, y=y, method=method)
        assert histogram == ref_histogram

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
    def test_fast_multi_class_ece(
        self, multi_class_method, expected_ece, vehicles_scores_df
    ):
        ece = calibration_error(
            vehicles_scores_df.drop("y", axis=1).__array__(),
            vehicles_scores_df.y,
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
    # x_not_test, y_not_test = make_classification(
    # n_samples=n_train + n_calib,
    # n_classes=n_classes,
    # n_features=n_features,
    # n_informative=n_informative,
    # random_state=314,
    # return_X_y=True,
    # )
    # x_train, x_calib, y_train, y_calib = train_test_split(
    # x_not_test, y_not_test, train_size=n_train, random_state=3141
    # )
    # x_test, y_test =  make_classification(
    # n_samples=n_test,
    # n_classes=n_classes,
    # n_features=n_features,
    # n_informative=n_informative,
    # random_state=31415,
    # return_X_y=True,
    # )
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
