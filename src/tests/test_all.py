import json
import os

import numpy as np
import pandas as pd
import pytest

import khalib
from khalib import Histogram


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
    ref_json_path = f"{data_root_dir}/histogram/ref/{short_test_id}.json"
    if os.path.exists(ref_json_path):
        with open(
            f"{data_root_dir}/histogram/ref/{short_test_id}.json"
        ) as ref_json_file:
            yield read_histogram_from_json_data(json.load(ref_json_file))
    else:
        yield None


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
        ("modl", "bool", True),
        ("modl", "float", True),
        ("modl", "int", False),
        ("modl", "int", True),
        ("modl", "intnl", True),
    ]

    @pytest.mark.parametrize(("method", "y_fixture", "use_y"), all_cases)
    def test_happy_path(
        self, y_fixtures, y_scores_fixtures, ref_histogram, method, y_fixture, use_y
    ):
        # Prepare the input data for the histogram
        y = y_fixtures[y_fixture] if use_y else None
        y_scores = y_scores_fixtures["original"]

        # Compute the histogram with the test settings, check it against the reference
        histogram = Histogram.from_data(y_scores, y=y, method=method)
        assert histogram == ref_histogram

    @pytest.mark.parametrize(("method", "y_fixture", "use_y"), all_cases)
    def _test_single_value_score(
        self, y_fixtures, y_scores_fixtures, ref_histogram, method, y_fixture, use_y
    ):
        # Prepare the input data for the histogram
        y = y_fixtures[y_fixture] if use_y else None
        y_scores = y_scores_fixtures["constant"]

        # Compute the histogram with the test settings, check it against the reference
        histogram = Histogram.from_data(y_scores, y=y, method=method)
        assert histogram == ref_histogram

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

    @pytest.mark.parametrize("method", ["eq-freq", "eq-width", "modl"])
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

        test_scores = [i / 10 for i in range(-1, 12)]
        np.testing.assert_array_equal(
            histogram.vfind(test_scores),
            [histogram.find(score) for score in test_scores],
        )


class TestECE:
    @pytest.mark.parametrize("y_fixture", ["bool", "float", "int", "intnl", "str"])
    @pytest.mark.parametrize("y_scores_fixture", ["original", "original-2d"])
    @pytest.mark.parametrize(
        ("method", "expected_ece"), [("bin", 0.036162213), ("label-bin", 0.086438357)]
    )
    def test_binary_ece(
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
        y_scores = y_scores_fixtures[y_scores_fixture]
        if is_target_inverted(y_fixture):
            if len(y_scores.shape) == 1:
                y_scores = 1 - y_scores
            else:
                y_scores[:, [0, 1]] = y_scores[:, [1, 0]]

        # Estimate the ECE
        ece = khalib.ece(y_scores, y, method=method)
        assert ece == pytest.approx(expected_ece)

    @pytest.mark.parametrize(
        ("multi_class_method", "expected_ece"),
        [("top-label", 0.0723944), ("classwise", 0.04642129)],
    )
    def test_multi_class_ece(
        self, multi_class_method, expected_ece, vehicles_scores_df
    ):
        ece = khalib.ece(
            vehicles_scores_df.drop("y", axis=1).__array__(),
            vehicles_scores_df.y,
            multi_class_method=multi_class_method,
        )
        assert ece == pytest.approx(expected_ece)
