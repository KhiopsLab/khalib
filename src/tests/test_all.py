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


@pytest.fixture(name="y_variants")
def fixture_y_variants(adult_scores_df):
    rng = np.random.default_rng(seed=1234567)
    return {
        "int": adult_scores_df.y,
        "str": np.where(adult_scores_df.y == 0, "zero", "one"),
        "intnl": np.where(adult_scores_df.y == 0, 10, 2),
        "bool": np.where(adult_scores_df.y == 0, False, True),
        "float": np.where(adult_scores_df.y == 0, 10.1, 2.1),
        "random": rng.integers(low=0, high=2, size=adult_scores_df.shape[0]),
    }


@pytest.fixture(name="y_scores_variants")
def fixture_y_scores_variants(adult_scores_df):
    y_scores = adult_scores_df.y_score
    return {
        "original": y_scores,
        "original-2D": np.transpose(np.vstack([1 - y_scores, y_scores])),
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


def is_target_inverted(target_mode):
    return target_mode in ["float", "intnl", "str"]


class TestHistogram:
    all_cases = [
        ("EqualFrequency", "bool", True),
        ("EqualFrequency", "float", True),
        ("EqualFrequency", "int", False),
        ("EqualFrequency", "int", True),
        ("EqualFrequency", "intnl", True),
        ("EqualWidth", "bool", True),
        ("EqualWidth", "float", True),
        ("EqualWidth", "int", False),
        ("EqualWidth", "int", True),
        ("EqualWidth", "intnl", True),
        ("MODL", "bool", True),
        ("MODL", "float", True),
        ("MODL", "int", False),
        ("MODL", "int", True),
        ("MODL", "intnl", True),
    ]

    @pytest.mark.parametrize(("method", "target_mode", "use_y"), all_cases)
    def test_happy_path(
        self, y_variants, y_scores_variants, ref_histogram, method, target_mode, use_y
    ):
        # Prepare the input data for the histogram
        y = y_variants[target_mode] if use_y else None
        y_scores = y_scores_variants["original"]

        # Compute the histogram with the test settings, check it against the reference
        histogram = Histogram.from_data(y_scores, y=y, method=method)
        assert histogram == ref_histogram

    @pytest.mark.parametrize(("method", "target_mode", "use_y"), all_cases)
    def _test_single_value_score(
        self, y_variants, y_scores_variants, ref_histogram, method, target_mode, use_y
    ):
        # Prepare the input data for the histogram
        y = y_variants[target_mode] if use_y else None
        y_scores = y_scores_variants["constant"]

        # Compute the histogram with the test settings, check it against the reference
        histogram = Histogram.from_data(y_scores, y=y, method=method)
        assert histogram == ref_histogram

    @pytest.mark.parametrize(("method", "target_mode", "use_y"), all_cases)
    def test_single_value_score(
        self, y_variants, y_scores_variants, ref_histogram, method, target_mode, use_y
    ):
        # Prepare the input data for the histogram
        y = y_variants[target_mode] if use_y else None
        y_scores = y_scores_variants["constant"]

        # Compute the histogram with the test settings, check it against the reference
        histogram = Histogram.from_data(y_scores, y=y, method=method)
        assert histogram == ref_histogram

    @pytest.mark.parametrize("method", ["EqualFrequency", "EqualWidth", "MODL"])
    def test_no_info_target(self, y_variants, y_scores_variants, ref_histogram, method):
        # Prepare the input data for the histogram
        y = y_variants["random"]
        y_scores = y_scores_variants["original"]

        # Compute the histogram with the test settings, check it against the reference
        histogram = Histogram.from_data(y_scores, y=y, method=method)
        assert histogram == ref_histogram

    def test_find_vfind_coherence(self, y_variants, y_scores_variants):
        # Prepare the input data for the histogram
        y = y_variants["int"]
        y_scores = y_scores_variants["original"]

        # Create an equal width histogram
        histogram = Histogram.from_data(y_scores, y=y, method="EqualWidth")

        test_scores = [i / 10 for i in range(-1, 12)]
        np.testing.assert_array_equal(
            histogram.vfind(test_scores),
            [histogram.find(score) for score in test_scores],
        )


class TestECE:
    @pytest.mark.parametrize("target_mode", ["bool", "float", "int", "intnl", "str"])
    @pytest.mark.parametrize("variant", ["original", "original-2D"])
    @pytest.mark.parametrize(
        ("method", "expected_ece"), [("bin", 0.036162213), ("label-bin", 0.086438357)]
    )
    def test_binary_ece(
        self, method, expected_ece, target_mode, variant, y_scores_variants, y_variants
    ):
        # Prepare the input data for the ECE estimation
        y = y_variants[target_mode]
        y_scores = y_scores_variants[variant]
        if is_target_inverted(target_mode):
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
