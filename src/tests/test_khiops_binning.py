import json
import os

import numpy as np
import pandas as pd
import pytest

import khalib


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
    return {
        "original": adult_scores_df.y_score,
        "constant": np.full(adult_scores_df.shape[0], 0.5),
    }


@pytest.fixture(name="short_test_id")
def fixture_data_df(request):
    trans_table = str.maketrans("[", "_", "]")
    return request.node.nodeid.split("::")[-1].translate(trans_table)


@pytest.fixture(name="ref_binning")
def fixture_ref_binning(data_root_dir, short_test_id):
    ref_json_path = f"{data_root_dir}/binning/ref/{short_test_id}.json"
    if os.path.exists(ref_json_path):
        with open(f"{data_root_dir}/binning/ref/{short_test_id}.json") as ref_json_file:
            yield read_binning_from_json_data(json.load(ref_json_file))
    else:
        yield None


def read_binning_from_json_data(json_data):
    return khalib.Binning(
        breakpoints=json_data["breakpoints"],
        freqs=json_data["freqs"],
        target_freqs=[tuple(cur_freqs) for cur_freqs in json_data["target_freqs"]],
        classes=json_data["classes"],
    )


class TestKhiopsBinning:
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
        self, y_variants, y_scores_variants, ref_binning, method, target_mode, use_y
    ):
        # Prepare the input data for the binning
        y = y_variants[target_mode] if use_y else None
        y_scores = y_scores_variants["original"]

        # Compute the binning with the test settings and check it against the reference
        binning = khalib.compute_khiops_bins(y_scores, y=y, method=method)
        assert binning == ref_binning

    @pytest.mark.parametrize(("method", "target_mode", "use_y"), all_cases)
    def _test_single_value_score(
        self, y_variants, y_scores_variants, ref_binning, method, target_mode, use_y
    ):
        # Prepare the input data for the binning
        y = y_variants[target_mode] if use_y else None
        y_scores = y_scores_variants["constant"]

        # Compute the binning with the test settings and check it against the reference
        binning = khalib.compute_khiops_bins(y_scores, y=y, method=method)
        assert binning == ref_binning

    @pytest.mark.parametrize(("method", "target_mode", "use_y"), all_cases)
    def test_single_value_score(
        self, y_variants, y_scores_variants, ref_binning, method, target_mode, use_y
    ):
        # Prepare the input data for the binning
        y = y_variants[target_mode] if use_y else None
        y_scores = y_scores_variants["constant"]

        # Compute the binning with the test settings and check it against the reference
        binning = khalib.compute_khiops_bins(y_scores, y=y, method=method)
        assert binning == ref_binning

    @pytest.mark.parametrize("method", ["EqualFrequency", "EqualWidth", "MODL"])
    def test_no_info_target(self, y_variants, y_scores_variants, ref_binning, method):
        # Prepare the input data for the binning
        y = y_variants["random"]
        y_scores = y_scores_variants["original"]

        # Compute the binning with the test settings and check it against the reference
        binning = khalib.compute_khiops_bins(y_scores, y=y, method=method)
        assert binning == ref_binning

    def test_find_vfind_coherence(self, y_variants, y_scores_variants):
        # Prepare the input data for the binning
        y = y_variants["int"]
        y_scores = y_scores_variants["original"]

        # Create an equal width binning
        binning = khalib.compute_khiops_bins(y_scores, y=y, method="EqualWidth")

        test_scores = [i / 10 for i in range(-1, 12)]
        np.testing.assert_array_equal(
            binning.vfind(test_scores), [binning.find(score) for score in test_scores]
        )


class TestECE:
    @pytest.mark.parametrize("target_mode", ["bool", "float", "int", "intnl", "str"])
    @pytest.mark.parametrize(
        ("method", "expected_ece"), [("bin", 0.036162213), ("label-bin", 0.086438357)]
    )
    def test_binary_ece(
        self, method, expected_ece, target_mode, y_scores_variants, y_variants
    ):
        # Prepare the input data for the ECE estimation
        y = y_variants[target_mode]
        y_scores = y_scores_variants["original"]
        if target_mode not in ["bool", "int"]:
            y_scores = 1 - y_scores

        ece = khalib.binary_ece(y_scores, y, method=method)
        assert ece == pytest.approx(expected_ece)
