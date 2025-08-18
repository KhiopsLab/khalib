import math
import os
import tempfile
import warnings
from bisect import bisect_left
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from khiops import core as kh
from khiops.core.internals.runner import KhiopsLocalRunner
from khiops.sklearn import KhiopsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.preprocessing import LabelEncoder, label_binarize, normalize

khiops_single_core_runner: KhiopsLocalRunner | None = None


def build_single_core_khiops_runner():
    """Build a Khiops runner with one core

    Running with a single core is faster for most of the calibration use-cases
    """
    default_max_cores = os.environ.get("KHIOPS_PROC_NUMBER")
    os.environ["KHIOPS_PROC_NUMBER"] = "1"
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=".*Too few cores:.*")
        global khiops_single_core_runner
        khiops_single_core_runner = KhiopsLocalRunner()
        if default_max_cores is None:
            del os.environ["KHIOPS_PROC_NUMBER"]
        else:
            os.environ["KHIOPS_PROC_NUMBER"] = default_max_cores


@contextmanager
def single_core_khiops_runner():
    """A context manager to run Khiops with a single core"""
    if khiops_single_core_runner is None:
        build_single_core_khiops_runner()
    default_runner = kh.get_runner()
    kh.set_runner(khiops_single_core_runner)
    try:
        yield
    finally:
        kh.set_runner(default_runner)


def compute_khiops_bins(y_scores, y=None, method="MODL", max_bins=0):
    """Computes a binning of an 1D vector with Khiops

    Parameters
    ----------
    y_scores : array-like of shape (n_samples,) or (n_samples, 1)
        Input scores.
    y : array-like of shape (n_samples,) or (n_samples, 1), optional
        Target values. If set, then the method parameters is ignored and set to "MODL".
    method : {"MODL", "EqualFrequency", "EqualWidth"}, default="MODL"
        Binning method:

        - "MODL": A non-parametric regularized binning method.
        - "EqualFrequency": All bins have the same number of elements. If many instances
          have too many values the algorithm will put it in its own bin, which will be
          larger than the other ones.
        - "EqualWidth": All bins have the same width.

        If the method is set to "EqualFrequency" or "EqualWidth" is set then 'y' is
        ignored.
    max_bins: int, default=0
        The maximum number of bins to be created. The algorithms usually create this
        number of bins but they may create less. The default value 0 means:

        - For "MODL": that there is no limit to the number of intervals.
        - For "EqualFrequency" or "EqualWidth": that 10 is the maximum number of
          intervals.

    Returns
    -------
    `Binning`
        The binning object containing the bin limits and frequencies.
    """
    # Check inputs
    if len(y_scores.shape) > 1 and y_scores.shape[1] > 1:
        raise ValueError(f"y_scores must be 1-D but it has shape {y_scores.shape}.")
    if y is not None and len(y.shape) > 1 and y.shape[1] > 1:
        raise ValueError(f"y must be 1-D but it has shape {y.shape}.")
    valid_methods = ["MODL", "EqualFrequency", "EqualWidth"]
    if method not in valid_methods:
        raise ValueError(f"method must be in {valid_methods}. It is '{method}'")
    if max_bins < 0:
        raise ValueError(f"max_bins must be non-negative. It is {max_bins}")

    # Set the y vector to be used by khiops
    # This is necessary because for the "EqualFrequency" and "EqualWidth" methods if
    # target is set then it uses MODL.
    y_khiops = y if method == "MODL" else None

    # Create Khiops dictionary
    kdom = kh.DictionaryDomain()
    kdic = kh.Dictionary()
    kdic.name = "scores"
    kdom.add_dictionary(kdic)
    var = kh.Variable()
    var.name = "y_score"
    var.type = "Numerical"
    kdic.add_variable(var)
    if y_khiops is not None:
        var = kh.Variable()
        var.name = "y"
        var.type = "Categorical"
        kdic.add_variable(var)

    # Create an execution context stack with:
    # - A temporary directory context
    # - A catch_warnings context
    # - A single_core_khiops_runner context
    with ExitStack() as ctx_stack:
        work_dir = ctx_stack.enter_context(
            tempfile.TemporaryDirectory(prefix="khiops-bins_")
        )
        ctx_stack.enter_context(warnings.catch_warnings())
        ctx_stack.enter_context(single_core_khiops_runner())

        # Create data table file for khiops
        df_spec = {
            "y_score": y_scores if len(y_scores.shape) == 1 else y_scores.flatten()
        }
        if y_khiops is not None:
            df_spec["y"] = y_khiops if len(y.shape) == 1 else y.flatten()
        output_df = pd.DataFrame(df_spec)
        output_table_path = f"{work_dir}/y_scores.txt"
        output_df.to_csv(output_table_path, sep="\t", index=False)

        # Ignore the non-informative warning of Khiops
        warnings.filterwarnings(
            action="ignore",
            message="(?s:.*No informative input variable available.*)",
        )

        # Execute Khiops recover the report
        kh.train_predictor(
            kdom,
            "scores",
            output_table_path,
            "y" if y_khiops is not None else "",
            f"{work_dir}/report.khj",
            sample_percentage=100,
            field_separator="\t",
            header_line=True,
            max_trees=0,
            discretization_method=method,
            max_parts=max_bins,
            do_data_preparation_only=True,
        )
        results = kh.read_analysis_results_file(f"{work_dir}/report.khj")

    # Initialize the binning
    if y is not None:
        le = LabelEncoder()
        le.fit(y)
        classes = le.classes_.tolist()
        # Note: When building `classes` variable is important with the `tolist` method,
        # because it converts the numpy types to native Python ones.

    # Obtain the target value indexes if they were calculated with Khiops
    if (target_values := results.preparation_report.target_values) is not None:
        if type(classes[0]) is bool:
            casted_target_values = [val == "True" for val in target_values]
        else:
            casted_target_values = [type(classes[0])(val) for val in target_values]
        target_indexes = le.transform(casted_target_values)

    # Recover the breakpoints from the variable statistics objects
    # Normal case: There is a data grid
    score_stats = results.preparation_report.variables_statistics[0]
    if (data_grid := score_stats.data_grid) is not None:
        # Create the breakpoints
        breakpoints = [part.lower_bound for part in data_grid.dimensions[0].partition]
        breakpoints.append(data_grid.dimensions[0].partition[-1].upper_bound)

        # Create the frequencies and target frequencies
        # Supervised khiops execution
        if data_grid.is_supervised:
            # Recover the frequencies and target frequencies
            # Note: Target frequencies must be reordered to the order of `classes`
            freqs = [
                sum(tfreqs) for tfreqs in score_stats.data_grid.part_target_frequencies
            ]
            target_freqs = [
                tuple(tfreqs[i] for i in target_indexes)
                for tfreqs in score_stats.data_grid.part_target_frequencies
            ]
        # Unsupervised khiops execution
        else:
            freqs = score_stats.data_grid.frequencies.copy()
            if y is not None:
                y_scores_bin_indexes = (
                    np.searchsorted(breakpoints[:-1], y_scores, side="left") - 1
                )
                y_scores_bin_indexes[y_scores_bin_indexes < 0] = 0
                y_indexes = le.transform(y)
                target_freqs = [[0 for _ in le.classes_] for _ in breakpoints[:-1]]
                for y_score_bin_index, y_index in np.nditer(
                    [y_scores_bin_indexes, y_indexes]
                ):
                    target_freqs[y_score_bin_index][y_index] += 1
                target_freqs = [tuple(freqs) for freqs in target_freqs]

    # Otherwise there is just one interval
    else:
        # Case of non-informative variable: binning consisting only of (min, max)
        if score_stats.min < score_stats.max:
            breakpoints = [score_stats.min, score_stats.max]
        # Case of variable with one value: binning consisting only of (min, min + eps)
        else:
            breakpoints = [score_stats.min, score_stats.min + 1.0e-9]

        # Add the total frequency and, if y was provided, the target frequencies
        freqs = [results.preparation_report.instance_number]
        if (
            target_freqs := results.preparation_report.target_value_frequencies
        ) is not None:
            target_freqs = [tuple(target_freqs[i] for i in target_indexes)]
        elif y is not None:
            target_freqs = [tuple(np.unique(y, return_counts=True)[1].tolist())]

    return Binning(
        breakpoints=breakpoints,
        freqs=freqs,
        target_freqs=target_freqs if y is not None else [],
        classes=le.classes_.tolist() if y is not None else [],
    )


def compute_binning_from_bins(bins: list[tuple[float, float]], y_scores, y):
    """Builds a binning from a list of bins and data

    This is a helper function mostly for massive experiments, as it avoids the overhead
    of a Khiops process executon.

    Parameters
    ----------
    bins : list[tuple[float, float]]
        The score bins.
    y_scores : array-like of shape (n_samples,) or (n_samples, 1)
        Input scores.
    y : array-like of shape (n_samples,) or (n_samples, 1)
        Target values.
    """
    # Check inputs
    if len(y_scores.shape) > 1 and y_scores.shape[1] > 1:
        raise ValueError(f"y_scores must be 1-D but it has shape {y_scores.shape}.")
    if y is not None and len(y.shape) > 1 and y.shape[1] > 1:
        raise ValueError(f"y must be 1-D but it has shape {y.shape}.")
    for i, a_bin in enumerate(bins):
        if a_bin[0] > a_bin[1]:
            raise ValueError(f"Bin at index {i} is not sorted: {a_bin}")
        if i > 0 and bins[i - 1][1] != a_bin[0]:
            raise ValueError(
                f"Bin at index {i} is not adjacent to the previous one: "
                f"{bins[i - 1]} {a_bin}"
            )

    # Initialize the breakpoints, score and target indexes
    le = LabelEncoder().fit(y)
    breakpoints = [a_bin[0] for a_bin in bins] + [a_bin[1]]
    y_scores_bin_indexes = np.searchsorted(breakpoints[:-1], y_scores, side="left") - 1
    y_scores_bin_indexes[y_scores_bin_indexes < 0] = 0
    y_indexes = le.transform(y)

    # Initialize the frequencies
    freqs = [0 for _ in len(bins)]
    target_freqs = [[0 for _ in le.classes_] for _ in bins]
    for y_score_bin_index, y_index in np.nditer([y_scores_bin_indexes, y_indexes]):
        freqs[y_score_bin_index] += 1
        target_freqs[y_score_bin_index][y_index] += 1

    return Binning(
        breakpoints=breakpoints,
        freqs=freqs,
        target_freqs=[tuple(freqs) for freqs in target_freqs],
        classes=le.classes_.tolist(),
    )


@dataclass()
class Binning:
    breakpoints: list[float]
    freqs: list[int]
    target_freqs: list[tuple] = field(default_factory=list)
    classes: list = field(default_factory=list)
    densities: list[float] = field(init=False)
    target_probas: list[tuple] = field(init=False, default_factory=list)

    def __post_init__(self):
        # Check consistency of the constructor parameters
        for i in range(n_bins := len(self.breakpoints) - 1):
            if (left := self.breakpoints[i]) >= (right := self.breakpoints[i + 1]):
                raise ValueError(
                    "`breakpoints` must be increasing, "
                    f"but at index {i} we have {left} >= {right}."
                )
        if (n_freqs := len(self.freqs)) != n_bins:
            raise ValueError(
                "`freqs` must match the size of `breakpoints` minus 1: "
                f"{n_freqs} != {n_bins}"
            )
        if self.target_freqs:
            if (n_target_freqs := len(self.target_freqs)) != n_freqs:
                raise ValueError(
                    "`target_freqs` length different from that of `freqs`: "
                    f"({n_target_freqs} != {n_freqs}"
                )
            for i, tfreqs in enumerate(self.target_freqs):
                if len(tfreqs) != len(self.classes):
                    raise ValueError(
                        f"`target_freqs` at bin index {i} has a length different from "
                        f"the number of classes: {len(tfreqs)} != {len(self.classes)}."
                    )
                if sum(tfreqs) != self.freqs[i]:
                    f"`target_freqs` at bin index {i} sums different from the bin "
                    f"frequency: {sum(tfreqs)} != {self.freqs[i]}"

        # Initialize the densities and target probabilities
        self.densities = [
            self.freqs[i]
            / sum(self.freqs)
            / (self.breakpoints[i + 1] - self.breakpoints[i])
            for i in range(len(self.freqs))
        ]
        if self.target_freqs:
            target_probas = []
            for bin_freq, bin_target_freqs in zip(
                self.freqs, self.target_freqs, strict=True
            ):
                target_probas.append(
                    tuple(tfreq / bin_freq for tfreq in bin_target_freqs)
                )

            # Initialize the target probabilities
            self.target_probas = target_probas

    def find(self, value: float) -> int:
        return max(bisect_left(self.breakpoints[:-1], value) - 1, 0)

    def vfind(self, values):
        # Note: searchsorted with side="left" gives the bin index shifted by 1, except
        # for the outliers in the left. We adjust them with a selection.
        indexes = np.searchsorted(self.breakpoints[:-1], values, side="left") - 1
        indexes[indexes < 0] = 0
        return indexes

    @property
    def n_bins(self):
        return max(len(self.breakpoints) - 1, 0)

    @property
    def bins(self):
        return [tuple(self.breakpoints[i : i + 2]) for i in range(self.n_bins)]

    @property
    def classes_type(self):
        if self.classes:
            return type(self.classes[0])
        return None


def robust_estimate_binary_ece(y, y_scores, y_pos):
    bins, _ = compute_khiops_bins(
        y_scores, y, method="EqualFrequency", max_parts=math.ceil(math.sqrt(len(y)))
    )
    return binary_ece(y_scores, y, method="label-bin")


def robust_estimate_top_level_ece(y, probas, classes):
    return robust_estimate_kht_level_ece(y, probas, 1, classes)


def robust_estimate_kht_level_ece(y, y_scores, k, classes):
    _classes = classes.__array__()
    _y = y.__array__()
    prediction_indexes = np.argsort(y_scores, axis=1)
    prediction_indexes = prediction_indexes[:, -k]
    correct_predictions = (_y == _classes[prediction_indexes]).astype(int)

    return robust_estimate_binary_ece(
        correct_predictions, y_scores[np.arange(len(y)), prediction_indexes], 1
    )


def robust_estimate_classwise_ece(y, y_scores, classes):
    y_classes, counts = np.unique(y, return_counts=True)
    class_probas = counts / sum(counts)
    ece = 0
    for i, y_class in enumerate(y_classes):
        ece += class_probas[i] * robust_estimate_class_ece(
            y, y_scores, y_class, classes
        )
    return ece


def robust_estimate_class_ece(y, y_scores, y_class, classes):
    _classes = classes.__array__()
    _y = y.__array__()
    y_pos_class = (_y == y_class).astype(int)
    i_class = np.where(_classes == y_class)
    return robust_estimate_binary_ece(y_pos_class, y_scores[:, i_class], 1)


def find_bin(value, bins):
    breakpoints = [a_bin[1] for a_bin in bins[0:-1]]
    i_val = bisect_left(breakpoints, value)
    return i_val


def binary_ece_knn(y, y_scores, y_pos):
    k = int(math.sqrt(len(y)))

    zipped_y = sorted(zip(y, y_scores, strict=False), key=lambda k: k[1])

    # Initialize windows sums
    win_sum_y = 0
    win_sum_y_scores = 0
    for i in range(1, k + 1):
        win_sum_y += int(zipped_y[i][0] == y_pos)
        win_sum_y_scores += zipped_y[i][1]

    diff_sum = math.fabs(win_sum_y - win_sum_y_scores)

    for i in range(k, len(y)):
        win_sum_y += int(zipped_y[i - k][0] == y_pos) - int(zipped_y[i - k][0] == y_pos)
        win_sum_y_scores += zipped_y[i][1] - zipped_y[i - k][1]
        diff_sum += math.fabs(win_sum_y - win_sum_y_scores)

    return diff_sum / (len(y) * k)


def binary_ece(
    y_scores,
    y,
    method: str = "label-bin",
    binning_method: str = "MODL",
    max_bins: int = 0,
    binning: Binning | None = None,
):
    """Estimates the ECE for a pair of score and label vectors

    Parameters
    ----------
    y_scores : array-like of shape (n_samples,) or (n_samples, 1)
        Input scores.
    y : array-like of shape (n_samples,) or (n_samples, 1)
        Target values. Must have exactly 2 different values.
    method : {"label-bin", "bin"}, default="label-bin"
        ECE estimation method. See below for details.
    binning_method : {"MODL", "EqualFrequency", "EqualWidth"}, default="MODL"
        Binning method:

        - "MODL": A non-parametric regularized binning method.
        - "EqualFrequency": All bins have the same number of elements. If many instances
          have too many values the algorithm will put it in its own bin, which will be
          larger than the other ones.
        - "EqualWidth": All bins have the same width.

        If the method is set to "EqualFrequency" or "EqualWidth" is set then 'y' is
        ignored.
    max_bins : int, default=0
        The maximum number of bins to be created. The algorithms usually create this
        number of bins but they may create less. The default value 0 means that there is
        no limit to the number of intervals.
    binning : `Binning`, optional
        A ready-made binning. If set then it is used for the ECE computation and the
        parameters bininng_method and max_bins are ignored.
    """
    # Compute the binning if necessary
    if binning is None:
        binning = compute_khiops_bins(
            y_scores, y=y, method=binning_method, max_bins=max_bins
        )

    # Check that the binning has only two classes
    if (n_classes := len(binning.classes)) != 2:
        raise ValueError(f"Target 'y' must have only 2 classes. It has {n_classes}.")

    # Estimate the ECE with the binning
    if method == "label-bin":
        sum_diffs = 0
        for y_score, i in np.nditer([y_scores, binning.vfind(y_scores)]):
            sum_diffs += math.fabs(
                y_score - binning.target_freqs[i][1] / binning.freqs[i]
            )
        return sum_diffs / len(y)
    else:
        assert method == "bin"
        sum_score_by_bin = [0 for _ in binning.bins]
        for y_score, i in np.nditer([y_scores, binning.vfind(y_scores)]):
            sum_score_by_bin[i] += y_score
        return sum(
            [
                math.fabs(sum_score_by_bin[i] - binning.target_freqs[i][1])
                for i in range(binning.n_bins)
            ]
        ) / len(y)


class KhiopsCalibrator(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    # noqa: N803
    def __init__(self, clf, normalize=False):
        self.clf = clf
        self.normalize = normalize

    def fit(self, X, y):  # noqa: N803
        probas = self.clf.predict_proba(X)
        if len(self.clf.classes_) == 2:
            self.calibrator_ = KhiopsClassifier(n_trees=0)
            self.calibrator_.fit(probas[:, 1].reshape(-1, 1), y)
        # One-vs-Rest
        else:
            binarized_labels = label_binarize(y, classes=self.clf.classes_)
            self.estimators_ = []
            self.calibrator_ = KhiopsClassifier(n_trees=0)
            for j in range(len(self.clf.classes_)):
                self.estimators_.append(self.calibrator_)
                self.estimators_[j].fit(
                    probas[:, j].reshape(-1, 1), binarized_labels[:, j]
                )
        return self

    def predict_proba(self, X):  # noqa: N803
        y_scores = self.clf.predict_proba(X)
        if len(self.clf.classes_) == 2:
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", message=".*(Underflow).*")
                calibrated_probas = self.calibrator_.predict_proba(
                    y_scores[:, 1].reshape(-1, 1)
                )
        # One-vs-Rest
        else:
            all_probas = []
            for j, estimator in enumerate(self.estimators_):
                with warnings.catch_warnings():
                    warnings.filterwarnings(action="ignore", message=".*(Underflow).*")
                    all_probas.append(
                        estimator.predict_proba(y_scores[:, j].reshape(-1, 1))[
                            :, 1
                        ].reshape(-1, 1)
                    )
            unnormalized_probas = np.concatenate(all_probas, axis=1)
            if self.normalize:
                calibrated_probas = normalize(unnormalized_probas, axis=1, norm="l1")
            else:
                calibrated_probas = []
                for probas in unnormalized_probas:
                    new_probas = []
                    for k in range(len(self.clf.classes_)):
                        new_probas.append(
                            1
                            / (len(self.clf.classes_))
                            * (
                                (len(self.clf.classes_) - 1) * probas[k]
                                + np.sum(np.delete(1 - probas, k))
                                - (len(self.clf.classes_) - 2)
                            )
                        )
                    calibrated_probas.append(new_probas)
                calibrated_probas = np.array(calibrated_probas)

        return calibrated_probas
