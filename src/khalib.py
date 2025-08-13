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
    binning = Binning()
    if y is not None:
        le = LabelEncoder()
        le.fit(y)
        binning.classes = le.classes_.tolist()

    # Obtain the target value indexes if they were calculated with Khiops
    if (target_values := results.preparation_report.target_values) is not None:
        if binning.classes_type is bool:
            casted_target_values = [val == "True" for val in target_values]
        else:
            casted_target_values = [binning.classes_type(val) for val in target_values]
        target_indexes = le.transform(casted_target_values)

    # Recover the breakpoints from the variable statistics objects
    # Normal case: There is a data grid
    score_stats = results.preparation_report.variables_statistics[0]
    if (data_grid := score_stats.data_grid) is not None:
        # Set the breakpoints
        for part in data_grid.dimensions[0].partition:
            binning.breakpoints.append(part.lower_bound)
        binning.breakpoints.append(data_grid.dimensions[0].partition[-1].upper_bound)

        # Set the frequencies and target frequencies
        # Supervised khiops execution
        if data_grid.is_supervised:
            # Recover the frequencies and target frequencies
            # Note: Target frequencies must be reordered to the order of binning.classes
            for part_target_freqs in score_stats.data_grid.part_target_frequencies:
                binning.freqs.append(sum(part_target_freqs))
                binning.target_freqs.append(
                    tuple(part_target_freqs[i] for i in target_indexes)
                )
        # Unsupervised khiops execution
        else:
            binning.freqs = score_stats.data_grid.frequencies.copy()
            if y is not None:
                y_indexes = le.transform(y)
                y_scores_bin_indexes = [binning.find(y_score) for y_score in y_scores]
                target_freqs = [[0] * len(le.classes_) for _ in range(binning.n_bins)]
                for y_score_bin_index, y_index in zip(y_scores_bin_indexes, y_indexes):
                    target_freqs[y_score_bin_index][y_index] += 1
                binning.target_freqs = [tuple(freqs) for freqs in target_freqs]

    # Otherwise there is just one interval
    else:
        # Case of non-informative variable: binning consisting only of (min, max)
        if score_stats.min < score_stats.max:
            binning.breakpoints = [score_stats.min, score_stats.max]
        # Case of variable with one value: binning consisting only of (min, min + eps)
        else:
            binning.breakpoints = [score_stats.min, score_stats.min + 1.0e-9]

        # Add the total frequency and, if y was provided, the target frequencies
        binning.freqs.append(results.preparation_report.instance_number)
        if target_freqs := results.preparation_report.target_value_frequencies:
            binning.target_freqs.append(tuple(target_freqs[i] for i in target_indexes))
        elif y is not None:
            binning.target_freqs.append(
                tuple(np.unique(y, return_counts=True)[1].tolist())
            )

    return binning


@dataclass
class Binning:
    breakpoints: list[float] = field(default_factory=list)
    freqs: list[int] = field(default_factory=list)
    target_freqs: list[tuple] = field(default_factory=list)
    classes: list = field(default_factory=list)

    def find(self, value: float) -> int:
        return max(bisect_left(self.breakpoints[:-1], value) - 1, 0)

    @property
    def n_bins(self):
        return max(len(self.breakpoints) - 1, 0)

    @property
    def classes_type(self):
        if self.classes:
            return type(self.classes[0])
        return None

    def get_bin(self, i: int) -> tuple[float, float]:
        if i >= self.n_bins:
            raise IndexError(f"There are only {self.n_bins} bins. Index: {i}")
        return tuple(self.breakpoints[i : i + 2])


def robust_estimate_binary_ece(y, y_scores, y_pos, return_estimations=False):
    bins, _ = compute_khiops_bins(
        y_scores, y, method="EqualFrequency", max_parts=math.ceil(math.sqrt(len(y)))
    )
    return binary_ece_lb(
        y,
        y_scores,
        y_pos,
        bins,
        return_estimations=return_estimations,
    )


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

    zipped_y = sorted(zip(y, y_scores), key=lambda k: k[1])

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


def binary_ece_bin(y, y_scores, y_pos, bins, return_estimations=False, debias=False):
    """Estimates the ECE_bin estimator (also known as "plugin")"""
    accuracy_by_bin = np.zeros(len(bins))
    average_score_by_bin = np.zeros(len(bins))
    n_samples_by_bin = np.zeros(len(bins))

    for y_value, y_score in zip(y, y_scores):
        i_bin = find_bin(y_score, bins)
        average_score_by_bin[i_bin] += y_score
        n_samples_by_bin[i_bin] += 1
        if y_value == y_pos:
            accuracy_by_bin[i_bin] += 1
    if return_estimations:
        estimations_df = pd.DataFrame(
            {"freq": n_samples_by_bin, "freq_pos": accuracy_by_bin}, index=bins
        )
    for i in range(0, len(bins)):
        if n_samples_by_bin[i] > 0:
            average_score_by_bin[i] /= n_samples_by_bin[i]
            accuracy_by_bin[i] /= n_samples_by_bin[i]

    if return_estimations:
        estimations_df["ave_score"] = average_score_by_bin
        estimations_df["acc"] = accuracy_by_bin

    ece = 0
    for i in range(0, len(bins)):
        ece += (
            n_samples_by_bin[i]
            / len(y)
            * math.fabs(average_score_by_bin[i] - accuracy_by_bin[i])
        )

    if return_estimations:
        return ece, estimations_df
    else:
        return ece


def binary_ece_lb(y, y_scores, y_pos, bins, return_estimations=False):
    accuracy_by_bin = np.zeros(len(bins))
    n_samples_by_bin = np.zeros(len(bins))
    y_score_bin_indexes = []

    for y_value, y_score in zip(y, y_scores):
        y_score_i_bin = find_bin(y_score, bins)
        y_score_bin_indexes.append(y_score_i_bin)
        n_samples_by_bin[y_score_i_bin] += 1
        if y_value == y_pos:
            accuracy_by_bin[y_score_i_bin] += 1

    if return_estimations:
        estimations_df = pd.DataFrame(
            {"freq": n_samples_by_bin, "freq_pos": accuracy_by_bin}, index=bins
        )
    for i in range(0, len(bins)):
        if n_samples_by_bin[i] > 0:
            accuracy_by_bin[i] /= n_samples_by_bin[i]

    diff_ave = np.zeros(len(bins))
    for i, y_score in zip(y_score_bin_indexes, y_scores):
        diff_ave[i] += abs(accuracy_by_bin[i] - y_score)

    for i in range(0, len(bins)):
        if n_samples_by_bin[i] > 0:
            diff_ave[i] /= n_samples_by_bin[i]

    if return_estimations:
        estimations_df["acc"] = accuracy_by_bin
        estimations_df["diff_ave"] = diff_ave

    ece = 0
    for i in range(0, len(bins)):
        ece += n_samples_by_bin[i] / len(y) * diff_ave[i]

    if return_estimations:
        return ece, estimations_df
    else:
        return ece


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
