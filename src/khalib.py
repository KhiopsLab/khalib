import bisect
import math
import os
import tempfile
import warnings
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from khiops import core as kh
from khiops.core.internals.runner import KhiopsLocalRunner
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.preprocessing import LabelEncoder, label_binarize

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


@dataclass()
class Histogram:
    breakpoints: list[float]
    freqs: list[int]
    target_freqs: list[tuple] = field(default_factory=list)
    classes: list = field(default_factory=list)
    densities: list[float] = field(init=False)
    target_probas: list[tuple] = field(init=False, default_factory=list)

    @classmethod
    def from_data(cls, x, y=None, method="modl", max_bins=0):
        """Computes a histogram of an 1D vector via Khiops

        Parameters
        ----------
        x : array-like of shape (n_samples,) or (n_samples, 1)
            Input scores.
        y : array-like of shape (n_samples,) or (n_samples, 1), optional
            Target values.
        method : {"modl", "eq-freq", "eq-width"}, default="modl"
            Histogram method:

            - "modl": A non-parametric regularized histogram method.
            - "eq-freq": All bins have the same number of elements. If many instances
              have too many values the algorithm will put it in its own bin, which will
              be larger than the other ones.
            - "eq-width": All bins have the same width.

            If the method is set to "eq-freq" or "eq-width" is set then 'y' is ignored.
        max_bins: int, default=0
            The maximum number of bins to be created. The algorithms usually create this
            number of bins but they may create less. The default value 0 means:

            - For "modl": that there is no limit to the number of intervals.
            - For "eq-freq" or "eq-width": that 10 is the maximum number of
              intervals.

        Returns
        -------
        `Histogram`
            The histogram object containing the bin limits and frequencies.
        """
        # Check inputs
        if len(x.shape) > 1 and x.shape[1] > 1:
            raise ValueError(f"x must be 1-D but it has shape {x.shape}.")
        if y is not None and len(y.shape) > 1 and y.shape[1] > 1:
            raise ValueError(f"y must be 1-D but it has shape {y.shape}.")
        valid_methods = ["modl", "eq-freq", "eq-width"]
        if method not in valid_methods:
            raise ValueError(f"method must be in {valid_methods}. It is '{method}'")
        if max_bins < 0:
            raise ValueError(f"max_bins must be non-negative. It is {max_bins}")

        # Set the y vector to be used by khiops
        # This is necessary because for the "eq-freq" and "eq-width" methods if
        # target is set then it uses 'modl'.
        y_khiops = y if method == "modl" else None

        # Transform the binning methods to the Khiops names
        if method == "modl":
            khiops_method = "MODL"
        elif method == "eq-freq":
            khiops_method = "EqualFrequency"
        elif method == "eq-width":
            khiops_method = "EqualWidth"
        else:
            raise ValueError(
                f"Unknown binning method '{method}'. "
                "Choose between 'modl', 'eq-freq' and 'eq-width'"
            )

        # Create Khiops dictionary
        kdom = kh.DictionaryDomain()
        kdic = kh.Dictionary()
        kdic.name = "scores"
        kdom.add_dictionary(kdic)
        var = kh.Variable()
        var.name = "x"
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
                tempfile.TemporaryDirectory(prefix="khiops-hist_")
            )
            ctx_stack.enter_context(warnings.catch_warnings())
            ctx_stack.enter_context(single_core_khiops_runner())

            # Create data table file for khiops
            df_spec = {"x": x if len(x.shape) == 1 else x.flatten()}
            if y_khiops is not None:
                df_spec["y"] = y_khiops if len(y.shape) == 1 else y.flatten()
            output_df = pd.DataFrame(df_spec)
            output_table_path = f"{work_dir}/hist-data.txt"
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
                discretization_method=khiops_method,
                max_parts=max_bins,
                do_data_preparation_only=True,
            )
            results = kh.read_analysis_results_file(f"{work_dir}/report.khj")

        # Initialize the histogram
        if y is not None:
            le = LabelEncoder()
            le.fit(y)
            classes = le.classes_.tolist()
            # Note: When building `classes` variable is important with the `tolist`
            # method, because it converts the numpy types to native Python ones.

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
            breakpoints = [
                part.lower_bound for part in data_grid.dimensions[0].partition
            ]
            breakpoints.append(data_grid.dimensions[0].partition[-1].upper_bound)

            # Create the frequencies and target frequencies
            # Supervised khiops execution
            if data_grid.is_supervised:
                # Recover the frequencies and target frequencies
                # Note: Target frequencies must be reordered to the order of `classes`
                freqs = [
                    sum(tfreqs)
                    for tfreqs in score_stats.data_grid.part_target_frequencies
                ]
                target_freqs = [
                    tuple(tfreqs[i] for i in target_indexes)
                    for tfreqs in score_stats.data_grid.part_target_frequencies
                ]
            # Unsupervised khiops execution
            else:
                freqs = score_stats.data_grid.frequencies.copy()
                if y is not None:
                    x_bin_indexes = (
                        np.searchsorted(breakpoints[:-1], x, side="left") - 1
                    )
                    x_bin_indexes[x_bin_indexes < 0] = 0
                    y_indexes = le.transform(y)
                    target_freqs = [[0 for _ in le.classes_] for _ in breakpoints[:-1]]
                    for y_score_bin_index, y_index in np.nditer(
                        [x_bin_indexes, y_indexes]
                    ):
                        target_freqs[y_score_bin_index][y_index] += 1
                    target_freqs = [tuple(freqs) for freqs in target_freqs]
        # Otherwise there is just one interval
        else:
            # Non-informative variable: histogram with only the bin (min, max)
            if score_stats.min < score_stats.max:
                breakpoints = [score_stats.min, score_stats.max]
            # Single-valued variable: histogram with only the bin (min, min + eps)
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

        return cls(
            breakpoints=breakpoints,
            freqs=freqs,
            target_freqs=target_freqs if y is not None else [],
            classes=le.classes_.tolist() if y is not None else [],
        )

    @classmethod
    def from_data_and_breakpoints(cls, x, breakpoints: list[float], y=None):
        """Builds a histogram from a list of breakpoints and data

        Parameters
        ----------
        x : array-like of shape (n_samples,) or (n_samples, 1)
            Vector with the values to discretize for the histogram.
        breakpoints : list[float]
            A sorted list of floats defining the bin edges.
        y : array-like of shape (n_samples,) or (n_samples, 1), optional
            Target values associated to each element in 'x'.
        """
        # Check inputs
        if len(x.shape) > 1 and x.shape[1] > 1:
            raise ValueError(f"x must be 1-D but it has shape {x.shape}.")
        if y is not None and len(y.shape) > 1 and y.shape[1] > 1:
            raise ValueError(f"y must be 1-D but it has shape {y.shape}.")
        for i in range(len(breakpoints) - 1):
            if (left := breakpoints[i]) > breakpoints[i + 1]:
                raise ValueError(f"Breakpoint at index {i} is not sorted: {left}")

        # Initialize the breakpoints and bin frequencies
        x_bin_indexes = np.searchsorted(breakpoints[:-1], x, side="left") - 1
        x_bin_indexes[x_bin_indexes < 0] = 0
        freqs = np.unique(x_bin_indexes, return_counts=True)[1].tolist()

        if y is None:
            return cls(breakpoints=breakpoints, freqs=freqs)
        else:
            # Obtain the class indexes of y
            le = LabelEncoder().fit(y)
            y_indexes = le.transform(y)

            # Initialize the target frequencies per bin
            target_freqs = [[0 for _ in le.classes_] for _ in breakpoints[:-1]]
            for x_bin_index, y_index in np.nditer([x_bin_indexes, y_indexes]):
                target_freqs[x_bin_index][y_index] += 1

            return cls(
                breakpoints=breakpoints,
                freqs=freqs,
                target_freqs=[tuple(freqs) for freqs in target_freqs],
                classes=le.classes_.tolist(),
            )

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
        """Returns the histogram bin index for a value"""
        if value <= self.breakpoints[0]:
            return 0
        elif value >= self.breakpoints[-1]:
            return len(self.breakpoints) - 2
        else:
            return bisect.bisect_right(self.breakpoints, value) - 1

    def vfind(self, values):
        """Returns the histogram bin indexes for a value sequence"""
        indexes = np.searchsorted(self.breakpoints[:-1], values, side="right") - 1
        indexes[np.array(values) < self.breakpoints[0]] = 0
        indexes[np.array(values) > self.breakpoints[-2]] = self.n_bins - 1
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


def ece(
    y_scores,
    y,
    method: str = "label-bin",
    histogram_method: str = "modl",
    max_bins: int = 0,
    multi_class_method: str = "top-label",
    histogram: Histogram | None = None,
):
    """Estimates the ECE for a pair of score and label vectors

    Parameters
    ----------
    y_scores : array-like of shape (n_samples,) or (n_samples, 1)
        Input scores.
    y : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target values.
    method : {"label-bin", "bin"}, default="label-bin"
        ECE estimation method. See below for details.
    multi_class_method : {"top-label", "classwise"}, default="top-label"
        Multi-class ECE estimation method:

        - "top-label": Estimates the ECE for the predicted class.
        - "classwise": Estimates the ECE for each class in a 1-vs-rest and the averages
          it.
    histogram_method : {"modl", "eq-freq", "eq-width"}, default="modl"
        Histogram method:

        - "modl": A non-parametric regularized histogram method.
        - "eq-freq": All bins have the same number of elements. If many instances
          have too many values the algorithm will put it in its own bin, which will be
          larger than the other ones.
        - "eq-width": All bins have the same width.

        If the method is set to "eq-freq" or "eq-width" is set then 'y' is ignored.
    max_bins : int, default=0
        The maximum number of bins to be created. The algorithms usually create this
        number of bins but they may create less. The default value 0 means:

        - For "modl": that there is no limit to the number of intervals.
        - For "eq-freq" or "eq-width": that 10 is the maximum number of intervals.
    histogram : `Histogram`, optional
        A ready-made histogram. If set then it is used for the ECE computation and the
        parameters histogram_method and max_bins are ignored.
    """
    if len(y_scores.shape) > 2:
        raise ValueError(
            "'y_scores' must be either a 1-D or 2-D array-like object, "
            f"but its shape is {y_scores.shape}"
        )
    # 1-D or 2-D array with 1 column: Binary ECE
    if len(y_scores.shape) == 1 or (
        len(y_scores.shape) == 2 and y_scores.shape[1] == 1
    ):
        return _binary_ece(
            y_scores,
            y,
            method=method,
            histogram_method=histogram_method,
            max_bins=max_bins,
            histogram=histogram,
        )
    # 2-D array with 2 columns: Also Binary ECE but a little pre-treatment is needed
    elif len(y_scores.shape) == 2 and y_scores.shape[1] == 2:
        # le = LabelEncoder().fit(y)
        # y_binarized = label_binarize(y, classes=le.classes_)
        return _binary_ece(
            y_scores[:, 1],
            y,
            method=method,
            histogram_method=histogram_method,
            max_bins=max_bins,
            histogram=histogram,
        )
    # 2-D array with > 2 columns: Multi-class ECE
    else:
        le = LabelEncoder().fit(y)
        if multi_class_method == "top-label":
            prediction_indexes = np.argsort(y_scores, axis=1)[:, -1]
            y_preds = (y == le.classes_[prediction_indexes]).astype(int)
            y_pred_scores = y_scores[np.arange(len(y)), prediction_indexes]
            return _binary_ece(
                y_pred_scores,
                y_preds,
                method=method,
                histogram_method=histogram_method,
                max_bins=max_bins,
                histogram=histogram,
            )
        else:
            assert multi_class_method == "classwise"
            y_binarized = label_binarize(y, classes=le.classes_)
            class_eces = [
                _binary_ece(
                    y_scores[:, k],
                    y_binarized[:, k],
                    method=method,
                    histogram_method=histogram_method,
                    max_bins=max_bins,
                    histogram=histogram,
                )
                for k in range(y_scores.shape[1])
            ]
            class_probas = np.unique(y, return_counts=True)[1] / len(y)
            return float(np.dot(class_eces, class_probas))


def _binary_ece(
    y_scores,
    y,
    method: str = "label-bin",
    histogram_method: str = "modl",
    max_bins: int = 0,
    histogram: Histogram | None = None,
):
    """Estimates the ECE for a 2-class problem

    See the function `ece` docstring for more details.
    """
    # Compute the histogram if necessary
    if histogram is None:
        histogram = Histogram.from_data(
            y_scores, y=y, method=histogram_method, max_bins=max_bins
        )

    # Check that the histogram has only two classes
    if (n_classes := len(histogram.classes)) != 2:
        raise ValueError(f"Target 'y' must have only 2 classes. It has {n_classes}.")

    # Estimate the ECE with the histogram
    if method == "label-bin":
        sum_diffs = 0
        for y_score, i in np.nditer([y_scores, histogram.vfind(y_scores)]):
            sum_diffs += math.fabs(
                y_score - histogram.target_freqs[i][1] / histogram.freqs[i]
            )
        return sum_diffs / len(y)
    else:
        assert method == "bin"
        sum_score_by_bin = [0 for _ in histogram.bins]
        for y_score, i in np.nditer([y_scores, histogram.vfind(y_scores)]):
            sum_score_by_bin[i] += y_score
        return sum(
            [
                math.fabs(sum_score_by_bin[i] - histogram.target_freqs[i][1])
                for i in range(histogram.n_bins)
            ]
        ) / len(y)


class KhiopsCalibrator(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):  # noqa: N803
        probas = self.estimator.predict_proba(X)
        n_classes = probas.shape[1]

        # Binary classification
        if n_classes == 2:
            self.histograms_ = [Histogram.from_data(probas[:, 1].reshape(-1, 1), y)]
        # Multiclass classification: One-vs-Rest
        else:
            y_binarized = label_binarize(y, classes=self.estimator.classes_)
            self.histograms_ = [
                Histogram.from_data(probas[:, k].reshape(-1, 1), y_binarized[:, k])
                for k in range(n_classes)
            ]
        return self

    def predict_proba(self, X):  # noqa: N803
        probas = self.estimator.predict_proba(X)
        n_classes = probas.shape[1]

        # Binary classification
        if n_classes == 2:
            calibrated_probas = calibrate_binary_scores(
                self.histograms_[0], probas[:, 1]
            )
        # Multiclass classification: One-vs-Rest
        else:
            binary_calibrated_probas = np.empty(probas.shape)
            for k, histogram in enumerate(self.histograms_):
                binary_calibrated_probas[:, k] = calibrate_binary_scores(
                    histogram, probas[:, k], only_positive=True
                )
            calibrated_probas = binary_calibrated_probas / np.sum(
                binary_calibrated_probas, axis=1, keepdims=True
            )

        return calibrated_probas


def calibrate_binary_scores(
    histogram: Histogram, y_scores, only_positive: bool = False
):
    y_scores_bin_indexes = histogram.vfind(y_scores.reshape(-1, 1))
    it = np.nditer(y_scores_bin_indexes, flags=["f_index"])
    if only_positive:
        calibrated_probas = np.empty(y_scores.shape[0])
        for bin_i in it:
            calibrated_probas[it.index] = histogram.target_probas[bin_i][1]
    else:
        calibrated_probas = np.empty((y_scores.shape[0], 2))
        for bin_i in it:
            calibrated_probas[it.index][0] = histogram.target_probas[bin_i][0]
            calibrated_probas[it.index][1] = histogram.target_probas[bin_i][1]

    return calibrated_probas
