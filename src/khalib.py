"""Classifier calibration with Khiops"""

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
from khiops.sklearn import KhiopsClassifier
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils._response import _get_response_values
from sklearn.utils.validation import check_is_fitted

khiops_single_core_runner: KhiopsLocalRunner | None = None


def build_single_core_khiops_runner():
    """Build a Khiops runner with one core

    Running with a single core is faster for most of the calibration use-cases.
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
    """A histogram with optional target variable statistics

    .. note::
        To obtain instances from real data, prefer the factory method `from_data` rather
        than the base constructor.

    Attributes
    ----------
    breakpoints : list[float]
        The breakpoints defining the histogram in increasing order.
    freqs : list of float
        The frequencies of each bin.
    densities : list[float]
        The densities of each bin.
    classes : list
        The class labels.
    target_freqs : list[tuple], optional
        The target frequencies in each bin. Each tuple is has the same size as
        :paramref:`classes`.
    target_probas : list[tuple], optional
        The target conditional probabilites in each bin. Each tuple is has the same size
        as :paramref:`classes`.
    """

    breakpoints: list[float]
    freqs: list[int]
    densities: list[float] = field(init=False)
    target_freqs: list[tuple] = field(default_factory=list)
    classes: list = field(default_factory=list)
    target_probas: list[tuple] = field(init=False, default_factory=list)

    @property
    def n_bins(self) -> int:
        """Number of histogram bins."""
        return max(len(self.breakpoints) - 1, 0)

    @property
    def bins(self) -> list[tuple]:
        """List of histogram bins."""
        return [tuple(self.breakpoints[i : i + 2]) for i in range(self.n_bins)]

    @property
    def classes_type(self) -> type | None:
        """Type of the target classes (only for histograms built with y)."""
        if self.classes:
            return type(self.classes[0])
        return None

    @classmethod
    def from_data(
        cls,
        x,
        y=None,
        method: str = "khiops",
        max_bins: int = 0,
        use_finest: bool = False,
    ) -> "Histogram":
        """Computes a histogram of an 1D vector via Khiops

        Parameters
        ----------
        x : array-like of shape (n_samples,) or (n_samples, 1)
            Input scores.
        y : array-like of shape (n_samples,) or (n_samples, 1), optional
            Target values.
        method : {"khiops", "eq-freq", "eq-width"}, default="khiops"
            Histogram method:

            - "khiops": A non-parametric regularized histogram method.
            - "eq-freq": All bins have the same number of elements. If many instances
              have too many values the algorithm will put it in its own bin, which will
              be larger than the other ones.
            - "eq-width": All bins have the same width.

            If the method is set to "eq-freq" or "eq-width" is set then 'y' is ignored.
        max_bins: int, default=0
            The maximum number of bins to be created. The algorithms usually create this
            number of bins but they may create less. The default value 0 means:

            - For "khiops": that there is no limit to the number of intervals.
            - For "eq-freq" or "eq-width": that 10 is the maximum number of
              intervals.
        use_finest: bool, default=False
            *Unsupervised 'khiops' histogram only*: If `True` it builds the finest
            histogram instead of the most interpretable.

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
        valid_methods = ["khiops", "eq-freq", "eq-width"]
        if method not in valid_methods:
            raise ValueError(f"method must be in {valid_methods}. It is '{method}'")
        if max_bins < 0:
            raise ValueError(f"max_bins must be non-negative. It is {max_bins}")

        # Set the y vector to be used by khiops
        # This is necessary because for the "eq-freq" and "eq-width" methods if
        # target is set then it uses 'khiops'.
        y_khiops = y if method == "khiops" else None

        # Transform the binning methods to the Khiops names
        if method == "khiops":
            khiops_method = "MODL"
        elif method == "eq-freq":
            khiops_method = "EqualFrequency"
        elif method == "eq-width":
            khiops_method = "EqualWidth"
        else:
            raise ValueError(
                f"Unknown binning method '{method}'. "
                "Choose between 'khiops', 'eq-freq' and 'eq-width'"
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
        if (dg := score_stats.data_grid) is not None:
            # Create the frequencies and target frequencies
            # Supervised khiops execution
            if dg.is_supervised:
                # Create the breakpoints
                breakpoints = [part.lower_bound for part in dg.dimensions[0].partition]
                breakpoints.append(dg.dimensions[0].partition[-1].upper_bound)

                # Recover the frequencies and target frequencies
                # Note: Target frequencies must be reordered to the order of `classes`
                freqs = [sum(tfreqs) for tfreqs in dg.part_target_frequencies]
                target_freqs = [
                    tuple(tfreqs[i] for i in target_indexes)
                    for tfreqs in dg.part_target_frequencies
                ]
            # Unsupervised khiops execution
            else:
                # Case 'eq-freq' or 'eq-width'
                if score_stats.modl_histograms is None:
                    # Create the breakpoints
                    breakpoints = [
                        part.lower_bound for part in dg.dimensions[0].partition
                    ]
                    breakpoints.append(dg.dimensions[0].partition[-1].upper_bound)

                    # Recover the frequencies
                    freqs = dg.frequencies.copy()
                # Case 'khiops': Recover the histogram from the modl_histogram field
                else:
                    if use_finest:
                        histogram_index = -1
                    else:
                        histogram_index = (
                            score_stats.modl_histograms.information_rates.index(100)
                        )
                    breakpoints = score_stats.modl_histograms.histograms[
                        histogram_index
                    ].bounds
                    freqs = score_stats.modl_histograms.histograms[
                        histogram_index
                    ].frequencies

                # Compute the supervised fields if y was provided
                if y is not None:
                    # Compute the class and  bin indexes
                    y_indexes = le.transform(y)
                    x_bin_indexes = (
                        np.searchsorted(breakpoints[:-1], x, side="left") - 1
                    ).reshape(1, -1)
                    x_bin_indexes[x_bin_indexes < 0] = 0
                    # Note: reshape is needed for the 1-column matrix case if not the
                    # iterator below does not work

                    # Compute the target frequencies
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
    def from_data_and_breakpoints(
        cls, x, breakpoints: list[float], y=None
    ) -> "Histogram":
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
        freqs = [0] * (len(breakpoints) - 1)
        for x_bin_index in x_bin_indexes:
            freqs[x_bin_index] += 1

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
                if bin_freq > 0:
                    target_probas.append(
                        tuple(tfreq / bin_freq for tfreq in bin_target_freqs)
                    )
                else:
                    target_probas.append(tuple(0 for _ in bin_target_freqs))

            # Initialize the target probabilities
            self.target_probas = target_probas

    def find(self, value: float) -> int:
        """Returns the histogram bin index for a value

        Parameters
        ----------
        value : float
            The value to look up in the histogram bins.

        Returns
        -------
        int
            The index of the bin containing the value.
        """
        if value <= self.breakpoints[0]:
            return 0
        elif value >= self.breakpoints[-1]:
            return len(self.breakpoints) - 2
        else:
            return bisect.bisect_right(self.breakpoints, value) - 1

    def vfind(self, values):
        """Returns the histogram bin indexes for a value sequence

        Parameters
        ----------
        value : :external:term:`array-like`
            The values to look up in the histogram bins.

        Returns
        -------
        :external:term:`array-like`
            The indexes of the bins containing the values.
        """
        indexes = np.searchsorted(self.breakpoints[:-1], values, side="right") - 1
        indexes[np.array(values) < self.breakpoints[0]] = 0
        indexes[np.array(values) > self.breakpoints[-2]] = self.n_bins - 1
        return indexes


def calibration_error(
    y_scores,
    y,
    method: str = "label-bin",
    histogram_method: str = "khiops",
    max_bins: int = 0,
    multi_class_method: str = "top-label",
    histogram: Histogram | None = None,
):
    r"""Estimates the ECE via binning

    The default binning method "khiops" finds a regularized histogram for which the
    distribution of the target as pure as possible.

    Parameters
    ----------
    y_scores : array-like of shape (n_samples,) or (n_samples, 1) or (n_samples, 2)
        Scores/probabilities. If it is a 2-D array the column indexed as 1 will be used
        for the estimation.
    y : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target values (class labels).
    method : {"label-bin", "bin"}, default="label-bin"
        ECE estimation method. See below for details.
    multi_class_method : {"top-label", "classwise"}, default="top-label"
        Multi-class ECE estimation method:

        - "top-label": Estimates the ECE for the confidence scores (ie. the predicted
          class score).
        - "classwise": Estimates the ECE for each class in a one-vs-rest fashion and the
          averages it.
    histogram_method : {"khiops", "eq-freq", "eq-width"}, default="khiops"
        Histogram method:

        - "khiops": A non-parametric regularized histogram method. It finds the best
          histogram such that the distribution of the target is constant in each bin.
        - "eq-freq": All bins have the same number of elements. If many instances
          have too many values the algorithm will put it in its own bin, which will be
          larger than the other ones.
        - "eq-width": All bins have the same width.

        If the method is set to "eq-freq" or "eq-width" is set then 'y' is ignored.
    max_bins : int, default=0
        The maximum number of bins to be created. The algorithms usually create this
        number of bins but they may create less. The default value 0 means:

        - For "khiops": that there is no limit to the number of intervals.
        - For "eq-freq" or "eq-width": that 10 is the maximum number of intervals.
    histogram : `Histogram`, optional
        A ready-made histogram. If set then it is used for the ECE computation and the
        parameters histogram_method and max_bins are ignored.


    Notes
    -----
    We present the formulas for the two binary ECE estimators used in this function, as
    described in [RCSM22]_. Both approximate the theoretical calibration error:

    .. math::
        \mathbb{E}\left[ \left| \mathbb{P}\left[ Y = 1 | S \right] - S  \right| \right]

    where the pair :math:`(S, Y)` is a random vector taking values in :math:`[0, 1]
    \times \lbrace{0, 1\rbrace}`. The random variable :math:`S` represent the
    (normalized) scores coming from a classifier, and :math:`Y` the target.

    Let
    :math:`\lbrace {B_i} \rbrace _{i=I}^I` a binning of :math:`[0,1]` and :math:`\lbrace
    (s_n, y_n) \rbrace_{n=1}^N` a sample following the distribution of :math:`(S, Y)`.


    The **label-bin** ECE estimation is given by:

    .. math::
        \text{ECE}_{\text{lb}} = \frac{1}{N} \sum_{n=1}^N \sum_{i = 1}^I
           \mathbb{1}_{s_n \in B_i} \left| s_n - \bar{y_i} \right|

    where :math:`\bar{y_i}` is the sample conditional expectation of :math:`Y` in the
    :math:`i`-th bin:

    .. math::
        \frac{\sum_{n=1}^N \mathbb{1}_{s_n \in B_i} \cdot y_n}{|B_i|}.

    On the other hand, the **bin** ECE estimation is given by:

    .. math::
        \text{ECE}_{\text{bin}} = \sum_{i = 1}^I
           \frac{\left| B_i \right| }{N} \left| \bar{s_i} - \bar{y_i} \right|

    where :math:`\bar{y_i}` is as above, and :math:`\bar{s_i}` is the sample
    conditional expectation of :math:`S` in the :math:`i`-th bin:

    .. math::
        \frac{\sum_{n=1}^N \mathbb{1}_{s_n \in B_i} \cdot s_n}{|B_i|}.


    The **bin** estimator is a lower bound for the real ECE for any binning used. And,
    for a given binning, it is a lower bound of the **label-bin** estimator.

    The **bin** estimator is the most common literature, usually called the *plugin*
    estimator, or even shown as the defitinion ECE. Despite this fact, we choose the
    **label-bin** estimator as default method because it is exact when the conditional
    distribution :math:`\mathbb{P}\left[ Y = 1 | S = s \right]` is piecewise constant on
    :math:`s`. These piecewise constant models are exactly the hypothesis space of any
    histogram estimation of the ECE.

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
    histogram_method: str = "khiops",
    max_bins: int = 0,
    histogram: Histogram | None = None,
):
    """Estimates the ECE for a 2-class problem

    See the function `calibration_error` docstring for more details.
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


class KhalibClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Classifier calibration via Khiops

    This estimator uses the Khiops regularized histogram construction method (MODL) to
    calibrate the probabilities/scores coming from a classifier.

    For binary classifiers, it fits a Khiops histogram for the scores for the positive
    class. Then, for a given score, it estimates its calibrated probability as the
    positive class probability of the bin in which the score falls.

    For multi-class classifiers, it fits khiops histogram for each class in a
    one-vs-rest fashion. Then, given a score vector, it estimates its calibrated
    probabilities by applying in each one-vs-rest the binary method described above and
    then normalizing to one.

    This class emulates closely the `~sklearn.calibration.CalibratedClassifierCV`
    interface but without cross-validation, as the Khiops models are regularized and do
    not need it.

    .. _KhiopsClassifier: https://khiops.org/11.0.0-b.0/api-docs/python-api/api/sklearn/generated/khiops.sklearn.estimators.html#khiops.sklearn.estimators.KhiopsClassifier

    Parameters
    ----------
    estimator : estimator instance, default=None
        The classifier whose output need to be calibrated to provide more accurate
        `predict_proba` outputs. If not provided, it trains a `KhiopsClassifier`_. If
        provided, but the estimator is not fitted then, on `fit`, it will
        be fitted with :paramref:`train_size` elements.
    train_size : float or int, default=0.75
        Same parameter as `~sklearn.model_selection.train_test_split`. Only used when
        :paramref:`estimator` is not provided or not fitted.
    random_state : int, default=None
        Same parameter as `~sklearn.model_selection.train_test_split`. Only used when
        :paramref:`estimator` is not provided or not fitted.


    Attributes
    ----------
    fitted_estimator_ :
        The underlying fitted classifier. It is the same instance as
        :paramref:`estimator` if it was provided, otherwise it is an instance of
        `KhiopsClassifier`_.
    histograms_ : list of size 1 or n_classes
        In the binary case: a list of size 1 with the `Histogram` instance for the
        positive class. In the multi class case: A list of size n_classes containing the
        one-vs-rest `Histogram` for each class.

    """

    def __init__(self, estimator=None, train_size=0.75, random_state=None):
        """See class docstring"""
        self.estimator = estimator
        self.train_size = train_size
        self.random_state = random_state

    def fit(self, X, y):  # noqa: N803
        """Fits the calibrated model

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
            The calling estimator instance.
        """
        # Check/initialize fitted estimator
        if self.estimator is None:
            self.fitted_estimator_ = KhiopsClassifier()
        else:
            self.fitted_estimator_ = self.estimator

        # Fit the base predictor if not fitted
        try:
            check_is_fitted(self.fitted_estimator_)
            x_calib, y_calib = X, y
        except NotFittedError:
            x_train, x_calib, y_train, y_calib = train_test_split(
                X, y, train_size=self.train_size, random_state=self.random_state
            )
            self.fitted_estimator_.fit(x_train, y_train)

        # Estimate the uncalibrated scores
        scores, _ = _get_response_values(
            self.fitted_estimator_,
            x_calib,
            response_method=["predict_proba", "decision_function"],
        )

        # Binary classification
        if (n_classes := len(self.fitted_estimator_.classes_)) == 2:
            self.histograms_ = [Histogram.from_data(scores, y_calib)]
        # Multiclass classification: One-vs-Rest
        else:
            y_binarized = label_binarize(
                y_calib, classes=self.fitted_estimator_.classes_
            )
            self.histograms_ = [
                Histogram.from_data(scores[:, k].reshape(-1, 1), y_binarized[:, k])
                for k in range(n_classes)
            ]

            calibrated_probas = np.empty(scores.shape)
            for k, histogram in enumerate(self.histograms_):
                calibrated_probas[:, k] = calibrate_binary(
                    scores[:, k], histogram, only_positive=True
                )
        return self

    def predict_proba(self, X):  # noqa: N803
        """Estimates the calibrated classification probabilities

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        P : array-like of shape (n_samples, n_classes)
            The estimated calibrated probabilities for each class.
        """
        # Estimate the uncalibrated scores
        scores, _ = _get_response_values(
            self.fitted_estimator_,
            X,
            response_method=["predict_proba", "decision_function"],
        )

        # Binary classification
        if len(self.histograms_) == 1:
            calibrated_probas = calibrate_binary(scores, self.histograms_[0])
        # Multiclass classification: One-vs-Rest
        else:
            calibrated_probas = np.empty(scores.shape)
            for k, histogram in enumerate(self.histograms_):
                calibrated_probas[:, k] = calibrate_binary(
                    scores[:, k], histogram, only_positive=True
                )

            calibrated_probas = calibrated_probas / np.sum(
                calibrated_probas, axis=1, keepdims=True
            )

        return calibrated_probas


def calibrate_binary(y_scores, histogram: Histogram, only_positive: bool = False):
    """Calibrates a binary score with an histogram

    This function may be used to calibrate binary classification scores without the use
    of the `KhalibClassifier` estimator.

    Parameters
    ----------
    y_scores : array-like of shape (n_samples,) or (n_samples, 1)
        Scores for the positive class.
    histogram : `Histogram`
        A histogram with target statistics (ie. :paramref:`Histogram.target_freqs` must
        be non-empty).
    only_positive : bool, default=False
        If ``True`` it returns only 1-D array with the probabilities of the positive
        class. Otherwise it returns a 2-D array with the probabilities for both classes.
    """
    # Check the inputs
    if not histogram.classes:
        raise ValueError("'histogram' must have target statistics.")
    if (n_classes := len(histogram.classes)) != 2:
        raise ValueError(f"'histogram' must have only 2 classes (it has {n_classes})")

    # Map the scores to their corresponding bin target probability
    y_scores_bin_indexes = histogram.vfind(y_scores)
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


def build_reliability_diagram(
    y_scores,
    y,
    dirac_threshold=1.0e-06,
    log_plot_threshold=3.0,
    min_density_bar_width=2.5e-03,
):
    """Builds a reliability diagram with the target score distribution below

    To build the diagram this function uses Khiops supervised histograms. To build the
    score distribution, it uses Khiops unsupervised histograms. Using the latter, it
    implements a heuristic to dectect when the scores is distributed as a sum of diracs
    and changes visualization accordingly.


    Parameters
    ----------
    y_scores : array-like of shape (n_samples,) or (n_samples, 1) or (n_samples, 2)
        Scores/probabilities. If it is a 2-D array the column indexed as 1 will be used
        for the estimation.
    y : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target values (class labels).
    dirac_threshold : float, default=1.0e-06
        If a bin in the scores' unsupervised histogram is lower than this
        'dirac_threshold' then it is considered a dirac mass.
    log_plot_threshold : float, default=3.0
        *Density plot only:* If the log-difference between the maximal and minimal
        positive density values is larger than 'log_plot_threshold' then the density
        plot uses a log scale in the y-axis.
    min_density_bar_width : float, default=5.0e-03
        *Density plot only:* If a bin of the scores' unsupervised histogram has a width
        lower than 'min_density_bar_width' then it is plotted as having a width of
        'min_density_bar_width'.

    Returns
    -------
    tuple
        A 2-tuple containing:

        - A `matplotlib.figure.Figure`.
        - A `dict` containing two `matplotlib.axes.Axes` with keys, "reliability
          diagram" and "score_distribution".
    """
    # Check the input parameters
    if len(y_scores.shape) > 2:
        raise ValueError(
            "'y_scores' must be either a 1-D or 2-D array-like object, "
            f"but its shape is {y_scores.shape}"
        )
    if len(y.shape) > 1:
        raise ValueError(
            f"'y_scores' must be a 1-D array-like object, but its shape is {y.shape}"
        )
    if dirac_threshold <= 0.0:
        raise ValueError("'dirac_threshold' must be positive")
    if log_plot_threshold <= 0.0:
        raise ValueError("'log_plot_threshold' must be positive")
    if min_density_bar_width <= 0.0:
        raise ValueError("'min_density_bar_width' must be positive")

    # Create the base plot
    fig, axs = plt.subplot_mosaic(
        [["reliability_diagram"], ["score_distribution"]],
        figsize=(6, 6),
        height_ratios=(4, 1),
        sharex=True,
    )
    fig.subplots_adjust(hspace=0)
    fig.suptitle("Reliability Diagram")

    # Build a unsupervised histogram to to detect the dirac masses case
    uhist_y_scores = Histogram.from_data(y_scores, use_finest=True)
    dirac_indexes = []
    if uhist_y_scores.freqs[1] == 0 and (
        uhist_y_scores.bins[0][1] - uhist_y_scores.bins[0][0] < dirac_threshold
    ):
        dirac_indexes.append(True)
    else:
        dirac_indexes.append(False)
    for i in range(1, uhist_y_scores.n_bins - 1):
        cur_left, cur_right = uhist_y_scores.bins[i]
        if (
            uhist_y_scores.freqs[i - 1] == 0
            and uhist_y_scores.freqs[i + 1] == 0
            and (cur_right - cur_left < dirac_threshold)
        ):
            dirac_indexes.append(True)
        else:
            dirac_indexes.append(False)
    if uhist_y_scores.freqs[-2] == 0 and (
        uhist_y_scores.bins[-1][1] - uhist_y_scores.bins[-1][0] < dirac_threshold
    ):
        dirac_indexes.append(True)
    else:
        dirac_indexes.append(False)

    # Compute the supervised score histogram
    hist_y_scores = Histogram.from_data(y_scores, y)

    # Case where are only dirac masses
    n_nonzero_bins = sum(freq > 0 for freq in uhist_y_scores.freqs)
    if sum(dirac_indexes) == n_nonzero_bins:
        # Compute the dirac masses plot data
        dirac_masses = [
            freq / sum(uhist_y_scores.freqs)
            for freq, is_dirac in zip(uhist_y_scores.freqs, dirac_indexes, strict=False)
            if is_dirac
        ]
        dirac_locations = [
            (left + right) / 2
            for (left, right), is_dirac in zip(
                uhist_y_scores.bins, dirac_indexes, strict=False
            )
            if is_dirac
        ]
        indexes = hist_y_scores.vfind(dirac_locations)
        target_probas = [hist_y_scores.target_probas[i][1] for i in indexes]
        axs["reliability_diagram"].scatter(
            dirac_locations, target_probas, color="tab:red", alpha=0.8
        )
        axs["reliability_diagram"].plot(
            (0, 1), (0, 1), color="tab:grey", linestyle="dotted"
        )
        for loc, mass in zip(dirac_locations, dirac_masses, strict=False):
            axs["score_distribution"].annotate(
                "",
                xy=(loc, mass),
                xytext=(loc, 0.0),
                label="dirac masses",
                arrowprops={
                    "arrowstyle": "-|>",
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "color": "tab:blue",
                },
            )
        axs["score_distribution"].grid(color="gainsboro")
        axs["score_distribution"].set_xlabel("Score Positive Class")
        axs["score_distribution"].set_ylabel("Probability")
    # Case where there are only proper densities or a mix of both
    else:
        # Plot the reliability diagram
        x = []
        for intv in hist_y_scores.bins:
            x.extend(intv)
        x[0] = 0
        x[-1] = 1
        y = []
        for probas in hist_y_scores.target_probas:
            y.extend((probas[1], probas[1]))
        axs["reliability_diagram"].plot(x, y)
        axs["reliability_diagram"].plot(x, x, linestyle="dotted", color="gray")
        axs["reliability_diagram"].fill_between(x, x, y, color="tab:red", alpha=0.2)
        axs["reliability_diagram"].set_ylabel("Positive Class Probability")
        axs["reliability_diagram"].grid(color="gainsboro")

        # Recompute the density, this time we took the most interpretable histogram
        uhist_y_scores = Histogram.from_data(y_scores)

        # Plot the density
        density_bar_starts = [
            left
            for (left, _), freq in zip(
                uhist_y_scores.bins, uhist_y_scores.freqs, strict=True
            )
            if freq > 0
        ]
        density_bar_widths = [
            right - left
            if right - left >= min_density_bar_width
            else min_density_bar_width
            for (left, right), freq in zip(
                uhist_y_scores.bins, uhist_y_scores.freqs, strict=True
            )
            if freq > 0
        ]
        density_bar_heights = [
            density
            for density, freq in zip(
                uhist_y_scores.densities,
                uhist_y_scores.freqs,
                strict=True,
            )
            if freq > 0
        ]
        density_bar_log_range = None
        if density_bar_heights:
            density_bar_log_range = math.log10(max(density_bar_heights)) - math.log10(
                min(density_bar_heights)
            )
        axs["score_distribution"].bar(
            x=density_bar_starts,
            height=density_bar_heights,
            width=density_bar_widths,
            align="edge",
            label="Density",
        )
        axs["score_distribution"].grid(color="gainsboro")
        axs["score_distribution"].set_xlabel("Score Positive Class")
        axs["score_distribution"].set_ylabel("Density")
        if density_bar_log_range > log_plot_threshold:
            axs["score_distribution"].set_yscale("log")
            axs["score_distribution"].yaxis.set_major_locator(
                LogLocator(base=10, subs=[1], numticks=5)
            )

    return fig, axs
