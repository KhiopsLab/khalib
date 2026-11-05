"""Classifier calibration with Khiops"""

import bisect
import math
import os
import sys
import tempfile
import threading
import warnings
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from khiops import core as kh
from khiops.core.internals.runner import KhiopsLocalRunner
from khiops.sklearn import KhiopsClassifier
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils._response import _get_response_values
from sklearn.utils.validation import check_consistent_length, check_is_fitted

khiops_single_core_runner: KhiopsLocalRunner | None = None
_khiops_runner_lock = threading.RLock()


def build_single_core_khiops_runner():
    """Build a Khiops runner with one core

    Running with a single core is faster for most of the calibration use-cases.
    The process core limit is restored even if runner initialization fails.
    """
    with _khiops_runner_lock:
        default_max_cores = os.environ.get("KHIOPS_PROC_NUMBER")
        os.environ["KHIOPS_PROC_NUMBER"] = "1"
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", message=".*Too few cores:.*")
                global khiops_single_core_runner
                khiops_single_core_runner = KhiopsLocalRunner()
        finally:
            if default_max_cores is None:
                del os.environ["KHIOPS_PROC_NUMBER"]
            else:
                os.environ["KHIOPS_PROC_NUMBER"] = default_max_cores


@contextmanager
def single_core_khiops_runner():
    """A context manager to run Khiops with a single core"""
    with _khiops_runner_lock:
        if khiops_single_core_runner is None:
            build_single_core_khiops_runner()
        default_runner = kh.get_runner()
        kh.set_runner(khiops_single_core_runner)
        try:
            yield
        finally:
            kh.set_runner(default_runner)


def _check_real_values(values, name):
    """Rejects sparse or complex values."""
    if issparse(values):
        raise ValueError(f"'{name}' must not be a sparse matrix.")
    values = np.asarray(
        values, dtype=None if isinstance(values, np.ndarray) else object
    )
    if values.dtype.kind == "c" or (
        values.dtype.kind == "O"
        and any(
            isinstance(value, (complex, np.complexfloating)) for value in values.flat
        )
    ):
        raise ValueError(f"'{name}' must contain only real values.")


def _check_classifier_features(values):
    """Allows sparse real feature matrices for scikit-learn estimators."""
    if issparse(values):
        if np.issubdtype(values.dtype, np.complexfloating):
            raise ValueError("'X' must contain only real values.")
    else:
        _check_real_values(values, "X")


def _as_1d_array(values, name):
    """Converts a vector or single-column matrix to a 1-D array."""
    _check_real_values(values, name)
    values = np.asarray(values)
    if values.ndim != 1 and not (values.ndim == 2 and values.shape[1] == 1):
        raise ValueError(
            f"{name} must be 1-D or a single-column matrix, "
            f"but it has shape {values.shape}."
        )
    return values.reshape(-1)


def _as_score_array(values, name):
    """Validates real scores while preserving finite-range integer precision."""
    _check_real_values(values, name)
    values = np.asarray(
        values, dtype=None if isinstance(values, np.ndarray) else object
    )
    _check_no_missing_values(values, name)
    if values.dtype.kind in "SU":
        raise ValueError(f"'{name}' must contain only numeric values.")
    if values.dtype.kind == "b":
        raise ValueError(f"'{name}' must not contain boolean values.")
    integer_scores = values.dtype.kind in "iu"
    if values.dtype.kind == "O":
        integer_scores = True
        for value in values.flat:
            if isinstance(value, (str, bytes, np.str_, np.bytes_)):
                raise ValueError(f"'{name}' must contain only numeric values.")
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"'{name}' must not contain boolean values.")
            integer_scores &= isinstance(value, (int, np.integer))
    if values.dtype.kind in "iu":
        values = values.astype(object)
    elif integer_scores:
        values = np.array([int(value) for value in values.flat], dtype=object).reshape(
            values.shape
        )

    try:
        float_values = np.asarray(values, dtype=float)
    except OverflowError as error:
        raise ValueError(
            f"'{name}' must contain values within the finite floating-point range."
        ) from error
    return values if integer_scores else float_values


def _check_no_infinite_values(values, name):
    """Rejects infinite values without converting arbitrary-size integers."""
    if any(value in (math.inf, -math.inf) for value in values.flat):
        raise ValueError(f"'{name}' must not contain infinite values.")


def _check_no_missing_values(values, name):
    """Rejects NaN or missing values."""
    if pd.isna(np.asarray(values, dtype=object)).any():
        raise ValueError(f"'{name}' must not contain NaN or missing values.")


def _validate_max_bins(max_bins):
    """Validates and normalizes a nonnegative integer bin limit."""
    _check_real_values(max_bins, "max_bins")
    if isinstance(max_bins, (bool, np.bool_)) or not isinstance(
        max_bins, (int, np.integer)
    ):
        raise ValueError("'max_bins' must be a nonnegative integer.")
    if max_bins < 0:
        raise ValueError(f"'max_bins' must be nonnegative. It is {max_bins}.")
    return int(max_bins)


def _find_bin_indexes(breakpoints, values):
    """Finds left-open, right-closed bins with exact integer comparisons."""
    values = np.asarray(values)
    if values.dtype.kind in "iuO" or any(
        isinstance(value, (int, np.integer)) for value in breakpoints
    ):
        boundaries = np.asarray(breakpoints, dtype=object)
        values = np.asarray(values, dtype=object)
    else:
        boundaries = np.asarray(breakpoints)
    indexes = np.searchsorted(boundaries[:-1], values, side="left") - 1
    return np.clip(indexes, 0, len(breakpoints) - 2)


def _validate_nonempty_bin_scores(histogram, y_scores):
    """Rejects score samples that map to empty bins."""
    bin_indexes = histogram.vfind(y_scores)
    empty_bin_samples = np.flatnonzero(np.asarray(histogram.freqs)[bin_indexes] == 0)
    if empty_bin_samples.size:
        raise ValueError(
            "Cannot compute ECE for scores in empty histogram bins: "
            f"sample indices {empty_bin_samples.tolist()}, "
            f"bin indices {bin_indexes[empty_bin_samples].tolist()}."
        )
    return bin_indexes


def _get_decision_function_shape(estimator):
    """Returns the decision score layout through pipelines and fitted searches."""
    while isinstance(estimator, (Pipeline, BaseSearchCV)):
        if isinstance(estimator, Pipeline):
            estimator = estimator.steps[-1][1]
        else:
            estimator = estimator.best_estimator_
    return getattr(estimator, "decision_function_shape", None)


@dataclass()
class Histogram:
    """A histogram with optional target variable statistics

    .. note::
        To obtain instances from real data, prefer the factory method `from_data` rather
        than the base constructor.

    Attributes
    ----------
    breakpoints : list[float]
        The breakpoints defining the histogram in increasing order. NumPy integer
        endpoints are stored as Python integers to avoid fixed-width overflow.
        Integer scores and endpoints retain exact comparisons during bin lookup.
    freqs : list of float
        The finite, nonnegative frequencies of each bin.
    densities : list[float]
        The densities of each bin. Positive infinite densities caused by overflow
        are capped at the largest finite float. For capped bins, density times
        bin width no longer equals the bin's probability mass. Widths exceeding
        the floating-point range are scaled to preserve representable densities.
    classes : list
        The class labels.
    target_freqs : list[tuple], optional
        The finite, nonnegative target frequencies in each bin. Each tuple has the
        same size as :paramref:`classes`.
    target_probas : list[tuple], optional
        The target conditional probabilites in each bin. Each tuple is has the same size
        as :paramref:`classes`.

    Raises
    ------
    ValueError
        Raised in the following cases:

        - Bin or target frequencies are negative or nonfinite.
        - Breakpoints, frequencies, target frequencies, or classes contain
          complex values.
        - Bin frequencies have a zero or nonfinite total.
        - Fewer than two breakpoints are supplied.
        - Breakpoints contain NaN or infinite values.
        - Classes contain NaN or missing values.
        - Breakpoints are not strictly increasing.
        - Bin frequencies, target frequencies, or classes have inconsistent sizes.
        - Target frequencies do not sum to their bin frequency.
        - Breakpoints contain infinite values.
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
            histogram instead of the most interpretable. If no histogram has an
            information rate of 100, the last histogram is used.

        Returns
        -------
        `Histogram`
            The histogram object containing the bin limits and frequencies.

        Raises
        ------
        ValueError
            Raised in the following cases:

            - x contains NaN or missing values.
            - x contains nonnumeric values.
            - x contains boolean values.
            - x contains infinite values.
            - x contains values outside the finite floating-point range.
            - x contains no samples.
            - x or y contains complex values.
            - y contains NaN or missing values.
            - max_bins is not a nonnegative integer.
            - x and y have different numbers of samples.
        TypeError
            `use_finest` is not a boolean.
        """
        # Check inputs
        if not isinstance(use_finest, (bool, np.bool_)):
            raise TypeError("'use_finest' must be a boolean.")
        max_bins = _validate_max_bins(max_bins)
        x = _as_1d_array(_as_score_array(x, "x"), "x")
        _check_no_infinite_values(x, "x")
        if y is not None:
            y = _as_1d_array(y, "y")
            _check_no_missing_values(y, "y")
            check_consistent_length(x, y)
        if x.size == 0:
            raise ValueError("'x' must contain at least one sample.")
        valid_methods = ["khiops", "eq-freq", "eq-width"]
        if method not in valid_methods:
            raise ValueError(f"method must be in {valid_methods}. It is '{method}'")
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
            df_spec = {"x": x}
            if y_khiops is not None:
                df_spec["y"] = y_khiops
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
            target_indexes = np.argsort(le.transform(casted_target_values))

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
                    elif 100 in score_stats.modl_histograms.information_rates:
                        histogram_index = (
                            score_stats.modl_histograms.information_rates.index(100)
                        )
                    else:
                        histogram_index = -1
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
                    x_bin_indexes = _find_bin_indexes(breakpoints, x)

                    # Compute the target frequencies
                    target_freqs = [[0 for _ in le.classes_] for _ in breakpoints[:-1]]
                    for y_score_bin_index, y_index in np.nditer(
                        [x_bin_indexes, y_indexes]
                    ):
                        target_freqs[y_score_bin_index][y_index] += 1
                    target_freqs = [tuple(tf) for tf in target_freqs]
        # Otherwise there is just one interval
        else:
            # Non-informative variable: histogram with only the bin (min, max)
            if score_stats.min < score_stats.max:
                breakpoints = [score_stats.min, score_stats.max]
            # Keep constant-score endpoints distinct despite floating-point rounding
            else:
                value = score_stats.min
                upper = math.nextafter(value, math.inf)
                if math.isfinite(upper):
                    breakpoints = [value, upper]
                else:
                    breakpoints = [math.nextafter(value, -math.inf), value]

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
            At least two sorted floats defining the bin edges, without NaN values.
        y : array-like of shape (n_samples,) or (n_samples, 1), optional
            Target values associated to each element in 'x'.

        Raises
        ------
        ValueError
            Raised in the following cases:

            - x contains NaN or missing values.
            - x contains nonnumeric values.
            - x contains boolean values.
            - x contains infinite values.
            - x contains values outside the finite floating-point range.
            - x contains no samples.
            - x, y, or breakpoints contains complex values.
            - y contains NaN or missing values.
            - x and y have different numbers of samples.
            - Breakpoints contain NaN or infinite values.
            - Fewer than two breakpoints are supplied.
        """
        # Check inputs
        x = _as_1d_array(_as_score_array(x, "x"), "x")
        _check_no_infinite_values(x, "x")
        if y is not None:
            y = _as_1d_array(y, "y")
            _check_no_missing_values(y, "y")
            check_consistent_length(x, y)
        if x.size == 0:
            raise ValueError("'x' must contain at least one sample.")
        if len(breakpoints) < 2:
            raise ValueError("`breakpoints` must contain at least two values.")
        _check_real_values(breakpoints, "breakpoints")
        for i, breakpoint in enumerate(breakpoints):
            if not isinstance(breakpoint, (int, np.integer)) and math.isnan(breakpoint):
                raise ValueError(f"`breakpoints` contains NaN at index {i}.")
            if not isinstance(breakpoint, (int, np.integer)) and math.isinf(breakpoint):
                raise ValueError(f"`breakpoints` contains infinity at index {i}.")
        for i in range(len(breakpoints) - 1):
            if (left := breakpoints[i]) > breakpoints[i + 1]:
                raise ValueError(f"Breakpoint at index {i} is not sorted: {left}")

        # Initialize the breakpoints and bin frequencies
        x_bin_indexes = _find_bin_indexes(breakpoints, x)
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
        # Reject complex inputs before comparisons or arithmetic can discard them
        for name in ("breakpoints", "freqs", "target_freqs", "classes"):
            _check_real_values(getattr(self, name), name)
        _check_no_missing_values(self.classes, "classes")

        # Preserve exact integer endpoints without fixed-width arithmetic
        self.breakpoints = [
            int(value) if isinstance(value, np.integer) else value
            for value in self.breakpoints
        ]

        # Check consistency of the constructor parameters
        if len(self.breakpoints) < 2:
            raise ValueError("`breakpoints` must contain at least two values.")
        for i, breakpoint in enumerate(self.breakpoints):
            if not isinstance(breakpoint, int) and math.isnan(breakpoint):
                raise ValueError(f"`breakpoints` contains NaN at index {i}.")
            if not isinstance(breakpoint, int) and math.isinf(breakpoint):
                raise ValueError(f"`breakpoints` contains infinity at index {i}.")
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
        for i, freq in enumerate(self.freqs):
            if not isinstance(freq, (int, np.integer)) and not math.isfinite(freq):
                raise ValueError(
                    f"`freqs` must be finite, but at bin index {i} got {freq}."
                )
            if freq < 0:
                raise ValueError(
                    f"`freqs` must be nonnegative, but at bin index {i} got {freq}."
                )
        if self.target_freqs:
            if len(set(self.classes)) != len(self.classes):
                raise ValueError(
                    "`classes` must contain unique labels when `target_freqs` are "
                    "provided."
                )
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
                for k, tfreq in enumerate(tfreqs):
                    if not isinstance(tfreq, (int, np.integer)) and not math.isfinite(
                        tfreq
                    ):
                        raise ValueError(
                            "`target_freqs` must be finite, "
                            f"but at bin index {i}, class index {k} got {tfreq}."
                        )
                    if tfreq < 0:
                        raise ValueError(
                            "`target_freqs` must be nonnegative, "
                            f"but at bin index {i}, class index {k} got {tfreq}."
                        )
                target_total = sum(
                    int(freq) if isinstance(freq, np.integer) else freq
                    for freq in tfreqs
                )
                if target_total != self.freqs[i]:
                    raise ValueError(
                        f"`target_freqs` at bin index {i} sums different from the bin "
                        f"frequency: {target_total} != {self.freqs[i]}"
                    )

        # Promote NumPy integers to avoid fixed-width overflow during summation
        total_freq = sum(
            int(freq) if isinstance(freq, np.integer) else freq for freq in self.freqs
        )
        if not isinstance(total_freq, int) and not math.isfinite(total_freq):
            raise ValueError("`freqs` must have a finite total frequency.")
        if total_freq == 0:
            raise ValueError("`freqs` must have a positive total frequency.")

        # Initialize the densities and target probabilities
        self.densities = []
        for i, freq in enumerate(self.freqs):
            left, right = self.breakpoints[i : i + 2]
            probability = freq / total_freq
            # Halve finite endpoints when their difference would overflow
            if (
                not isinstance(left, int)
                and not isinstance(right, int)
                and math.isfinite(left)
                and math.isfinite(right)
                and left < 0 < right
                and right > sys.float_info.max + left
            ):
                density = (probability / 2) / (right / 2 - left / 2)
            else:
                width = right - left
                if isinstance(width, int) and width.bit_length() > 1023:
                    shift = width.bit_length() - 1023
                    density = math.ldexp(probability / (width >> shift), -shift)
                else:
                    density = probability / width
            self.densities.append(
                sys.float_info.max if density == math.inf else density
            )
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

        Raises
        ------
        ValueError
            If value is nonnumeric, NaN, missing, boolean, infinite, complex,
            sparse, outside the finite floating-point range, or is not scalar.
        """
        value = _as_score_array(value, "value")
        if value.ndim != 0:
            raise ValueError("'value' must be a scalar.")
        _check_no_infinite_values(value, "value")
        value = value.item()
        if isinstance(value, np.integer):
            value = int(value)
        if value <= self.breakpoints[0]:
            return 0
        elif value >= self.breakpoints[-1]:
            return len(self.breakpoints) - 2
        else:
            return bisect.bisect_left(self.breakpoints, value) - 1

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

        Raises
        ------
        ValueError
            If values contain nonnumeric, NaN, missing, boolean, infinite,
            complex, or out-of-range values.
        """
        values = _as_score_array(values, "values")
        _check_no_infinite_values(values, "values")
        return _find_bin_indexes(self.breakpoints, values)


def calibration_error(
    y_scores,
    y,
    method: str = "label-bin",
    histogram_method: str = "khiops",
    max_bins: int = 0,
    multi_class_method: str = "top-label",
    histogram: Histogram | None = None,
    reuse_stats: bool = False,
):
    r"""Estimates the ECE via binning

    The default binning method "khiops" finds a regularized histogram for which the
    distribution of the target as pure as possible.

    Parameters
    ----------
    y_scores : array-like of shape (n_samples,) or (n_samples, n_classes)
        Scores/probabilities. For binary ECE, the positive class is the second
        distinct label in sorted order. For two-column binary scores, column 1
        must contain scores for that class.
        For multiclass scores, columns must correspond to the numerical or
        lexicographical order of the labels in y, with every class present in y.
        Single-column binary scores are also accepted.
    y : array-like of shape (n_samples,) or (n_samples, 1)
        Target values (class labels).
    method : {"label-bin", "bin"}, default="label-bin"
        ECE estimation method. See below for details.
    multi_class_method : {"top-label", "classwise"}, default="top-label"
        Multi-class ECE estimation method:

        - "top-label": Estimates the ECE for the confidence scores (ie. the predicted
          class score). Ties select the first maximal score column.
          Requires both correct and incorrect predictions.
        - "classwise": Estimates the ECE for each class in a one-vs-rest fashion and
          averages the class ECEs weighted by each class's prevalence in `y`.
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
        A histogram providing the bin boundaries. If set, histogram_method and
        max_bins are ignored. Bin and target frequencies are recomputed from
        y_scores and y unless reuse_stats is True. When reusing binary statistics,
        the histogram's classes must match the sorted distinct labels in y.
    reuse_stats : bool, default=False
        Reuse the supplied histogram's statistics without recomputing them. Set
        this to True only if it was built from the same scores and labels used
        for this ECE computation. Multiclass score matrices are rejected when
        this is True. For binary scores, it has no effect when histogram is None.

    Raises
    ------
    ValueError
        Raised in the following cases:

        - Binary targets do not contain exactly two distinct labels.
        - Top-label predictions are all correct or all incorrect.
        - A classwise target lacks positive or negative examples.
        - The number of distinct labels differs from the number of multiclass
          score columns.
        - Scores and labels have different numbers of samples.
        - Scores contain nonnumeric values.
        - Scores contain NaN or missing values.
        - Scores contain boolean values.
        - Scores contain infinite values.
        - Scores contain values outside the finite floating-point range.
        - Scores or labels contain complex values.
        - Labels contain NaN or missing values.
        - max_bins is not a nonnegative integer when histogram is None.
        - Reused histogram statistics lack target frequencies.
        - Reused binary histogram classes do not match the sorted labels in y.
        - reuse_stats is True for multiclass scores.
        - Scores map to empty bins in the supplied or constructed histogram.
        - method or multi_class_method is not a supported estimation method.
    TypeError
        `reuse_stats` is not a boolean.

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
    # Validate estimation methods before computing histogram statistics
    if not isinstance(reuse_stats, (bool, np.bool_)):
        raise TypeError("'reuse_stats' must be a boolean.")
    if histogram is None:
        max_bins = _validate_max_bins(max_bins)
    if method not in ("label-bin", "bin"):
        raise ValueError(f"'method' must be 'label-bin' or 'bin'. Got {method!r}.")
    if multi_class_method not in ("top-label", "classwise"):
        raise ValueError(
            "'multi_class_method' must be 'top-label' or 'classwise'. "
            f"Got {multi_class_method!r}."
        )

    # Normalize labels while preserving binary and multiclass score matrices
    y = _as_1d_array(y, "y")
    _check_no_missing_values(y, "y")
    y_scores = _as_score_array(y_scores, "y_scores")
    if y_scores.ndim not in (1, 2) or (y_scores.ndim == 2 and y_scores.shape[1] == 0):
        raise ValueError(
            "'y_scores' must be either a 1-D or 2-D array-like object, "
            f"but its shape is {y_scores.shape}"
        )
    check_consistent_length(y_scores, y)
    _check_no_infinite_values(y_scores, "y_scores")

    # Keep reusable histogram statistics restricted to binary score inputs
    if reuse_stats and y_scores.ndim == 2 and y_scores.shape[1] > 2:
        raise ValueError("'reuse_stats' is not supported for multiclass scores.")

    # Binary ECE uses the positive-class column when two columns are supplied
    if y_scores.ndim == 1 or y_scores.shape[1] <= 2:
        binary_scores = y_scores if y_scores.ndim == 1 else y_scores[:, -1]
        if len(np.unique(y)) != 2:
            raise ValueError(
                "Binary ECE requires exactly two distinct target labels in 'y'."
            )
        return _binary_ece(
            binary_scores,
            y,
            method=method,
            histogram_method=histogram_method,
            max_bins=max_bins,
            histogram=histogram,
            reuse_stats=reuse_stats,
        )
    # 2-D array with > 2 columns: Multi-class ECE
    else:
        le = LabelEncoder().fit(y)
        if len(le.classes_) != y_scores.shape[1]:
            raise ValueError(
                "Number of distinct labels in 'y' must match the number of "
                "columns in multiclass 'y_scores': "
                f"{len(le.classes_)} != {y_scores.shape[1]}."
            )
        if multi_class_method == "top-label":
            prediction_indexes = np.argmax(y_scores, axis=1)
            y_preds = (y == le.classes_[prediction_indexes]).astype(int)
            y_pred_scores = y_scores[np.arange(len(y)), prediction_indexes]
            if len(np.unique(y_preds)) != 2:
                raise ValueError(
                    "Top-label ECE requires both correct and incorrect predictions."
                )
            return _binary_ece(
                y_pred_scores,
                y_preds,
                method=method,
                histogram_method=histogram_method,
                max_bins=max_bins,
                histogram=histogram,
                reuse_stats=reuse_stats,
            )
        else:
            assert multi_class_method == "classwise"
            y_binarized = label_binarize(y, classes=le.classes_)
            class_eces = []
            for k in range(y_scores.shape[1]):
                if len(np.unique(y_binarized[:, k])) != 2:
                    raise ValueError(
                        "Classwise ECE requires both positive and negative examples "
                        f"for class {le.classes_[k]!r}."
                    )
                class_eces.append(
                    _binary_ece(
                        y_scores[:, k],
                        y_binarized[:, k],
                        method=method,
                        histogram_method=histogram_method,
                        max_bins=max_bins,
                        histogram=histogram,
                        reuse_stats=reuse_stats,
                    )
                )
            class_probas = np.unique(y, return_counts=True)[1] / len(y)
            return float(np.dot(class_eces, class_probas))


def _binary_ece(
    y_scores,
    y,
    method: str = "label-bin",
    histogram_method: str = "khiops",
    max_bins: int = 0,
    histogram: Histogram | None = None,
    reuse_stats: bool = False,
):
    """Estimates the ECE for a 2-class problem

    See the function `calibration_error` docstring for more details.
    """
    # Normalize binary scores and labels
    y_scores = _as_1d_array(y_scores, "y_scores")
    y = _as_1d_array(y, "y")
    assert len(np.unique(y)) == 2

    # Validate supplied statistics before any refresh can populate empty bins
    supplied_histogram = histogram is not None
    if supplied_histogram:
        bin_indexes = _validate_nonempty_bin_scores(histogram, y_scores)

    # Build or refresh statistics unless reuse is explicitly requested
    if histogram is None:
        histogram = Histogram.from_data(
            y_scores, y=y, method=histogram_method, max_bins=max_bins
        )
    elif not reuse_stats:
        histogram = Histogram.from_data_and_breakpoints(
            y_scores,
            histogram.breakpoints,
            y=y,
        )

    # Require target statistics for exactly two classes
    if not histogram.target_freqs:
        raise ValueError("'histogram' must have target statistics.")
    if (n_classes := len(histogram.classes)) != 2:
        raise ValueError(
            f"The histogram must have exactly 2 target classes. It has {n_classes}."
        )
    expected_classes = LabelEncoder().fit(y).classes_
    if reuse_stats and not np.array_equal(histogram.classes, expected_classes):
        raise ValueError(
            "Reused binary histogram classes must match the sorted distinct labels "
            f"in 'y'. Got {histogram.classes!r}; expected "
            f"{expected_classes.tolist()!r}."
        )

    # Validate the constructed histogram when no statistics were supplied
    if not supplied_histogram:
        bin_indexes = _validate_nonempty_bin_scores(histogram, y_scores)

    # Estimate the ECE with the histogram
    if method == "label-bin":
        sum_diffs = 0
        for y_score, i in zip(y_scores, bin_indexes, strict=True):
            sum_diffs += math.fabs(
                y_score - histogram.target_freqs[i][1] / histogram.freqs[i]
            )
        return sum_diffs / len(y)
    else:
        assert method == "bin"
        sum_score_by_bin = [0 for _ in histogram.bins]
        for y_score, i in zip(y_scores, bin_indexes, strict=True):
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
        provided and not fitted, a clone is fitted with :paramref:`train_size`
        elements. If already fitted, the estimator is copied for calibration.
    train_size : float or int, default=0.75
        Same parameter as `~sklearn.model_selection.train_test_split`. Only used when
        :paramref:`estimator` is not provided or not fitted.
    random_state : int, default=None
        Same parameter as `~sklearn.model_selection.train_test_split`. Only used when
        :paramref:`estimator` is not provided or not fitted.


    Attributes
    ----------
    fitted_estimator_ :
        The independent fitted classifier used for calibration. It is an instance
        of `KhiopsClassifier`_ when :paramref:`estimator` is not provided.
    histograms_ : list of size 1 or n_classes
        In the binary case: a list of size 1 with the `Histogram` instance for the
        positive class. In the multi class case: A list of size n_classes containing the
        one-vs-rest `Histogram` for each class.
    classes_ : ndarray of shape (n_classes,)
        Class labels in the order of the columns returned by `predict_proba`. For
        binary classification, the second class is positive and its probability is
        returned in column 1.

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

        Raises
        ------
        ValueError
            Raised in the following cases:

            - Calibration labels contain fewer than two classes.
            - Sparse matrices are used outside classifier feature input X.
            - Features, labels, train_size, random_state, or estimator scores
              contain complex values.
            - Labels contain NaN or missing values.
            - Calibration labels do not exactly match the fitted estimator's classes.
            - Multiclass scores use a decision function with
              decision_function_shape="ovo", including inside pipelines or searches.
        TypeError
            `random_state` is a boolean.

        Notes
        -----
        Fitted attributes are updated only after calibration succeeds. A failed
        refit preserves the previous fitted state. The supplied estimator is cloned
        before fitting, or copied if already fitted, so later changes to it do not
        affect this classifier.
        """
        # Validate features and reject unsupported sparse inputs
        _check_classifier_features(X)
        if isinstance(self.random_state, (bool, np.bool_)):
            raise TypeError("'random_state' must not be a boolean.")
        for name, values in (
            ("y", y),
            ("train_size", self.train_size),
            ("random_state", self.random_state),
        ):
            _check_real_values(values, name)
        _check_no_missing_values(y, "y")

        # Isolate the estimator from subsequent changes by its caller
        estimator = self.estimator if self.estimator is not None else KhiopsClassifier()
        try:
            check_is_fitted(estimator)
        except NotFittedError:
            fitted_estimator = clone(estimator)
            x_train, x_calib, y_train, y_calib = train_test_split(
                X, y, train_size=self.train_size, random_state=self.random_state
            )
            fitted_estimator.fit(x_train, y_train)
        else:
            fitted_estimator = deepcopy(estimator)
            x_calib, y_calib = X, y

        # Require calibration examples for every estimator class
        calibration_classes = np.unique(y_calib)
        _check_real_values(fitted_estimator.classes_, "classes")
        if len(calibration_classes) < 2 or not np.array_equal(
            calibration_classes, fitted_estimator.classes_
        ):
            raise ValueError(
                "Calibration labels must contain at least two classes and exactly "
                "match the fitted estimator's classes. "
                f"Got {calibration_classes.tolist()!r}; "
                f"expected {fitted_estimator.classes_.tolist()!r}."
            )

        # Estimate the uncalibrated scores
        scores, _, response_method = _get_response_values(
            fitted_estimator,
            x_calib,
            response_method=["predict_proba", "decision_function"],
            return_response_method_used=True,
        )
        _check_real_values(scores, "scores")
        if (
            len(fitted_estimator.classes_) > 2
            and response_method == "decision_function"
            and _get_decision_function_shape(fitted_estimator) == "ovo"
        ):
            raise ValueError(
                "Multiclass calibration does not support "
                'decision_function_shape="ovo". '
                'Use decision_function_shape="ovr" or an estimator with predict_proba.'
            )

        # Binary classification
        if (n_classes := len(fitted_estimator.classes_)) == 2:
            histograms = [Histogram.from_data(scores, y_calib)]
        # Multiclass classification: One-vs-Rest
        else:
            y_binarized = label_binarize(y_calib, classes=fitted_estimator.classes_)
            histograms = [
                Histogram.from_data(scores[:, k], y_binarized[:, k])
                for k in range(n_classes)
            ]

        # Publish matching estimator and calibration state only after success
        self.fitted_estimator_ = fitted_estimator
        self.histograms_ = histograms
        return self

    @property
    def classes_(self):
        """Class labels in probability column order."""
        check_is_fitted(self, ["fitted_estimator_", "histograms_"])
        return self.fitted_estimator_.classes_

    def predict(self, X):  # noqa: N803
        """Predicts class labels using calibrated probabilities

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

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

        Raises
        ------
        ValueError
            Raised in the following cases:

            - Multiclass calibrated probabilities sum to zero and the base
              estimator cannot provide valid probabilities for the fallback.
            - Sparse matrices are used outside classifier feature input X.
            - Features or estimator scores contain complex values.
            - Multiclass scores use a decision function with
              decision_function_shape="ovo", including inside pipelines or searches.

        Warns
        -----
        UserWarning
            If multiclass calibrated probabilities sum to zero. The uncalibrated
            probabilities are returned for those samples instead.
        """
        # Check that is fitted
        check_is_fitted(self, ["fitted_estimator_", "histograms_"])
        _check_classifier_features(X)

        # Estimate the uncalibrated scores
        scores, _, response_method = _get_response_values(
            self.fitted_estimator_,
            X,
            response_method=["predict_proba", "decision_function"],
            return_response_method_used=True,
        )
        _check_real_values(scores, "scores")
        if (
            len(self.fitted_estimator_.classes_) > 2
            and response_method == "decision_function"
            and _get_decision_function_shape(self.fitted_estimator_) == "ovo"
        ):
            raise ValueError(
                "Multiclass calibration does not support "
                'decision_function_shape="ovo". '
                'Use decision_function_shape="ovr" or an estimator with predict_proba.'
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

            # Detect zero-sum calibration
            normalization_factors = calibrated_probas.sum(axis=1)
            zero_sum_rows = np.flatnonzero(normalization_factors == 0)
            if zero_sum_rows.size:
                # Decision scores cannot serve as fallback probabilities
                if not hasattr(self.fitted_estimator_, "predict_proba"):
                    raise ValueError(
                        "Cannot fall back to uncalibrated probabilities for samples "
                        f"{zero_sum_rows.tolist()}: the fitted estimator does not "
                        "provide predict_proba."
                    )

                # Require valid probabilities for the fallback
                fallback_probas = scores[zero_sum_rows]
                if (
                    not np.all(np.isfinite(fallback_probas))
                    or np.any(fallback_probas < 0)
                    or not np.allclose(fallback_probas.sum(axis=1), 1)
                ):
                    raise ValueError(
                        "Cannot fall back to uncalibrated probabilities for samples "
                        f"{zero_sum_rows.tolist()}: predict_proba returned invalid "
                        "probabilities."
                    )

                # Fall back only for affected samples
                warnings.warn(
                    "Calibrated probabilities sum to zero for samples "
                    f"{zero_sum_rows.tolist()}; using uncalibrated probabilities "
                    "for those samples.",
                    UserWarning,
                    stacklevel=2,
                )
                calibrated_probas[zero_sum_rows] = fallback_probas

            # Normalize only the calibrated samples
            nonzero_sum_rows = normalization_factors != 0
            calibrated_probas[nonzero_sum_rows] /= normalization_factors[
                nonzero_sum_rows, np.newaxis
            ]

        return calibrated_probas


def calibrate_binary(y_scores, histogram: Histogram, only_positive: bool = False):
    """Calibrates a binary score with an histogram

    This function may be used to calibrate binary classification scores without the use
    of the `KhalibClassifier` estimator.

    Parameters
    ----------
    y_scores : array-like of shape (n_samples,) or (n_samples, 1)
        Scores for the second class in ``histogram.classes``. NaN, missing,
        infinite, complex, and values outside the finite floating-point range are
        not accepted. Finite scores outside the histogram's breakpoint range map
        to the nearest end bin.
    histogram : `Histogram`
        A histogram with target statistics (ie. :paramref:`Histogram.target_freqs` must
        be non-empty). Its class order defines the output probability-column order;
        the second class is the positive class.
    only_positive : bool, default=False
        If ``True`` it returns only 1-D array with the probabilities of the positive
        class. Otherwise it returns a 2-D array with the probabilities for both classes.

    Raises
    ------
    ValueError
        Raised in the following cases:

        - The histogram lacks target statistics.
        - The histogram does not have exactly two classes.
        - Any score is NaN, missing, infinite, or complex.
        - Any score is nonnumeric.
        - Any score is boolean.
        - Any score is outside the finite floating-point range.
        - Any score maps to an empty bin.
    TypeError
        ``only_positive`` is not a boolean.
    """
    # Check the inputs
    if not isinstance(only_positive, (bool, np.bool_)):
        raise TypeError("'only_positive' must be a boolean.")
    y_scores = _as_1d_array(_as_score_array(y_scores, "y_scores"), "y_scores")
    if not histogram.classes or not histogram.target_freqs:
        raise ValueError("'histogram' must have target statistics.")
    if (n_classes := len(histogram.classes)) != 2:
        raise ValueError(f"'histogram' must have only 2 classes (it has {n_classes})")

    # Reject scores without calibration observations in their bins
    y_scores_bin_indexes = histogram.vfind(y_scores)
    empty_bin_samples = np.flatnonzero(
        np.asarray(histogram.freqs)[y_scores_bin_indexes] == 0
    )
    if empty_bin_samples.size:
        raise ValueError(
            "Cannot calibrate scores in empty histogram bins: "
            f"sample indices {empty_bin_samples.tolist()}, "
            f"bin indices {y_scores_bin_indexes[empty_bin_samples].tolist()}."
        )

    # Map each score to its bin target probabilities
    calibrated_probas = np.asarray(histogram.target_probas)[y_scores_bin_indexes]
    if only_positive:
        return calibrated_probas[:, 1]
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
        Positive-class scores/probabilities. For a two-column array, column 1
        is used; a single-column array is flattened.
    y : array-like of shape (n_samples,) or (n_samples, 1)
        Target values containing exactly two distinct class labels.
    dirac_threshold : float, default=1.0e-06
        If a bin in the scores' unsupervised histogram is lower than this
        'dirac_threshold' then it is considered a dirac mass.
    log_plot_threshold : float, default=3.0
        *Density plot only:* If the log-difference between the maximal and minimal
        positive density values is larger than 'log_plot_threshold' then the density
        plot uses a log scale in the y-axis.
    min_density_bar_width : float, default=2.5e-03
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

    Raises
    ------
    ValueError
        Raised in the following cases:

        - Scores and labels have different numbers of samples.
        - Target labels do not contain exactly two distinct classes.
        - Scores contain nonnumeric values.
        - Scores contain NaN or missing values.
        - Scores contain boolean values.
        - Scores contain infinite values.
        - Scores contain values outside the finite floating-point range.
        - Scores are outside the range [0, 1].
        - Labels contain NaN or missing values.
        - Plotting thresholds are nonfinite or nonpositive.
        - Scores, labels, or plotting thresholds contain complex values.
    TypeError
        A plotting threshold is not a real number.
    """
    # Check the input parameters
    for name, value in (
        ("dirac_threshold", dirac_threshold),
        ("log_plot_threshold", log_plot_threshold),
        ("min_density_bar_width", min_density_bar_width),
    ):
        _check_real_values(value, name)
        try:
            is_valid = (
                not isinstance(value, (bool, np.bool_))
                and math.isfinite(value)
                and value > 0.0
            )
        except TypeError as error:
            raise TypeError(f"'{name}' must be a real number.") from error
        except OverflowError:
            is_valid = False
        if not is_valid:
            raise ValueError(f"'{name}' must be finite and positive.")
    y_scores = _as_score_array(y_scores, "y_scores")
    if y_scores.ndim not in (1, 2) or (
        y_scores.ndim == 2 and y_scores.shape[1] not in (1, 2)
    ):
        raise ValueError(
            "'y_scores' must be 1-D or 2-D with one or two columns, "
            f"but its shape is {y_scores.shape}"
        )
    y = _as_1d_array(y, "y")
    _check_no_missing_values(y, "y")
    check_consistent_length(y_scores, y)
    _check_no_infinite_values(y_scores, "y_scores")
    if np.any(y_scores < 0) or np.any(y_scores > 1):
        raise ValueError("'y_scores' must be in the range [0, 1].")
    if len(np.unique(y)) != 2:
        raise ValueError(
            "Reliability diagrams require exactly two distinct target labels in 'y'."
        )
    # Normalize positive-class scores for all histogram computations
    if y_scores.ndim == 2:
        y_scores = y_scores[:, -1]

    # Create the base plot
    fig, axs = plt.subplot_mosaic(
        [["reliability_diagram"], ["score_distribution"]],
        figsize=(6, 6),
        height_ratios=(4, 1),
        sharex=True,
    )
    fig.subplots_adjust(hspace=0)
    fig.suptitle("Reliability Diagram")

    # Build a unsupervised histogram
    try:
        uhist_y_scores = Histogram.from_data(y_scores, use_finest=True)
    except Exception:
        plt.close(fig)
        raise

    # Compute the dirac mass indexes
    dirac_indexes = compute_dirac_indexes(uhist_y_scores, dirac_threshold)

    # Compute the supervised score histogram
    try:
        hist_y_scores = Histogram.from_data(y_scores, y)
    except Exception:
        plt.close(fig)
        raise

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
        try:
            uhist_y_scores = Histogram.from_data(y_scores)
        except Exception:
            plt.close(fig)
            raise

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
        density_bar_log_range = 0
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


def compute_dirac_indexes(uhist, dirac_threshold):
    """Computes the dirac mass indexes of a histogram

    We declare a dirac mass bin if:

    - it contains observations
    - it is surrounded by empty bins or if it is the only one
    - its length is less than ``dirac_threshold``

    Raises
    ------
    ValueError
        If dirac_threshold is complex.
    """
    _check_real_values(dirac_threshold, "dirac_threshold")

    # Early return for just one bin
    if len(uhist.freqs) == 1:
        if uhist.freqs[0] > 0 and uhist.bins[0][1] - uhist.bins[0][0] < dirac_threshold:
            return [True]
        else:
            return [False]

    # More than one bin
    dirac_indexes = []
    if (
        len(uhist.freqs) > 1
        and uhist.freqs[0] > 0
        and uhist.freqs[1] == 0
        and (uhist.bins[0][1] - uhist.bins[0][0] < dirac_threshold)
    ):
        dirac_indexes.append(True)
    else:
        dirac_indexes.append(False)
    for i in range(1, uhist.n_bins - 1):
        cur_left, cur_right = uhist.bins[i]
        if (
            uhist.freqs[i] > 0
            and uhist.freqs[i - 1] == 0
            and uhist.freqs[i + 1] == 0
            and (cur_right - cur_left < dirac_threshold)
        ):
            dirac_indexes.append(True)
        else:
            dirac_indexes.append(False)
    if (
        len(uhist.freqs) > 1
        and uhist.freqs[-1] > 0
        and uhist.freqs[-2] == 0
        and (uhist.bins[-1][1] - uhist.bins[-1][0] < dirac_threshold)
    ):
        dirac_indexes.append(True)
    else:
        dirac_indexes.append(False)

    return dirac_indexes
