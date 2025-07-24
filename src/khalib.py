import json
import math
import os
import tempfile
import subprocess
import warnings
from bisect import bisect_left
import shutil

import numpy as np
import pandas as pd
from khiops import core as kh
from khiops.core.internals.runner import KhiopsLocalRunner
from khiops.sklearn import KhiopsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.preprocessing import label_binarize, normalize


class KhiopsDataBinner:
    def __init__(self):
        default_max_cores = os.environ.get("KHIOPS_PROC_NUMBER")
        os.environ["KHIOPS_PROC_NUMBER"] = "1"
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message=".*Too few cores:.*")
            self.runner = KhiopsLocalRunner()
        if default_max_cores is None:
            del os.environ["KHIOPS_PROC_NUMBER"]
        else:
            os.environ["KHIOPS_PROC_NUMBER"] = default_max_cores

    def compute_bins(
        self, y_scores, y=None, method="MODL", max_parts=0, return_freqs=False, cleanup=True
    ):
        results = self._compute_bins_with_khiops(
            y_scores, y=y, method=method, max_parts=max_parts
        )
        bins, freqs = self._extract_bins_and_freq(results)

        if return_freqs:
            return bins, freqs
        else:
            return bins

    def _compute_bins_with_khiops(
        self, y_scores, y=None, method="MODL", max_parts=0, cleanup=True
    ):
        assert y is None or (len(y.shape) == 1 or y.shape[1] == 1)
        assert len(y_scores.shape) == 1 or y_scores.shape[1] == 1

        # Create Khiops dictionary
        kdom = kh.DictionaryDomain()
        kdic = kh.Dictionary()
        kdic.name = "scores"
        kdom.add_dictionary(kdic)
        var = kh.Variable()
        var.name = "y_score"
        var.type = "Numerical"
        kdic.add_variable(var)
        if y is not None:
            var = kh.Variable()
            var.name = "y"
            var.type = "Categorical"
            kdic.add_variable(var)

        # Create output dataframe
        df_spec = {
            "y_score": y_scores if len(y_scores.shape) == 1 else y_scores.flatten()
        }
        if y is not None:
            df_spec["y"] = y if len(y.shape) == 1 else y.flatten()
        output_df = pd.DataFrame(df_spec)

        # Export the output dataframe to a temporary file
        tmp_dir = tempfile.mkdtemp()
        output_table_path = f"{tmp_dir}/y_scores.txt"
        output_df.to_csv(output_table_path, sep="\t", index=False)

        # Execute Khiops recover the report
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message="(?s:.*No informative input variable available.*)",
            )
            default_runner = kh.get_runner()
            kh.set_runner(self.runner)
            kh.train_predictor(
                kdom,
                "scores",
                output_table_path,
                "y" if y is not None else "",
                f"{tmp_dir}/report.khj",
                sample_percentage=100,
                field_separator="\t",
                header_line=True,
                max_trees=0,
                discretization_method=method,
                max_parts=max_parts,
                do_data_preparation_only=True,
            )
            kh.set_runner(default_runner)
        results = kh.read_analysis_results_file(f"{tmp_dir}/report.khj")

        # Cleanup if necessary
        if cleanup:
            shutil.rmtree(tmp_dir)
        else:
            print(f"Khiops execution files available at {tmp_dir}")

        return results


    def _extract_bins_and_freq(self, results):
        # Recover the bins from the variable statistics objects
        score_stats = results.preparation_report.variables_statistics[0]
        bins = []
        freqs = []
        if score_stats.data_grid is not None:
            if score_stats.data_grid.is_supervised:
                for part in score_stats.data_grid.dimensions[0].partition:
                    bins.append((part.lower_bound, part.upper_bound))
                    for target_freqs in score_stats.data_grid.part_target_frequencies:
                        freqs.append(np.sum(target_freqs))
            else:
                for part in score_stats.data_grid.dimensions[0].partition:
                    bins.append((part.lower_bound, part.upper_bound))
                freqs.extend(score_stats.data_grid.frequencies)
        # If there are no variable grids for the variable
        else:
            bins.append((score_stats.min, score_stats.max + 1e-8))
            freqs.append(results.preparation_report.instance_number)
        return bins, freqs


khiops_binner = KhiopsDataBinner()


def robust_estimate_binary_ece(y, y_scores, y_pos, return_estimations=False):
    global khiops_binner
    bins = khiops_binner.compute_bins(
        y_scores, y, method="EqualFrequency", max_parts=math.ceil(math.sqrt(len(y)))
    )
    return plugin_binary_ece(
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

    zipped_y = sorted(list(zip(y, y_scores)), key=lambda k: k[1])

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
        df = pd.DataFrame(
            {"freq": n_samples_by_bin, "freq_pos": accuracy_by_bin}, index=bins
        )
    for i in range(0, len(bins)):
        if n_samples_by_bin[i] > 0:
            average_score_by_bin[i] /= n_samples_by_bin[i]
            accuracy_by_bin[i] /= n_samples_by_bin[i]

    if return_estimations:
        df["ave_score"] = average_score_by_bin
        df["acc"] = accuracy_by_bin

    ece = 0
    for i in range(0, len(bins)):
        ece += (
            n_samples_by_bin[i]
            / len(y)
            * math.fabs(average_score_by_bin[i] - accuracy_by_bin[i])
        )
    if debias:
        # Estimate the normal standard deviations
        std_devs = accuracy_by_bin * (1 - accuracy_by_bin)
        for i in range(len(std_devs)):
            if std_devs[i] > 0:
                std_devs[i] /= n_samples_by_bin[i]
                std_devs[i] = math.sqrt(std_devs[i])

        if return_estimations:
            bin_biases = []
            ave_bins_abs_diffs = []
        ece = 2 * ece
        debias_samples = 100
        for i in range(0, len(bins)):
            # binomial_samples = np.random.binomial(
            # size=debias_samples, n=n_samples_by_bin[i], p=accuracy_by_bin[i]
            # )
            samples = np.random.normal(
                size=debias_samples, loc=accuracy_by_bin[i], scale=std_devs[i]
            )
            bin_bias_abs_diffs = np.abs(samples - average_score_by_bin[i])
            ave_bin_abs_diff = np.average(bin_bias_abs_diffs)
            bin_bias = n_samples_by_bin[i] / len(y) * ave_bin_abs_diff
            ece -= bin_bias
            if return_estimations:
                bin_biases.append(bin_bias)
                ave_bins_abs_diffs.append(ave_bin_abs_diff)
        if ece <= 0.0:
            ece = 0.0

        if return_estimations:
            df["bias"] = bin_biases

    if return_estimations:
        return ece, df
    else:
        return ece


def binary_ece_lb(y, y_scores, y_pos, bins, return_estimations=False):
    accuracy_by_bin = np.zeros(len(bins))
    n_samples_by_bin = np.zeros(len(bins))
    y_score_bin_indexes = []

    for y_value, y_score in zip(y, y_scores):
        i_y_score_bin = find_bin(y_score, bins)
        y_score_bin_indexes.append(i_y_score_bin)
        n_samples_by_bin[i_y_score_bin] += 1
        if y_value == y_pos:
            accuracy_by_bin[i_y_score_bin] += 1

    if return_estimations:
        df = pd.DataFrame(
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
        df["acc"] = accuracy_by_bin
        df["diff_ave"] = diff_ave

    ece = 0
    for i in range(0, len(bins)):
        ece += n_samples_by_bin[i] / len(y) * diff_ave[i]

    if return_estimations:
        return ece, df
    else:
        return ece


def binary_ece_lb_alt(y, y_scores, y_pos, bins, return_estimations=False):
    pass


# Set a single core runner for the calibration
default_core_number = os.environ.get("KHIOPS_PROC_NUMBER")
os.environ["KHIOPS_PROC_NUMBER"] = "1"
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message=".*Too few cores:.*")
    khcalib_runner = KhiopsLocalRunner()
if default_core_number is None:
    del os.environ["KHIOPS_PROC_NUMBER"]
else:
    os.environ["KHIOPS_PROC_NUMBER"] = default_core_number


def run_with_single_core(func):
    def _run_with_single_core(self, *args, **kwargs):
        default_runner = kh.get_runner()
        global khcalib_runner
        kh.set_runner(khcalib_runner)
        result = func(self, *args, **kwargs)
        kh.set_runner(default_runner)
        return result

    return _run_with_single_core


class KhiopsCalibrator(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    def __init__(self, clf, normalize=False):
        self.clf = clf
        self.normalize = normalize

    @run_with_single_core
    def fit(self, X, y):
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

    @run_with_single_core
    def predict_proba(self, X):
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
