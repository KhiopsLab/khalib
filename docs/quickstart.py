# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
#  # Quickstart
#
#
#  {download}`⬇️  Download this Notebook <./quickstart.ipynb>`
#
#  ## KhalibClassifier scikit-learn Estimator
#
#  We begin by creating a synthetic dataset with 35,000 samples. We reserve 2,500 for
#  training and 2,500 for calibration, then use the remaining 30,000 as a held-out test
#  set for measuring calibration error.

# %%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=35_000, n_features=20, n_informative=2, n_redundant=10, random_state=42
)
X_train, X_not_train, y_train, y_not_train = train_test_split(
    X, y, train_size=2500, random_state=42
)
X_calib, X_test, y_calib, y_test = train_test_split(
    X_not_train, y_not_train, train_size=2500, random_state=42
)

# %% [markdown]
# Next, we fit a `GaussianNB` classifier. Gaussian Naive Bayes assumes conditionally
# independent features with Gaussian class-conditional distributions. Our data include
# redundant features, so they violate the conditional-independence assumption. We
# measure the model's calibration error with `calibration_error`, which computes
# cross-validated V-ECE by default:

# %%
from sklearn.naive_bayes import GaussianNB

import khalib

# We fit the model and compute the positive-class scores on the test set:
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_scores_test = gnb.predict_proba(X_test)[:, 1]

# We compute and display the raw model's calibration error:
ece_test = khalib.calibration_error(y_scores_test, y_test)
print("RAW GNB ECE:", ece_test)


# %% [markdown]
# Before calibrating the model, we inspect its reliability diagram:

# %% [markdown]
# We build the diagram with `build_reliability_diagram`, which also plots the score
# distribution:

# %%
# %config InlineBackend.figure_formats = ['svg']
_ = khalib.build_reliability_diagram(y_scores_test, y_test)

# %% [markdown]
# Now we wrap the fitted `GaussianNB` model in `KhalibClassifier` and fit the wrapper
# on the calibration split. This learns the calibration mapping without retraining the
# classifier on calibration data.

# %%
# We fit the wrapper and compute calibrated scores for the test set:
calib_gnb = khalib.KhalibClassifier(gnb)
calib_gnb.fit(X_calib, y_calib)
y_calib_scores_test = calib_gnb.predict_proba(X_test)[:, 1]

# We compute and display the calibrated model's error and its relative reduction:
calib_ece_test = khalib.calibration_error(y_calib_scores_test, y_test)
print("CALIBRATED GNB ECE:", calib_ece_test)
print("Reduction:", (ece_test - calib_ece_test) / ece_test)

# %% [markdown]
# In this run, we see the calibration error fall by about 90%. We now plot the
# calibrated scores. The reliability diagram detects point-mass score distributions
# and displays those masses as Dirac deltas:

# %%
_ = khalib.build_reliability_diagram(y_calib_scores_test, y_test)

# %% [markdown]
# ## calibrate_binary function + Histogram class
#
# We also apply the same calibration directly with `calibrate_binary` and a
# `Histogram` object. This lets us build and inspect the histogram ourselves.

# %%
# First, we compute scores on the calibration split and build a supervised histogram
# from those scores and labels:
y_scores_calib = gnb.predict_proba(X_calib)[:, 1]
hist = khalib.Histogram.from_data(y_scores_calib, y=y_calib)

# Then, we apply the histogram to the test scores:
calib_hist_y_test_scores = khalib.calibrate_binary(
    y_scores_test, hist, only_positive=True
)

# Finally, we compute the error and plot the resulting reliability diagram:
calib_hist_ece_test = khalib.calibration_error(calib_hist_y_test_scores, y=y_test)
print("CALIBRATED HISTOGRAM ECE:", calib_hist_ece_test)
print("Reduction     :", (ece_test - calib_ece_test) / ece_test)
_ = khalib.build_reliability_diagram(calib_hist_y_test_scores, y_test)

# %%

# %%
