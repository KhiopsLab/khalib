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
#  ## KhalibClassifier Scikit-Learn Estimator
#
#  We first create our train, calibration and test datasets. We use 45k as test to
#  ensure good error estimations. The rest is divided evenly into train and calibration.

# %%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=50_000, n_features=20, n_informative=2, n_redundant=10, random_state=42
)
X_train, X_not_train, y_train, y_not_train = train_test_split(
    X, y, train_size=2500, random_state=42
)
X_calib, X_test, y_calib, y_test = train_test_split(
    X_not_train, y_not_train, train_size=2500, random_state=42
)

# %% [markdown]
# We now train a `GaussianNB` classifier. This kind of model is usually uncalibrated
# because data never fullfill its hypotheses. We also estimate its expected calibration
# error (ECE):

# %%
from sklearn.naive_bayes import GaussianNB

import khalib

# Compute the positive scores with a Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_scores_test = gnb.predict_proba(X_test)[:, 1]

# Compute and display the ECE
ece_test = khalib.calibration_error(y_scores_test, y_test)
print("RAW GNB ECE:", ece_test)


# %% [markdown]
# To calibrate our `GaussianNB` we create an instance of `KhalibClassifier` with it:

# %% [markdown]
# We can also plot the reliability diagram using the `build_reliability_diagram`
# function:

# %%
# %config InlineBackend.figure_formats = ['svg']
_ = khalib.build_reliability_diagram(y_scores_test, y_test)

# %% [markdown]
# We now calibrate the model with a `KhalibClassifier` object. It uses the uncalibrated
# model as parameter. We then `fit` it on the `calib` split.

# %%
# Train the calibrated classifier and obtain the calibrated scores
calib_gnb = khalib.KhalibClassifier(gnb)
calib_gnb.fit(X_calib, y_calib)
y_calib_scores_test = calib_gnb.predict_proba(X_test)[:, 1]

# Compute the ECE
calib_ece_test = khalib.calibration_error(y_calib_scores_test, y_test)
print("CALIB ECE:", calib_ece_test)
print("Reduction:", (ece_test - calib_ece_test) / ece_test)

# %% [markdown]
# We observe that `khalib` reduced the ECE by ~90%. We now plot the reliability diagram
# for the calibrated scores. The `reliability_diagram` uses a heuristic to detect when
# the scores are distributed as Dirac deltas and changes the visualization accordingly:

# %%
_ = khalib.build_reliability_diagram(y_calib_scores_test, y_test)

# %% [markdown]
# ## calibrate_binary function + Histogram class
#
# We can achieve the same result "manually" by using the function `calibrate_binary`
# which calibrates the scores with a `Histogram` object.

# %%
# Obtain the scores on the calib split and build a supervised histogram with it
y_scores_calib = gnb.predict_proba(X_calib)[:, 1]
hist = khalib.Histogram.from_data(y_scores_calib, y=y_calib)

# Calibrate the scores of the test split
calib_hist_y_test_scores = khalib.calibrate_binary(
    y_scores_test, hist, only_positive=True
)

# Print the error and plot the reliability diagram
calib_hist_ece_test = khalib.calibration_error(calib_hist_y_test_scores, y=y_test)
print("CALIB HIST ECE:", calib_hist_ece_test)
print("Reduction     :", (ece_test - calib_ece_test) / ece_test)
_ = khalib.build_reliability_diagram(calib_hist_y_test_scores, y_test)
