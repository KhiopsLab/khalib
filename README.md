<!-- start-summary -->
# khalib
`khalib` is a binary and multiclass classifier probability calibration package powered by the
[Khiops][khiops-org] AutoML suite.


## Features
- `KhalibClassifier`: A scikit-learn estimator to calibrate classifiers with a similar interface
  as [CalibratedClassifierCV][sk-calclf].
- `calibration_error` : A function to estimate the Estimated Calibration Error (ECE).
- `build_reliability_diagram` : A function to build a reliability diagram.
- `Histogram` and `calibrate_binary`: Lower-level APIs for histogram construction and binary
  score calibration.

These features are based on Khiops's non-parametric supervised histograms, so there is no need to
specify the number and width of the bins, as they are automatically estimated from data.

[khiops-org]: https://khiops.org
[sk-calclf]: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html

<!-- end-summary -->

<!-- start-install -->
## Installation

Requires Python 3.11 through 3.14. Khiops is installed automatically; no separate installation is needed.

```sh
pip install khalib
```

<!-- end-install -->

## Example

Train a classifier, calibrate it on held-out data, and evaluate on a separate test set:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from khalib import KhalibClassifier, calibration_error

X, y = load_breast_cancer(return_X_y=True)
X_train, X_heldout, y_train, y_heldout = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_calib, X_test, y_calib, y_test = train_test_split(
    X_heldout, y_heldout, test_size=0.5, stratify=y_heldout, random_state=42
)

estimator = make_pipeline(StandardScaler(), LogisticRegression())
estimator.fit(X_train, y_train)

calibrated = KhalibClassifier(estimator=estimator).fit(X_calib, y_calib)
raw_probabilities = estimator.predict_proba(X_test)
probabilities = calibrated.predict_proba(X_test)
print("Raw ECE:       ", calibration_error(raw_probabilities, y_test))
print("Calibrated ECE:", calibration_error(probabilities, y_test))
```

Binary probability column 1 corresponds to `calibrated.classes_[1]`.

## Documentation

See the [documentation][khalib-docs] for tutorials and API details, and the
[changelog][khalib-changelog] for release notes.


[khalib-docs]: https://khiopslab.github.io/khalib
[khalib-changelog]: https://github.com/KhiopsLab/khalib/blob/main/CHANGELOG.md
