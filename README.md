<!-- start-summary -->
# khalib
`khalib` is a classifier probability calibration package powered by the [Khiops][khiops-org] AutoML
suite.


## Features
- `KhalibClassifier`: A scikit-learn estimator to calibrate classifiers with a similar interface
  as [CalibratedClassifierCV][sk-calclf].
- `calibration_error` : A function to estimate the Estimated Calibration Error (ECE).
- `build_reliability_diagram` : A function to build a reliability diagram.

These features are based on Khiops's non-parametric supervised histograms, so there is no need to
specify the number and width of the bins, as they are automatically estimated from data.

[khiops-org]: https://khiops.org
[sk-calclf]: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html

<!-- end-summary -->

See the [documentation][khalib-docs] for more information.

<!-- start-install -->
## Installation

- [Install Khiops][khiops-setup]
- Install the `khalib` python library:

```sh
pip install khalib
```
[khiops-setup]: https://khiops.org/setup/

<!-- end-install -->


## Documentation

See https://khiopslab.github.io/khalib/


[khalib-docs]: https://khiopslab.github.io/khalib
