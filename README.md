<!-- start-summary -->
# khalib
`khalib` is a classifier probability calibration package powered by the [Khiops][khiops-org] AutoML
suite.


## Features
- `KhalibClassifier`: A scikit-learn estimator to calibrate classifiers with a similar interface
  fashion as [CalibratedClassifierCV][sk-calclf].
- `calibration_error` : A function to estimate the Estimated Calibration Error (ECE).
- `build_reliability_diagram` : A function that builds a reliability diagram.

These features are based on Khiops's non-parametric supervised histograms, so there is no need to
specify the number and width of the bins, as they are automatically estimated from data.

<!-- end-summary -->

See the [documentation][khalib-docs] for more information.

<!-- start-install -->
## Installation

*Note: We'll improve this installation procedure soon!*

- Make sure you have installed [Khiops 11 Beta][khiops11-setup]
- Execute

```bash
pip install https://github.com/KhiopsLab/khalib/archive/refs/tags/0.1.zip
```

<!-- end-install -->


## Documentation

See https://khiopslab.github.io/khalib/


[khiops-org]: https://khiops.org
[khiops11-setup]: https://khiops.org/11.0.0-b.0/setup/
[sk-calclf]: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
[khalib-docs]: https://khiopslab.github.io/khalib

