<!-- start-summary -->
# khalib: Classifier Calibration made Simple
`khalib` is a classifier calibration package powered by the [Khiops][khiops-org] AutoML suite. It
helps you bring your model probabilities back to reality: for better decision thresholds, risk
estimation, and interpretability.


## Features
- `KhalibClassifier`: A scikit-learn estimator to build calibrated probabilities in a similar fashion
  as [CalibratedClassifierCV][sk-calclf]
- Fully non-parametric classifier calibration and ECE : the user does not need to input any model
  parameters.

## How does it work

`khalib` is a histogram-based calibration method. These construct

The differenciating factor from other methods, is
that it doesn't require the histogram as a parameter nor it resorts to

It uses the [Khiops][khiops]' supervised discretization
algorithm (MODL) to find a histogram for the input scores such that in each bin

<!-- end-summary -->

[khiops-org]: https://khiops.org
[sk-calclf]: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
