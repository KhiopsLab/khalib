.. _input-requirements:

Input requirements
==================

Each API's parameter descriptions specify its accepted shapes and class ordering.
The following rules apply across APIs.

Scores
------

Scores must be real numeric values representable as finite floats. Booleans,
numeric strings, complex values (even with zero imaginary part), missing values
(including NaN and pandas NA), infinities, and sparse matrices are not supported.
Numeric object arrays are accepted when their elements satisfy these rules.
Integer scores retain exact comparisons during histogram bin lookup.

Histogram construction requires at least one score. Binary calibration also
accepts empty score arrays and returns empty probability arrays.
Only reliability diagrams require scores in the range [0, 1]; elsewhere finite
scores outside histogram boundaries map to the nearest end bin.

Labels
------

Labels must be real numeric, boolean, or string class labels, without missing or
complex values. Sparse label matrices are not supported. When scores and labels
are supplied together, they must have matching sample counts.

Binary ECE and reliability diagrams require exactly two distinct labels.
Multiclass ECE requires one distinct label per score column. Classifier
calibration labels must exactly match the fitted estimator's classes.

Histogram fields
----------------

Breakpoints must contain at least two finite, strictly increasing real values.
Python integers may exceed the floating-point range; NumPy integer endpoints are
stored as Python integers.

Bin and target frequencies must be finite and nonnegative. Bin frequencies must
have a positive, finite total. Arbitrary-size Python integer counts are supported.
Target frequencies must match the bin and class dimensions and sum to their bin
frequency. Classes must have no missing or complex values and must be unique when
target statistics are supplied.

Classifier features and options
-------------------------------

Unlike score inputs, classifier feature matrices may be sparse. Features must be
real; additional requirements are enforced by the underlying estimator.

Boolean options accept Python and NumPy booleans. Bin limits must be nonnegative
Python or NumPy integers, not booleans. Plotting thresholds must be finite,
positive real numbers, not booleans. Classifier splitting options follow
scikit-learn's conventions; complex values and boolean random seeds are rejected.
