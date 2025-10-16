```{include} ../README.md
:start-after: <!-- start-summary -->
:end-before:  <!-- end-summary -->
```

```{include} ../README.md
:start-after: <!-- start-install -->
:end-before:  <!-- end-install -->
```

## How does it work

`khalib` proposes histogram-based calibration and its error estimation. Its differentiating factor
is that uses [Khiops][khiops-org] to construct the histogram in which {math}`P(Y = 1 | S)` is
estimated. These histograms have the following properties:
- They balance class purity, model complexity and data fitness.
- They are non-parametric: The optimal histogram is searched without constraint in number of bins or
  bin width. This implies that the user doesn't need to set a number of bins nor their widths.


[khiops-org]: https://khiops.org
[khiops11-setup]: https://khiops.org/11.0.0-b.0/setup/
[sk-calclf]: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
[khalib-docs]: https://khiopsml.github.io/khalib


See the [Quickstart](quickstart) and [API reference](api) to learn how to use the library.


```{toctree}
:hidden:

Home <self>
Quickstart <quickstart>
API Reference <api>
```

```{toctree}
:caption: See Also
:hidden:

Khiops <https://www.khiops.org>
```

