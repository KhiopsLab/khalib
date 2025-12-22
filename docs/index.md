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
is that uses Khiops to construct the histogram in which {math}`P(Y = 1 | S)` is estimated. These
histograms have the following properties:
- They balance class purity, model complexity and data fitness.
- They are non-parametric: The optimal histogram is searched without constraint in number of bins or
  bin width. This implies that the user doesn't need to set a number of bins nor their widths.


<!--[khiops-org]: https://khiops.org-->


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
Github <https://github.com/KhiopsLab/khalib>
```

