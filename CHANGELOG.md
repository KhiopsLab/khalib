# Changelog

## [0.7] - 2026-11-06

### Added

- `KhalibClassifier.predict` and `classes_` for scikit-learn integration.
- `calibration_error(reuse_stats=True)` to explicitly reuse histogram statistics.

### Changed

- Require Python >= 3.11 and < 3.15.
- Validate inputs consistently, including rejecting calibration and ECE scores in empty histogram
  bins.
- Copy supplied classifiers instead of modifying them; preserve fitted state when refitting fails.
- Recompute supplied ECE histogram statistics from the current data by default.

### Fixed

- Histogram bin boundaries, large-integer precision, constant scores, and Khiops target ordering.
- Multiclass ECE tie-breaking and zero-sum calibrated probability handling.
- Reliability diagrams with binary score matrices or empty bins in Dirac detection.
- Restore Khiops runner settings reliably after failures and concurrent use.
- Prevent Open MPI shared-memory leaks during single-core Khiops execution on Linux.

## [0.6] - 2026-07-03

### Changed
- Simplify installation by using the latest Khiops version: No other installation is necessary.

## [0.5] - 2026-04-30

### Fixed

- `compute_dirac_indexes`: Incorrect result for single-bin histograms.

## [0.4] - 2026-01-05

### Fixed

- `build_reliability_diagram`: Error when scores are distributed uniformly.

## [0.3] - 2025-12-26

### Fixed
- `ModuleNotFound` packaging error.

## [0.2] - 2025-12-26

### Changed
- Update Khiops dependency to version 11.

## [0.1] - 2025-10-17

### Added
- `KhalibClassifier` class
- `calibration_error` function
- `build_reliability_diagram` function

[Unreleased]: https://github.com/khiopslab/khalib/compare/0.7...HEAD
[0.7]: https://github.com/khiopslab/khalib/compare/0.6...0.7
[0.6]: https://github.com/khiopslab/khalib/compare/0.5...0.6
[0.5]: https://github.com/khiopslab/khalib/compare/0.4...0.5
[0.4]: https://github.com/khiopslab/khalib/compare/0.3...0.4
[0.3]: https://github.com/khiopslab/khalib/compare/0.2...0.3
[0.2]: https://github.com/khiopslab/khalib/compare/0.1...0.2
[0.1]: https://github.com/khiopslab/khalib/releases/tag/0.1
