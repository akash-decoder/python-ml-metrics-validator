# Python ML Metrics Validator

Three labels and two predictions can produce a misleading 100% accuracy score. Python’s `zip()` stops at the shorter input, so an evaluation can silently skip a missing prediction. This project validates the inputs before it scores them.

## Reproduce the review

From the repository root, run the following with Python 3.10 or later.

```sh
python -m pip install -e ".[test]"
python review_demo.py
python -m unittest -v
```

The demo compares labels `[1, 0, 1]` with predictions `[1, 0]`. Both available predictions match. The naive comparison prints **100%**, while the validator reports **length mismatch: 3 actual, 2 predicted**. Supply `0` for the third prediction and the measured accuracy becomes **2/3**, with **one false negative**. The test `test_zip_truncation_regression` covers missing and extra predictions.

## Score a batch

```python
from ml_metrics import binary_report, mean_absolute_error

report = binary_report([0, 0, 1, 1, 1], [1, 0, 0, 0, 1])
print(report.accuracy, report.false_positives, report.false_negatives)
# 0.4 1 2

print(mean_absolute_error([10, 12, 14], [9, 14, 14]))
# 1.0
```

| Function | Input contract | Result |
| --- | --- | --- |
| `binary_report(actual, predicted)` | Equal-length binary vectors. Label 1 is positive. Boolean labels are accepted. | Accuracy and all four confusion-matrix counts |
| `mean_absolute_error(actual, predicted)` | Equal-length vectors of finite real numbers | Mean absolute error as a Python float |

Both functions accept non-empty, one-dimensional numeric inputs representable by NumPy integer or floating dtypes up to 64 bits, and boolean arrays. They reject NaN, infinity, masked observations, strings, and matrices. Pairing is positional. Align sample IDs before scoring.

## Verification

The tests compare valid batches with scikit-learn’s accuracy, confusion matrix, and mean absolute error. Separate fixtures check false-positive versus false-negative orientation, single-class batches, unequal lengths, and invalid inputs on either side.

The regression implementation uses exact intermediate arithmetic to avoid unsigned wraparound and retain one-unit differences in large integer arrays. This favors reference checks on small batches. A result outside the finite Python float range raises `OverflowError`.
