"""Validate paired observations before computing metrics."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class BinaryReport:
    accuracy: float
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int


def _vector(values: ArrayLike, name: str) -> np.ndarray:
    if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
        raise ValueError(f"{name} contains masked observations")
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if array.dtype.kind not in "biuf" or array.dtype.itemsize > 8:
        raise ValueError(f"{name} must contain real numeric values up to 64 bits")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _paired(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    actual = _vector(actual, "actual")
    predicted = _vector(predicted, "predicted")
    if actual.size != predicted.size:
        raise ValueError(f"length mismatch: {actual.size} actual, {predicted.size} predicted")
    return actual, predicted


def binary_report(actual: ArrayLike, predicted: ArrayLike) -> BinaryReport:
    """Compute binary accuracy and confusion counts with 1 as the positive label."""
    actual, predicted = _paired(actual, predicted)
    for values in (actual, predicted):
        if not ((values == 0) | (values == 1)).all():
            raise ValueError("binary labels must be 0 or 1")
    positive = actual == 1
    called_positive = predicted == 1
    return BinaryReport(
        accuracy=float(np.mean(positive == called_positive)),
        false_positives=int(np.count_nonzero(~positive & called_positive)),
        false_negatives=int(np.count_nonzero(positive & ~called_positive)),
        true_positives=int(np.count_nonzero(positive & called_positive)),
        true_negatives=int(np.count_nonzero(~positive & ~called_positive)),
    )


def mean_absolute_error(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Average absolute error using exact intermediate arithmetic for reference checks."""
    actual, predicted = _paired(actual, predicted)
    # Subtracting in the array dtype can wrap unsigned integers. Converting
    # to float first can erase a one-unit difference between large integers.
    errors = (
        abs(Fraction(a) - Fraction(p))
        for a, p in zip(actual.tolist(), predicted.tolist(), strict=True)
    )
    return float(sum(errors) / actual.size)
