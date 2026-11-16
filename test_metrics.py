import unittest

import numpy as np
from sklearn import metrics

from ml_metrics import binary_report, mean_absolute_error


class MetricsTests(unittest.TestCase):
    def test_zip_truncation_regression(self):
        for actual, predicted in [([1, 0, 1], [1, 0]), ([0], [0, 1])]:
            for metric in (binary_report, mean_absolute_error):
                with self.subTest(metric=metric, actual=actual):
                    with self.assertRaisesRegex(ValueError, "length mismatch"):
                        metric(actual, predicted)

    def test_invalid_vectors_on_either_side(self):
        invalid = [[], [[1]], 1, [np.nan], [np.inf], [-np.inf], ["1"],
                   [1j], [None], np.ma.array([1], mask=[True])]
        for bad in invalid:
            for a, p in [(bad, [1]), ([1], bad)]:
                for metric in (binary_report, mean_absolute_error):
                    with self.subTest(metric=metric, actual=a, predicted=p):
                        with self.assertRaises(ValueError):
                            metric(a, p)

    def test_rejects_nonbinary_labels(self):
        for bad in (-1, 2, 0.5):
            for a, p in [([bad], [1]), ([1], [bad])]:
                with self.assertRaisesRegex(ValueError, "binary labels"):
                    binary_report(a, p)

    def test_orientation(self):
        report = binary_report([0, 0, 1, 1, 1], [1, 0, 0, 0, 1])
        self.assertEqual((report.false_positives, report.false_negatives), (1, 2))
        self.assertEqual(report.accuracy, 0.4)

    def test_binary_against_sklearn(self):
        rng = np.random.default_rng(42)
        for n in (1, 2, 7, 128):
            a, p = rng.integers(0, 2, size=(2, n))
            report = binary_report(a, p)
            tn, fp, fn, tp = metrics.confusion_matrix(a, p, labels=[0, 1]).ravel()
            self.assertEqual((report.true_negatives, report.false_positives,
                              report.false_negatives, report.true_positives), (tn, fp, fn, tp))
            self.assertEqual(report.accuracy, metrics.accuracy_score(a, p))

    def test_boolean_single_class(self):
        self.assertEqual(binary_report([True], [True]).true_positives, 1)
        self.assertEqual(binary_report([False], [False]).true_negatives, 1)

    def test_regression_against_sklearn(self):
        rng = np.random.default_rng(13)
        for n in (1, 5, 64):
            a, p = rng.normal(size=(2, n))
            self.assertAlmostEqual(mean_absolute_error(a, p), metrics.mean_absolute_error(a, p))

    def test_large_integer_precision_and_unsigned_subtraction(self):
        a = np.array([2**63, 0], dtype=np.uint64)
        p = np.array([2**63 + 1, 3], dtype=np.uint64)
        self.assertEqual(mean_absolute_error(a, p), 2.0)

    def test_extreme_finite_mean(self):
        largest = np.finfo(np.float64).max
        self.assertEqual(mean_absolute_error([largest, 0.0], [-largest, 0.0]), largest)

    def test_unrepresentable_mean(self):
        largest = np.finfo(np.float64).max
        with self.assertRaises(OverflowError):
            mean_absolute_error([largest], [-largest])


if __name__ == "__main__":
    unittest.main()
