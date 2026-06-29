import unittest

from coca.metrics import auroc_score, brier_score, compute_eval_metrics, expected_calibration_error


class MetricsTest(unittest.TestCase):
    def test_brier(self):
        self.assertAlmostEqual(brier_score([0.0, 1.0], [False, True]), 0.0)

    def test_ece_perfect(self):
        self.assertAlmostEqual(expected_calibration_error([0.0, 1.0], [False, True]), 0.0)

    def test_auroc(self):
        self.assertAlmostEqual(auroc_score([0.1, 0.9], [False, True]), 1.0)

    def test_compute_metrics_skips_invalid_confidence_for_calibration(self):
        metrics = compute_eval_metrics([None, 0.9], [False, True])
        self.assertEqual(metrics.count, 2)
        self.assertEqual(metrics.parse_success_rate, 0.5)
        self.assertAlmostEqual(metrics.accuracy, 0.5)


if __name__ == "__main__":
    unittest.main()

