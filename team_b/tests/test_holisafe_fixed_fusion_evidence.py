import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_holisafe_fixed_fusion_evidence import fit_platt, platt_predict, score_metrics


class FixedFusionEvidenceChecks(unittest.TestCase):
    def test_platt_score_is_bounded_and_monotonic(self):
        statistics = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
        labels = [0, 0, 0, 1, 1, 1]
        model = fit_platt(statistics, labels)
        probabilities = [platt_predict(value, model) for value in statistics]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities))
        self.assertEqual(probabilities, sorted(probabilities))
        self.assertLess(probabilities[0], 0.5)
        self.assertGreater(probabilities[-1], 0.5)

    def test_metrics_use_frozen_half_threshold(self):
        result = score_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
        self.assertEqual(result["accuracy_at_0_5"], 1.0)
        self.assertEqual(result["auroc"], 1.0)
        self.assertAlmostEqual(result["brier"], 0.085)

    def test_platt_rejects_single_class(self):
        with self.assertRaises(ValueError):
            fit_platt([0.0, 1.0], [0, 0])


if __name__ == "__main__":
    unittest.main()
