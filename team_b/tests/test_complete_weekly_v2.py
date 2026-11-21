import importlib.util
import math
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/complete_weekly_v2.py"
SPEC = importlib.util.spec_from_file_location("complete_weekly_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompleteWeeklyV2Tests(unittest.TestCase):
    def test_threshold_selection_balanced(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        threshold, metrics = MODULE.select_threshold(labels, scores)
        self.assertGreater(threshold, 0.2)
        self.assertLess(threshold, 0.8)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)

    def test_strict_gt_boundary(self):
        metrics = MODULE.classification_metrics([0, 1], [0.5, 0.6], 0.5, "gt")
        self.assertEqual((metrics["tn"], metrics["tp"]), (1, 1))

    def test_balanced_logistic_direction(self):
        x = MODULE.np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        y = MODULE.np.asarray([0.0, 0.0, 1.0, 1.0])
        coefficient, intercept = MODULE.fit_balanced_logistic(x, y, iterations=1500)
        self.assertGreater(coefficient[0], 0.0)
        probabilities = MODULE.sigmoid(x @ coefficient + intercept)
        self.assertLess(probabilities[0], probabilities[-1])
        self.assertTrue(all(math.isfinite(value) for value in probabilities))


if __name__ == "__main__":
    unittest.main()
