import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_week5_stable_delivery", ROOT / "scripts/build_week5_stable_delivery.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Week5DeliveryTests(unittest.TestCase):
    def test_confusion_names(self):
        self.assertEqual(MODULE.confusion_name(1, 1), "TP")
        self.assertEqual(MODULE.confusion_name(0, 0), "TN")
        self.assertEqual(MODULE.confusion_name(0, 1), "FP")
        self.assertEqual(MODULE.confusion_name(1, 0), "FN")

    def test_frozen_config_does_not_add_features(self):
        config = json.loads((ROOT / "config.week5_stable_v2.json").read_text())
        self.assertFalse(config["threshold_selection"]["test_used_for_selection"])
        self.assertEqual(config["interface_policy"], "freeze ImageEvidence; no new fusion features")
        self.assertEqual(config["threshold"], 0.0022773519158363342)


if __name__ == "__main__":
    unittest.main()
