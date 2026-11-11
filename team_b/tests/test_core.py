import json
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_b import statistic, rows, load_config, ROOT


class Checks(unittest.TestCase):
    def test_direction_and_all_stages(self):
        self.assertAlmostEqual(statistic([0.4, 0.5, 0.1, 0.3]), 0.3)
        self.assertLess(statistic([0.1, 0.2, 0.3]), 0)
        with self.assertRaises(ValueError):
            statistic([0.1, float("nan")])

    def test_manifest_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.jsonl"
            row = {"sample_id": "a", "image_path": "images/a.png", "user_text": "example", "split": "test"}
            path.write_text(json.dumps(row), encoding="utf-8")
            self.assertEqual(rows(path)[0]["image_path"], str(Path(folder) / "images/a.png"))
            path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
            with self.assertRaises(ValueError):
                rows(path)

    def test_no_invented_threshold(self):
        self.assertIsNone(load_config(ROOT / "config.json")["threshold"])


if __name__ == "__main__":
    unittest.main()
