import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b_pipeline import build_evidence, convolve_reflect, gaussian_kernel, handcrafted_features


class PipelineChecks(unittest.TestCase):
    def test_gaussian_kernel_and_constant_image(self):
        kernel = gaussian_kernel(5, 1.0)
        self.assertEqual(kernel.shape, (5, 5))
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=6)
        self.assertTrue(np.allclose(kernel, kernel[::-1, ::-1]))
        image = np.full((16, 16), 7.0, dtype=np.float32)
        self.assertTrue(np.allclose(convolve_reflect(image, kernel), image, atol=1e-5))

    def test_features_are_bounded_and_constant_is_zero(self):
        image = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
        result = handcrafted_features(image, analysis_size=64)
        for value in result["visual_features"].values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertLess(result["visual_features"]["gaussian_residual_mae"], 1e-6)
        self.assertEqual(result["visual_features"]["high_frequency_ratio"], 0.0)

    def test_evidence_preserves_unknown_cider_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            feature, ocr = root / "features.jsonl", root / "ocr.jsonl"
            output = root / "evidence.jsonl"
            feature.write_text(json.dumps({"sample_id": "s1", "status": "success",
                                           "visual_features": {"high_frequency_ratio": 0.2}}) + "\n")
            ocr.write_text(json.dumps({"sample_id": "s1", "ocr_status": "no_text",
                                       "ocr_text": "", "regions": []}) + "\n")
            build_evidence([{"sample_id": "s1"}], feature, ocr, output)
            row = json.loads(output.read_text())
            self.assertTrue(row["available"])
            self.assertIsNone(row["visual_score"])
            self.assertIsNone(row["cider"])


if __name__ == "__main__":
    unittest.main()
