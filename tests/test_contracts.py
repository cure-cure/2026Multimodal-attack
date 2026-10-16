import unittest

from mm_jailbreak_guard.contracts import ImageEvidence, TextEvidence


class ContractTests(unittest.TestCase):
    def test_score_must_be_in_unit_interval(self):
        with self.assertRaises(ValueError):
            ImageEvidence(
                sample_id="x",
                available=True,
                ocr_status="success",
                visual_score=1.2,
            )

    def test_available_text_requires_at_least_one_score(self):
        with self.assertRaises(ValueError):
            TextEvidence(sample_id="x", available=True)

    def test_unavailable_image_cannot_claim_success(self):
        with self.assertRaises(ValueError):
            ImageEvidence(sample_id="x", available=False, ocr_status="success")

    def test_unavailable_image_cannot_carry_scores(self):
        with self.assertRaises(ValueError):
            ImageEvidence(
                sample_id="x",
                available=False,
                ocr_status="unavailable",
                visual_score=0.9,
            )

    def test_unavailable_text_cannot_carry_scores(self):
        with self.assertRaises(ValueError):
            TextEvidence(sample_id="x", available=False, user_text_score=0.9)

    def test_available_must_be_json_boolean(self):
        with self.assertRaises(ValueError):
            ImageEvidence.from_dict(
                {"sample_id": "x", "available": "false", "ocr_status": "unavailable"}
            )

    def test_visual_feature_rejects_boolean(self):
        with self.assertRaises(ValueError):
            ImageEvidence.from_dict(
                {
                    "sample_id": "x",
                    "available": True,
                    "ocr_status": "no_text",
                    "visual_features": {"high_frequency_ratio": True},
                }
            )


if __name__ == "__main__":
    unittest.main()
