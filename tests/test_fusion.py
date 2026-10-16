import unittest

from mm_jailbreak_guard.contracts import DetectionRequest, ImageEvidence, TextEvidence
from mm_jailbreak_guard.fusion import (
    FusionBundle,
    FusionEngine,
    ModeConfig,
    TuningExample,
    build_feature_vector,
    fit_bundle,
    text_overlap,
)


def bundle():
    return FusionBundle(
        config_id="test",
        trained_on_split="validation",
        algorithm="manual_weighted_late_fusion",
        visual_weight_cap=0.2,
        modes={
            "user_text_only": ModeConfig(0.5, {"user_text_score": 1.0}),
            "user_text_ocr": ModeConfig(
                0.5,
                {
                    "user_text_score": 0.4,
                    "ocr_text_score": 0.3,
                    "combined_text_score": 0.2,
                    "cross_modal_score": 0.1,
                },
            ),
            "full": ModeConfig(
                0.5,
                {
                    "user_text_score": 0.25,
                    "ocr_text_score": 0.2,
                    "combined_text_score": 0.25,
                    "visual_score": 0.1,
                    "cross_modal_score": 0.2,
                },
            ),
        },
        selection={},
    )


class FusionTests(unittest.TestCase):
    def test_character_overlap(self):
        self.assertAlmostEqual(text_overlap("课程安排", "课程安排"), 1.0)
        self.assertEqual(text_overlap("课程安排", "校园通知"), 0.0)

    def test_high_text_scores_produce_risk(self):
        request = DetectionRequest("x", "安全评测占位文本", None)
        image = ImageEvidence("x", True, "安全评测占位文本", "success", 0.1)
        text = TextEvidence("x", True, 0.9, 0.8, 0.9)
        result = FusionEngine(bundle()).fuse(request, image, text, "full")
        self.assertEqual(result.prediction, 1)
        self.assertEqual(result.label_name, "risk")

    def test_visual_only_is_insufficient(self):
        request = DetectionRequest("x", "", None)
        image = ImageEvidence("x", True, "", "no_text", 0.99)
        text = TextEvidence.unavailable("x", "C 未接入")
        result = FusionEngine(bundle()).fuse(request, image, text, "full")
        self.assertIsNone(result.prediction)
        self.assertEqual(result.label_name, "insufficient_evidence")

    def test_unavailable_image_excludes_ocr_and_combined_scores(self):
        request = DetectionRequest("x", "普通文字", None)
        image = ImageEvidence.unavailable("x", "B 未接入")
        text = TextEvidence("x", True, 0.1, 0.99, 0.99)
        result = FusionEngine(bundle()).fuse(request, image, text, "full")
        self.assertEqual(result.prediction, 0)
        self.assertIsNone(result.component_scores["ocr_text_score"])
        self.assertIsNone(result.component_scores["combined_text_score"])
        self.assertIsNone(result.component_scores["visual_score"])
        self.assertIsNone(result.component_scores["cross_modal_score"])
        self.assertEqual(set(result.effective_weights), {"user_text_score"})

    def test_missing_visual_score_is_renormalized(self):
        request = DetectionRequest("x", "普通文字", None)
        image = ImageEvidence("x", True, "普通文字", "success", None)
        text = TextEvidence("x", True, 0.1, 0.1, 0.1)
        result = FusionEngine(bundle()).fuse(request, image, text, "full")
        self.assertAlmostEqual(sum(result.effective_weights.values()), 1.0)
        self.assertIn("visual_score", " ".join(result.warnings))

    def test_cross_modal_score_detects_ocr_concealment(self):
        request = DetectionRequest("x", "请描述图片", None)
        image = ImageEvidence("x", True, "安全评测占位文字", "success", 0.2)
        text = TextEvidence("x", True, 0.1, 0.9, 0.85)
        features = build_feature_vector(request, image, text)
        self.assertGreater(features["cross_modal_score"], 0.55)

    def test_fit_respects_visual_weight_cap(self):
        examples = []
        for index, (label, value) in enumerate(((0, 0.1), (0, 0.2), (1, 0.8), (1, 0.9))):
            examples.append(
                TuningExample(
                    sample_id=str(index),
                    label=label,
                    features={
                        "user_text_score": value,
                        "ocr_text_score": value,
                        "combined_text_score": value,
                        "visual_score": 1.0 - value,
                        "cross_modal_score": value,
                        "text_overlap": 0.0,
                    },
                )
            )
        fitted = fit_bundle(examples, weight_step=0.25, visual_weight_cap=0.2)
        self.assertLessEqual(fitted.modes["full"].weights.get("visual_score", 0), 0.2)
        self.assertEqual(fitted.trained_on_split, "validation")
        self.assertFalse(fitted.selection["test_set_used"])


if __name__ == "__main__":
    unittest.main()
