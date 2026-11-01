import json
import tempfile
import unittest
from pathlib import Path

from mm_jailbreak_guard.contracts import CiderEvidence, DetectionRequest, ImageEvidence, TextEvidence
from mm_jailbreak_guard.fusion import FusionBundle, FusionEngine
from mm_jailbreak_guard.handoff import validate_handoff
from mm_jailbreak_guard.cli import main

ROOT = Path(__file__).resolve().parents[1]


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.engine = FusionEngine(FusionBundle.load(ROOT / "configs/fusion_config.example.json"))

    def test_overlap_kept_for_explanation_without_affecting_weights(self):
        request = DetectionRequest("x", "相同文字")
        image = ImageEvidence("x", True, "相同文字", "success", 0.1)
        text = TextEvidence("x", True, 0.1, 0.1, 0.1)
        for mode in ("full", "text_only", "user_text_ocr"):
            with self.subTest(mode=mode):
                result = self.engine.fuse(request, image, text, mode)
                self.assertEqual(result.as_dict()["component_scores"]["text_overlap"], 1.0)
                self.assertNotIn("text_overlap", result.effective_weights)
                if mode == "text_only":
                    self.assertAlmostEqual(result.risk_score, 0.1)
        for mode in ("image_only", "user_text_only"):
            with self.subTest(mode=mode):
                self.assertIsNone(self.engine.fuse(request, image, text, mode).component_scores["text_overlap"])

    @staticmethod
    def manifest_row():
        return dict(sample_id="a", image_path=None, user_text="", source="mock",
                    split="fusion_train", image_group_id="i1", request_group_id="r1")

    def test_manifest_requires_all_documented_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            for field in self.manifest_row():
                row = self.manifest_row()
                del row[field]
                with self.subTest(field=field):
                    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, field):
                        validate_handoff(path)

    def test_manifest_rejects_invalid_input_field_types(self):
        invalid = (("user_text", None), ("user_text", 123), ("image_path", False),
                   ("image_path", " "), ("source", None), ("source", " "),
                   ("split", []))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            for field, value in invalid:
                row = self.manifest_row()
                row[field] = value
                with self.subTest(field=field, value=value):
                    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, field):
                        validate_handoff(path)

    def test_manifest_accepts_empty_text_and_explicit_missing_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            path.write_text(json.dumps(self.manifest_row()) + "\n", encoding="utf-8")
            self.assertEqual(validate_handoff(path)["status"], "valid")

    def test_mock_package_and_three_modes(self):
        p = ROOT / "examples/handoff"
        result = validate_handoff(p / "manifest.jsonl", p / "image_evidence.jsonl", p / "text_evidence.jsonl")
        self.assertEqual(result["sample_count"], 6)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "predictions.jsonl"
            code = main(["compare", "--manifest", str(p / "manifest.jsonl"),
                         "--image-evidence", str(p / "image_evidence.jsonl"),
                         "--text-evidence", str(p / "text_evidence.jsonl"),
                         "--config", str(ROOT / "configs/fusion_config.example.json"),
                         "--output", str(output)])
            self.assertEqual(code, 0)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(set(rows[0]["schemes"]), {"image_only", "text_only", "full"})
            self.assertEqual(rows[1]["schemes"]["image_only"]["image_decision"]["threshold"], 0.2)
            self.assertTrue(all(s["prediction"] is None for s in rows[-1]["schemes"].values()))

    def test_cider_raw_negative_values_roundtrip(self):
        c = CiderEvidence((-0.2, 0.1), 0.3, "delta", 0.2, "ge", "B-test", "mock")
        image = ImageEvidence("x", True, cider=c)
        restored = ImageEvidence.from_dict(image.as_dict())
        self.assertEqual(restored.cider, c)
        result = self.engine.fuse(DetectionRequest("x", "参照文字"), image,
                                  TextEvidence.unavailable("x", "未接入"), "image_only")
        self.assertEqual(result.prediction, 1)

    def test_cider_rejects_nonfinite_and_wrong_direction(self):
        for stat, direction in ((float("nan"), "ge"), (0.3, "bad")):
            with self.assertRaises(ValueError):
                CiderEvidence((0.1, 0.2), stat, "delta", 0.2, direction, "B-test", "mock")

    def test_text_only_ignores_visual_and_failed_ocr(self):
        image = ImageEvidence("x", True, ocr_status="failed", visual_score=1.0)
        text = TextEvidence("x", True, 0.1, 0.99, 0.99)
        result = self.engine.fuse(DetectionRequest("x", "普通文字"), image, text, "text_only")
        self.assertEqual(result.prediction, 0)
        self.assertIsNone(result.component_scores["visual_score"])
        self.assertIsNone(result.component_scores["ocr_text_score"])

    def test_image_only_ignores_high_text_scores(self):
        image = ImageEvidence("x", True, visual_score=0.1)
        text = TextEvidence("x", True, 0.99)
        result = self.engine.fuse(DetectionRequest("x", "文字"), image, text, "image_only")
        self.assertEqual(result.prediction, 0)
        self.assertIsNone(result.component_scores["user_text_score"])

    def test_group_leak_and_missing_upstream_rejected(self):
        rows = [dict(self.manifest_row(), sample_id="a", split="fusion_train", image_group_id="same", request_group_id="r1"),
                dict(self.manifest_row(), sample_id="b", split="test", image_group_id="same", request_group_id="r2")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "跨划分泄漏"):
                validate_handoff(path)
            rows[1]["image_group_id"] = "different"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            evidence = Path(tmp) / "image.jsonl"
            evidence.write_text('{"sample_id":"a","available":false}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "编号不一致"):
                validate_handoff(path, evidence)


if __name__ == "__main__":
    unittest.main()
