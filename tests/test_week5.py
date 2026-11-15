import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from evaluate_predictions import select_predictions
from mm_jailbreak_guard.cli import main
from mm_jailbreak_guard.io import load_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "examples/handoff"


class Week5Tests(unittest.TestCase):
    def test_raw_cider_stays_in_image_mode_and_outputs_are_readable_by_a(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            evidence = load_jsonl(HANDOFF / "image_evidence.jsonl")
            cider_row = next(row for row in evidence if row["sample_id"] == "mock-002")
            cider_row["visual_score"] = None
            write_jsonl(folder / "image.jsonl", evidence)
            output = folder / "run"
            args = self.arguments(output, folder / "image.jsonl", "synthetic")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 0)
                self.assertEqual(main(args), 2)  # No overwrite.
            image = load_jsonl(output / "predictions.image_only.jsonl")
            full = load_jsonl(output / "predictions.full.jsonl")
            text = load_jsonl(output / "predictions.text_only.jsonl")
            image_predictions, _ = select_predictions(image, "D", "image_adversarial", mode="image_only")
            full_predictions, _ = select_predictions(full, "D", "overall_risk", mode="full")
            text_predictions, _ = select_predictions(text, "D", "overall_risk", mode="text_only")
            self.assertEqual(image_predictions["mock-002"]["prediction"], 1)
            self.assertEqual(text_predictions["mock-002"]["prediction"], 1)
            self.assertIsNone(full_predictions["mock-002"]["prediction"])
            self.assertEqual(full_predictions["mock-001"]["prediction"], 0)
            logs = {row["sample_id"]: row for row in load_jsonl(output / "run_log.jsonl")}
            self.assertFalse(logs["mock-002"]["full_used_both_sides"])
            self.assertEqual(logs["mock-003"]["ocr_status"], "failed")
            self.assertFalse(logs["mock-003"]["ocr_text_score_used"])
            self.assertTrue(logs["mock-003"]["full_used_both_sides"])
            summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "synthetic_only")
            self.assertEqual(summary["decision_counts"], {"image_only": 4, "text_only": 5, "full": 3})

    def test_real_rejects_simulated_inputs_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            args = self.arguments(output, HANDOFF / "image_evidence.jsonl", "real")
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(main(args), 2)
            self.assertIn("不接受模拟数据", err.getvalue())
            self.assertFalse(output.exists())

    @staticmethod
    def arguments(output, image, data_kind):
        return ["week5-batch", "--manifest", str(HANDOFF / "manifest.jsonl"),
                "--image-evidence", str(image), "--text-evidence", str(HANDOFF / "text_evidence.jsonl"),
                "--config", str(ROOT / "configs/fusion_config.week5_fixed.json"),
                "--data-kind", data_kind, "--run-id", "D-week5-test", "--output-dir", str(output)]
