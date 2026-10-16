import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from mm_jailbreak_guard.cli import main
from mm_jailbreak_guard.io import load_jsonl


class IoAndCliTests(unittest.TestCase):
    def test_duplicate_sample_id_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.jsonl"
            path.write_text('{"sample_id":"x"}\n{"sample_id":"x"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_jsonl(path, unique_key="sample_id")

    def test_fit_rejects_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            image = root / "image.jsonl"
            text = root / "text.jsonl"
            manifest.write_text(
                json.dumps({"sample_id": "x", "user_text": "", "label": 0, "split": "test"}) + "\n",
                encoding="utf-8",
            )
            image.write_text(
                json.dumps({"sample_id": "x", "available": False, "ocr_status": "unavailable"}) + "\n",
                encoding="utf-8",
            )
            text.write_text(
                json.dumps({"sample_id": "x", "available": True, "user_text_score": 0.1}) + "\n",
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "fit",
                        "--manifest",
                        str(manifest),
                        "--image-evidence",
                        str(image),
                        "--text-evidence",
                        str(text),
                        "--output",
                        str(root / "config.json"),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("只允许 validation", stderr.getvalue())

    def test_fit_rejects_missing_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"sample_id": "x", "user_text": "", "label": 0}) + "\n",
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "fit",
                        "--manifest",
                        str(manifest),
                        "--output",
                        str(root / "config.json"),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("<missing>", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
