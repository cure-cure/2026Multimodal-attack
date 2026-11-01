import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from mm_jailbreak_guard.fusion import FusionBundle, FusionEngine
from mm_jailbreak_guard.io import JsonlAuditLog
from mm_jailbreak_guard.pipeline import DetectionPipeline
from mm_jailbreak_guard.providers import JsonlImageProvider, JsonlTextProvider
from mm_jailbreak_guard.web import make_handler
from mm_jailbreak_guard.contracts import DetectionRequest

ROOT = Path(__file__).resolve().parents[1]


class WebRequestTests(unittest.TestCase):
    def test_page_mode_switch_and_batch_consistency(self):
        folder = ROOT / "examples/handoff"
        pipeline = DetectionPipeline(JsonlImageProvider(folder / "image_evidence.jsonl"),
                                     JsonlTextProvider(folder / "text_evidence.jsonl"),
                                     FusionEngine(FusionBundle.load(ROOT / "configs/fusion_config.example.json")))
        with tempfile.TemporaryDirectory() as tmp:
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(pipeline, JsonlAuditLog(Path(tmp) / "log.jsonl")))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base, timeout=5) as response:
                    self.assertIn('name="mode"', response.read().decode("utf-8"))
                for mode in ("image_only", "text_only", "full"):
                    fields = {"sample_id": "mock-002", "user_text": "模拟风险占位", "mode": mode}
                    body = "".join(f'--check\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
                                   for key, value in fields.items()) + "--check--\r\n"
                    request = Request(base + "/api/detect", data=body.encode("utf-8"),
                                      headers={"Content-Type": "multipart/form-data; boundary=check"})
                    with urlopen(request, timeout=5) as response:
                        result = json.load(response)["fusion"]
                    self.assertEqual(result["mode"], mode)
                    self.assertEqual(result["prediction"], 1)
                    expected = pipeline.detect(DetectionRequest("mock-002", "模拟风险占位"), mode).fusion
                    self.assertEqual(result["prediction"], expected.prediction)
                    self.assertEqual(result["risk_score"], expected.risk_score)
                    if mode == "image_only":
                        self.assertEqual(result["decision_rule"]["threshold"], 0.2)
                        self.assertEqual(result["prediction_target"], "image_adversarial")
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)
