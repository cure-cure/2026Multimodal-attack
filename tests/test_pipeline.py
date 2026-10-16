import unittest

from mm_jailbreak_guard.contracts import DetectionRequest, ImageEvidence, TextEvidence
from mm_jailbreak_guard.fusion import FusionEngine
from mm_jailbreak_guard.pipeline import DetectionPipeline
from test_fusion import bundle


class BrokenImageProvider:
    name = "B broken"
    version = "test"

    def analyze(self, request):
        raise RuntimeError("simulated failure")


class WorkingTextProvider:
    name = "C test"
    version = "test"

    def analyze(self, request, image_evidence):
        return TextEvidence(request.sample_id, True, 0.1, None, 0.1)


class PipelineTests(unittest.TestCase):
    def test_upstream_failure_isolated(self):
        pipeline = DetectionPipeline(BrokenImageProvider(), WorkingTextProvider(), FusionEngine(bundle()))
        report = pipeline.detect(DetectionRequest("x", "普通文字"), "full")
        self.assertFalse(report.image_evidence.available)
        self.assertEqual(report.fusion.prediction, 0)
        self.assertTrue(any("B 图像模块运行失败" in warning for warning in report.fusion.warnings))


if __name__ == "__main__":
    unittest.main()
