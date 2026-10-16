from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from . import __version__
from .contracts import (
    DetectionRequest,
    FusionResult,
    ImageEvidence,
    ImageEvidenceProvider,
    TextEvidence,
    TextEvidenceProvider,
)
from .fusion import FusionEngine
from .providers import provider_status


@dataclass(slots=True, frozen=True)
class DetectionReport:
    request: DetectionRequest
    image_evidence: ImageEvidence
    text_evidence: TextEvidence
    fusion: FusionResult
    module_versions: dict[str, str]

    def as_dict(self, *, include_upstream: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "request": self.request.as_dict(),
            "fusion": self.fusion.as_dict(),
            "module_versions": self.module_versions,
        }
        if include_upstream:
            data["image_evidence"] = self.image_evidence.as_dict()
            data["text_evidence"] = self.text_evidence.as_dict()
        return data

    def as_prediction_record(self) -> dict[str, Any]:
        fusion = self.fusion
        return {
            "sample_id": fusion.sample_id,
            "prediction": fusion.prediction,
            "label_name": fusion.label_name,
            "risk_score": fusion.risk_score,
            "threshold": fusion.threshold,
            "mode": fusion.mode,
            "component_scores": fusion.component_scores,
            "effective_weights": fusion.effective_weights,
            "risk_sources": [asdict(item) for item in fusion.risk_sources],
            "warnings": list(fusion.warnings),
            "module_versions": self.module_versions,
            "fusion_config_id": fusion.fusion_config_id,
        }


class DetectionPipeline:
    def __init__(
        self,
        image_provider: ImageEvidenceProvider,
        text_provider: TextEvidenceProvider,
        fusion_engine: FusionEngine,
    ):
        self.image_provider = image_provider
        self.text_provider = text_provider
        self.fusion_engine = fusion_engine

    def detect(self, request: DetectionRequest, mode: str = "full") -> DetectionReport:
        image, text = self.collect_evidence(request)
        fusion = self.fusion_engine.fuse(request, image, text, mode)
        return DetectionReport(
            request=request,
            image_evidence=image,
            text_evidence=text,
            fusion=fusion,
            module_versions={
                "B": image.module_version,
                "C": text.module_version,
                "D": f"D-{__version__}",
            },
        )

    def collect_evidence(
        self, request: DetectionRequest
    ) -> tuple[ImageEvidence, TextEvidence]:
        """运行并隔离 B、C，供单条检测和三方案对比复用。"""

        image = self._run_image(request)
        text = self._run_text(request, image)
        return image, text

    def status(self) -> dict[str, Any]:
        return {
            "B": provider_status(self.image_provider),
            "C": provider_status(self.text_provider),
            "D": {
                "name": "D 融合模块",
                "version": f"D-{__version__}",
                "available": True,
                "config_id": self.fusion_engine.bundle.config_id,
            },
        }

    def _run_image(self, request: DetectionRequest) -> ImageEvidence:
        try:
            evidence = self.image_provider.analyze(request)
            if evidence.sample_id != request.sample_id:
                raise ValueError("B 返回的 sample_id 不一致")
            return evidence
        except Exception as exc:  # 模块隔离：B 失败不应使整个服务退出
            return ImageEvidence.unavailable(
                request.sample_id, f"B 图像模块运行失败：{type(exc).__name__}: {exc}"
            )

    def _run_text(
        self, request: DetectionRequest, image: ImageEvidence
    ) -> TextEvidence:
        try:
            evidence = self.text_provider.analyze(request, image)
            if evidence.sample_id != request.sample_id:
                raise ValueError("C 返回的 sample_id 不一致")
            return evidence
        except Exception as exc:  # 模块隔离：保留 B 结果并显式报告 C 失败
            return TextEvidence.unavailable(
                request.sample_id, f"C 文本模块运行失败：{type(exc).__name__}: {exc}"
            )
