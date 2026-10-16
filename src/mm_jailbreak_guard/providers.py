from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .contracts import (
    DetectionRequest,
    ImageEvidence,
    ImageEvidenceProvider,
    TextEvidence,
    TextEvidenceProvider,
)
from .io import load_jsonl


class UnavailableImageProvider:
    name = "B 图像模块"
    version = "unavailable"

    def analyze(self, request: DetectionRequest) -> ImageEvidence:
        return ImageEvidence.unavailable(request.sample_id, "B 图像模块尚未接入")


class UnavailableTextProvider:
    name = "C 文本模块"
    version = "unavailable"

    def analyze(
        self, request: DetectionRequest, image_evidence: ImageEvidence
    ) -> TextEvidence:
        return TextEvidence.unavailable(request.sample_id, "C 文本模块尚未接入")


class JsonlImageProvider:
    name = "B 图像证据文件"

    def __init__(self, path: str | Path):
        records = load_jsonl(path, unique_key="sample_id")
        self._records = {str(record["sample_id"]): record for record in records}
        self.version = f"jsonl:{Path(path).name}"

    def analyze(self, request: DetectionRequest) -> ImageEvidence:
        record = self._records.get(request.sample_id)
        if record is None:
            return ImageEvidence.unavailable(
                request.sample_id, f"B 证据文件中不存在 {request.sample_id}"
            )
        return ImageEvidence.from_dict(record)


class JsonlTextProvider:
    name = "C 文本证据文件"

    def __init__(self, path: str | Path):
        records = load_jsonl(path, unique_key="sample_id")
        self._records = {str(record["sample_id"]): record for record in records}
        self.version = f"jsonl:{Path(path).name}"

    def analyze(
        self, request: DetectionRequest, image_evidence: ImageEvidence
    ) -> TextEvidence:
        record = self._records.get(request.sample_id)
        if record is None:
            return TextEvidence.unavailable(
                request.sample_id, f"C 证据文件中不存在 {request.sample_id}"
            )
        return TextEvidence.from_dict(record)


def load_provider(spec: str, expected: type[Any]) -> Any:
    """从 `package.module:factory` 加载 B 或 C 的在线实现。"""

    if ":" not in spec:
        raise ValueError("提供方格式必须是 package.module:factory")
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, object_name)
    provider = factory()
    if not isinstance(provider, expected):
        raise TypeError(f"{spec} 未实现所需统一接口")
    return provider


def provider_status(provider: Any) -> dict[str, str | bool]:
    return {
        "name": str(getattr(provider, "name", provider.__class__.__name__)),
        "version": str(getattr(provider, "version", "unknown")),
        "available": not str(getattr(provider, "version", "unknown")).startswith(
            "unavailable"
        ),
    }


def make_image_provider(
    provider_spec: str | None = None, evidence_path: str | None = None
) -> ImageEvidenceProvider:
    if provider_spec and evidence_path:
        raise ValueError("--image-provider 与 --image-evidence 只能选择一个")
    if provider_spec:
        return load_provider(provider_spec, ImageEvidenceProvider)
    if evidence_path:
        return JsonlImageProvider(evidence_path)
    return UnavailableImageProvider()


def make_text_provider(
    provider_spec: str | None = None, evidence_path: str | None = None
) -> TextEvidenceProvider:
    if provider_spec and evidence_path:
        raise ValueError("--text-provider 与 --text-evidence 只能选择一个")
    if provider_spec:
        return load_provider(provider_spec, TextEvidenceProvider)
    if evidence_path:
        return JsonlTextProvider(evidence_path)
    return UnavailableTextProvider()
