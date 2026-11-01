from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
from typing import Any, Protocol, runtime_checkable


VALID_OCR_STATUSES = {"success", "no_text", "failed", "unavailable"}
VALID_LABELS = {"normal", "risk", "insufficient_evidence"}


def _boolean(value: Any, field_name: str, *, default: bool = True) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是 JSON 布尔值 true 或 false")
    return value


def _score(value: Any, field_name: str, *, optional: bool = True) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} 必须是 0 到 1 之间的数值")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} 超出 [0, 1]：{number}")
    return number


@dataclass(slots=True, frozen=True)
class DetectionRequest:
    sample_id: str
    user_text: str
    image_path: str | None = None
    image_name: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id or not self.sample_id.strip():
            raise ValueError("sample_id 不能为空")
        if len(self.sample_id) > 128:
            raise ValueError("sample_id 不能超过 128 个字符")
        if self.image_path and not isinstance(self.image_path, str):
            raise ValueError("image_path 必须是字符串或 null")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class TextRegion:
    text: str
    confidence: float | None = None
    box: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        _score(self.confidence, "TextRegion.confidence")
        if self.box is not None and len(self.box) != 4:
            raise ValueError("TextRegion.box 必须包含四个坐标")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextRegion":
        box = data.get("box")
        return cls(
            text=str(data.get("text", "")),
            confidence=_score(data.get("confidence"), "TextRegion.confidence"),
            box=tuple(float(x) for x in box) if box is not None else None,
        )


@dataclass(slots=True, frozen=True)
class EvidenceItem:
    source: str
    description: str
    score: float | None = None
    locations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("EvidenceItem.source 不能为空")
        if not self.description:
            raise ValueError("EvidenceItem.description 不能为空")
        _score(self.score, "EvidenceItem.score")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceItem":
        return cls(
            source=str(data.get("source", "unknown")),
            description=str(data.get("description", "未提供描述")),
            score=_score(data.get("score"), "EvidenceItem.score"),
            locations=tuple(str(x) for x in data.get("locations", ())),
        )


@dataclass(slots=True, frozen=True)
class CiderEvidence:
    """B 提供的可追溯中间证据；D 不在此复现 CIDER。"""
    similarities: tuple[float, ...]
    statistic: float
    statistic_name: str
    threshold: float
    comparison: str
    config_id: str
    reproduction: str

    def __post_init__(self) -> None:
        if len(self.similarities) < 2:
            raise ValueError("CIDER 至少提供原图和一个去噪阶段的相似度")
        for value in (*self.similarities, self.statistic, self.threshold):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("CIDER 原始统计量必须是有限数值")
        if self.comparison not in {"ge", "le"}:
            raise ValueError("CIDER comparison 必须是 ge 或 le")
        if not self.statistic_name or not self.config_id:
            raise ValueError("CIDER 需提供 statistic_name 和 config_id")
        if self.reproduction not in {"original", "adapted", "mock"}:
            raise ValueError("CIDER reproduction 必须是 original/adapted/mock")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CiderEvidence":
        return cls(tuple(data["similarities"]), data["statistic"], str(data["statistic_name"]),
                   data["threshold"], str(data["comparison"]), str(data["config_id"]), str(data["reproduction"]))


@dataclass(slots=True, frozen=True)
class ImageEvidence:
    sample_id: str
    available: bool
    ocr_text: str = ""
    ocr_status: str = "unavailable"
    visual_score: float | None = None
    visual_features: dict[str, float] = field(default_factory=dict)
    regions: tuple[TextRegion, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    warnings: tuple[str, ...] = ()
    module_version: str = "unknown"
    cider: CiderEvidence | None = None

    def __post_init__(self) -> None:
        if not self.sample_id or not self.sample_id.strip():
            raise ValueError("ImageEvidence.sample_id 不能为空")
        if self.ocr_status not in VALID_OCR_STATUSES:
            raise ValueError(f"未知 ocr_status：{self.ocr_status}")
        _score(self.visual_score, "ImageEvidence.visual_score")
        for name, value in self.visual_features.items():
            _score(value, f"ImageEvidence.visual_features.{name}")
        if not self.available and self.ocr_status not in {"failed", "unavailable"}:
            raise ValueError("图像模块不可用时 ocr_status 必须为 failed 或 unavailable")
        if not self.available and (
            self.ocr_text
            or self.visual_score is not None
            or self.visual_features
            or self.regions
            or self.cider is not None
        ):
            raise ValueError("图像模块不可用时不得携带 OCR 或视觉分数")

    @classmethod
    def unavailable(cls, sample_id: str, reason: str) -> "ImageEvidence":
        return cls(
            sample_id=sample_id,
            available=False,
            ocr_status="unavailable",
            warnings=(reason,),
            module_version="unavailable",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageEvidence":
        features: dict[str, float] = {}
        for key, value in dict(data.get("visual_features", {})).items():
            score = _score(value, f"ImageEvidence.visual_features.{key}", optional=False)
            features[str(key)] = float(score)
        return cls(
            sample_id=str(data["sample_id"]),
            available=_boolean(data.get("available"), "ImageEvidence.available"),
            ocr_text=str(data.get("ocr_text", "")),
            ocr_status=str(data.get("ocr_status", "unavailable")),
            visual_score=_score(data.get("visual_score"), "ImageEvidence.visual_score"),
            visual_features=features,
            regions=tuple(TextRegion.from_dict(x) for x in data.get("regions", ())),
            evidence=tuple(EvidenceItem.from_dict(x) for x in data.get("evidence", ())),
            warnings=tuple(str(x) for x in data.get("warnings", ())),
            module_version=str(data.get("module_version", "unknown")),
            cider=CiderEvidence.from_dict(data["cider"]) if data.get("cider") is not None else None,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class TextEvidence:
    sample_id: str
    available: bool
    user_text_score: float | None = None
    ocr_text_score: float | None = None
    combined_text_score: float | None = None
    evidence: tuple[EvidenceItem, ...] = ()
    warnings: tuple[str, ...] = ()
    module_version: str = "unknown"

    def __post_init__(self) -> None:
        if not self.sample_id or not self.sample_id.strip():
            raise ValueError("TextEvidence.sample_id 不能为空")
        _score(self.user_text_score, "TextEvidence.user_text_score")
        _score(self.ocr_text_score, "TextEvidence.ocr_text_score")
        _score(self.combined_text_score, "TextEvidence.combined_text_score")
        if self.available and all(
            value is None
            for value in (
                self.user_text_score,
                self.ocr_text_score,
                self.combined_text_score,
            )
        ):
            raise ValueError("文本模块 available=true 时至少应提供一个风险分数")
        if not self.available and any(
            value is not None
            for value in (
                self.user_text_score,
                self.ocr_text_score,
                self.combined_text_score,
            )
        ):
            raise ValueError("文本模块不可用时不得携带风险分数")

    @classmethod
    def unavailable(cls, sample_id: str, reason: str) -> "TextEvidence":
        return cls(
            sample_id=sample_id,
            available=False,
            warnings=(reason,),
            module_version="unavailable",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextEvidence":
        return cls(
            sample_id=str(data["sample_id"]),
            available=_boolean(data.get("available"), "TextEvidence.available"),
            user_text_score=_score(
                data.get("user_text_score"), "TextEvidence.user_text_score"
            ),
            ocr_text_score=_score(
                data.get("ocr_text_score"), "TextEvidence.ocr_text_score"
            ),
            combined_text_score=_score(
                data.get("combined_text_score"), "TextEvidence.combined_text_score"
            ),
            evidence=tuple(EvidenceItem.from_dict(x) for x in data.get("evidence", ())),
            warnings=tuple(str(x) for x in data.get("warnings", ())),
            module_version=str(data.get("module_version", "unknown")),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class FusionResult:
    sample_id: str
    label_name: str
    prediction: int | None
    risk_score: float | None
    threshold: float
    mode: str
    component_scores: dict[str, float | None]
    effective_weights: dict[str, float]
    risk_sources: tuple[EvidenceItem, ...] = ()
    warnings: tuple[str, ...] = ()
    fusion_config_id: str = "unknown"
    decision_rule: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.label_name not in VALID_LABELS:
            raise ValueError(f"未知 label_name：{self.label_name}")
        if self.prediction not in {0, 1, None}:
            raise ValueError("prediction 必须是 0、1 或 null")
        _score(self.risk_score, "FusionResult.risk_score")
        _score(self.threshold, "FusionResult.threshold", optional=False)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["prediction_target"] = "image_adversarial" if self.mode == "image_only" else "overall_risk"
        return data


@runtime_checkable
class ImageEvidenceProvider(Protocol):
    name: str
    version: str

    def analyze(self, request: DetectionRequest) -> ImageEvidence:
        """读取图片并返回 B 约定的证据。"""


@runtime_checkable
class TextEvidenceProvider(Protocol):
    name: str
    version: str

    def analyze(
        self, request: DetectionRequest, image_evidence: ImageEvidence
    ) -> TextEvidence:
        """分析用户文字与 OCR 文字并返回 C 约定的证据。"""


def validate_image_path(request: DetectionRequest) -> Path | None:
    if request.image_path is None:
        return None
    path = Path(request.image_path)
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在：{path}")
    return path
