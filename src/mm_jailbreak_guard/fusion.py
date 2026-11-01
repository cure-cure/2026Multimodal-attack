from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .contracts import DetectionRequest, EvidenceItem, FusionResult, ImageEvidence, TextEvidence


MODE_FEATURES: dict[str, tuple[str, ...]] = {
    "image_only": ("visual_score",),
    "text_only": ("user_text_score", "ocr_text_score", "combined_text_score"),
    "user_text_only": ("user_text_score",),
    "user_text_ocr": (
        "user_text_score",
        "ocr_text_score",
        "combined_text_score",
        "cross_modal_score",
    ),
    "full": (
        "user_text_score",
        "ocr_text_score",
        "combined_text_score",
        "visual_score",
        "cross_modal_score",
    ),
}
TEXT_FEATURES = {"user_text_score", "ocr_text_score", "combined_text_score"}
PRIMARY_MODES = ("image_only", "text_only", "full")


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _character_ngrams(text: str, n: int = 2) -> set[str]:
    normalized = re.sub(r"\s+", "", text.casefold())
    if not normalized:
        return set()
    if len(normalized) < n:
        return {normalized}
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def text_overlap(user_text: str, ocr_text: str) -> float:
    """手写字符二元组 Jaccard 相似度，适用于中英文混合文本。"""

    left = _character_ngrams(user_text)
    right = _character_ngrams(ocr_text)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def build_feature_vector(
    request: DetectionRequest,
    image: ImageEvidence,
    text: TextEvidence,
) -> dict[str, float | None]:
    image_available = image.available
    text_available = text.available
    overlap = text_overlap(request.user_text, image.ocr_text) if image_available else 0.0
    user_score = text.user_text_score if text_available else None
    ocr_available = image_available and image.ocr_status == "success"
    ocr_score = text.ocr_text_score if text_available and ocr_available else None
    combined_score = text.combined_text_score if text_available and ocr_available else None

    cross_score: float | None = None
    if ocr_score is not None and (user_score is not None or combined_score is not None):
        user_value = user_score if user_score is not None else combined_score
        combined_value = combined_score
        if combined_value is None:
            combined_value = (float(user_value) + ocr_score) / 2.0
        strongest = max(float(user_value), ocr_score)
        synergy = math.sqrt(_clip(combined_value) * _clip(strongest))
        concealment = max(0.0, ocr_score - float(user_value)) * (1.0 - overlap)
        cross_score = _clip(
            0.7 * synergy * (0.75 + 0.25 * overlap) + 0.3 * concealment
        )

    return {
        "user_text_score": user_score,
        "ocr_text_score": ocr_score,
        "combined_text_score": combined_score,
        "visual_score": image.visual_score if image_available else None,
        "cross_modal_score": cross_score,
        "text_overlap": overlap,
    }


@dataclass(slots=True, frozen=True)
class ModeConfig:
    threshold: float
    weights: dict[str, float]

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("融合阈值必须位于 [0, 1]")
        if not self.weights or any(value < 0 for value in self.weights.values()):
            raise ValueError("融合权重必须非空且非负")
        total = sum(self.weights.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"融合权重之和必须为 1，当前为 {total}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModeConfig":
        return cls(
            threshold=float(data["threshold"]),
            weights={str(k): float(v) for k, v in data["weights"].items()},
        )

    def as_dict(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "weights": self.weights}


@dataclass(slots=True, frozen=True)
class FusionBundle:
    config_id: str
    trained_on_split: str | None
    algorithm: str
    visual_weight_cap: float
    modes: dict[str, ModeConfig]
    selection: dict[str, Any]

    def __post_init__(self) -> None:
        missing = {"user_text_only", "user_text_ocr", "full"} - set(self.modes)
        extra = set(self.modes) - set(MODE_FEATURES)
        if missing:
            raise ValueError(f"融合配置缺少模式：{sorted(missing)}")
        if extra:
            raise ValueError(f"融合配置包含未知模式：{sorted(extra)}")
        for mode, config in self.modes.items():
            unexpected = set(config.weights) - set(MODE_FEATURES[mode])
            if unexpected:
                raise ValueError(f"{mode} 含不允许的特征：{sorted(unexpected)}")
        visual_weight = self.modes["full"].weights.get("visual_score", 0.0)
        if visual_weight > self.visual_weight_cap + 1e-9:
            raise ValueError("视觉异常权重超过配置上限")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FusionBundle":
        return cls(
            config_id=str(data["config_id"]),
            trained_on_split=data.get("trained_on_split"),
            algorithm=str(data.get("algorithm", "manual_weighted_late_fusion")),
            visual_weight_cap=float(data.get("visual_weight_cap", 0.2)),
            modes={k: ModeConfig.from_dict(v) for k, v in data["modes"].items()},
            selection=dict(data.get("selection", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FusionBundle":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "trained_on_split": self.trained_on_split,
            "algorithm": self.algorithm,
            "visual_weight_cap": self.visual_weight_cap,
            "modes": {name: mode.as_dict() for name, mode in self.modes.items()},
            "selection": self.selection,
        }


class FusionEngine:
    """D 自主实现的可解释加权后融合器。"""

    def __init__(self, bundle: FusionBundle):
        self.bundle = bundle

    def fuse(
        self,
        request: DetectionRequest,
        image: ImageEvidence,
        text: TextEvidence,
        mode: str = "full",
    ) -> FusionResult:
        if mode not in MODE_FEATURES:
            raise ValueError(f"未知融合模式：{mode}")
        if image.sample_id != request.sample_id or text.sample_id != request.sample_id:
            raise ValueError("上下游 sample_id 与请求不一致")

        features = build_feature_vector(request, image, text)
        config = self.bundle.modes.get(mode)
        if config is None:
            config = (ModeConfig(0.5, {"visual_score": 1.0}) if mode == "image_only"
                      else ModeConfig(0.5, {"user_text_score": 0.4, "ocr_text_score": 0.3,
                                           "combined_text_score": 0.3}))
        allowed = set(MODE_FEATURES[mode])
        # 重合度只用于解释，不加入权重；含 OCR 的模式仍需将它返回给页面。
        if mode in {"full", "text_only", "user_text_ocr"}:
            allowed.add("text_overlap")
        features = {name: value if name in allowed else None for name, value in features.items()}
        warnings = list(image.warnings) + list(text.warnings)
        if mode == "image_only":
            warnings.append("仅图像输出针对图像对抗扰动，不代表图文整体风险；CIDER 仍使用用户文字作为语义参照")
            if image.cider is None:
                warnings.append("缺少 CIDER 证据；visual_score 仅作图像分数基线，不能宣称为 CIDER 复现")
        missing = [name for name in config.weights if features.get(name) is None]
        if missing:
            warnings.append("缺失证据已从本次融合中移除：" + ", ".join(missing))

        has_text_score = any(features.get(name) is not None for name in TEXT_FEATURES)
        if not has_text_score and mode != "image_only":
            return FusionResult(
                sample_id=request.sample_id,
                label_name="insufficient_evidence",
                prediction=None,
                risk_score=None,
                threshold=config.threshold,
                mode=mode,
                component_scores=features,
                effective_weights={},
                warnings=tuple(warnings + ["没有可用文本风险分数，拒绝仅凭图像噪声判定"]),
                fusion_config_id=self.bundle.config_id,
            )

        usable = {
            name: weight
            for name, weight in config.weights.items()
            if weight > 0 and features.get(name) is not None
        }
        total_weight = sum(usable.values())
        if mode == "image_only" and image.cider is not None:
            c = image.cider
            prediction = int(c.statistic >= c.threshold if c.comparison == "ge" else c.statistic <= c.threshold)
            return FusionResult(request.sample_id, "risk" if prediction else "normal",
                                prediction, image.visual_score, config.threshold, mode,
                                features, {"visual_score": 1.0} if image.visual_score is not None else {},
                                image.evidence if prediction else (), tuple(warnings + ["判定使用 B 的原始统计量阈值，见 image_evidence.cider"]),
                                self.bundle.config_id,
                                {"statistic": c.statistic, "statistic_name": c.statistic_name,
                                 "threshold": c.threshold, "comparison": c.comparison, "config_id": c.config_id})
        if total_weight <= 0:
            return FusionResult(
                sample_id=request.sample_id,
                label_name="insufficient_evidence",
                prediction=None,
                risk_score=None,
                threshold=config.threshold,
                mode=mode,
                component_scores=features,
                effective_weights={},
                warnings=tuple(warnings + ["当前模式没有可用融合特征"]),
                fusion_config_id=self.bundle.config_id,
            )

        effective = {name: weight / total_weight for name, weight in usable.items()}
        score = _clip(
            sum(effective[name] * float(features[name]) for name in effective)
        )
        prediction = int(score >= config.threshold)
        sources = self._select_sources(
            prediction, score, config.threshold, features, image, text
        )
        if mode == "image_only":
            sources = [item for item in sources if item.source in {"image", "visual_score", "cider"}]
        elif mode != "full":
            sources = [item for item in sources if item.source not in {"image", "visual_score", "cider"}]
        return FusionResult(
            sample_id=request.sample_id,
            label_name="risk" if prediction else "normal",
            prediction=prediction,
            risk_score=score,
            threshold=config.threshold,
            mode=mode,
            component_scores=features,
            effective_weights=effective,
            risk_sources=tuple(sources),
            warnings=tuple(warnings),
            fusion_config_id=self.bundle.config_id,
        )

    @staticmethod
    def _select_sources(
        prediction: int,
        score: float,
        threshold: float,
        features: dict[str, float | None],
        image: ImageEvidence,
        text: TextEvidence,
    ) -> list[EvidenceItem]:
        if prediction == 0:
            return []
        cutoff = max(0.4, threshold * 0.75)
        items = [
            item
            for item in (*text.evidence, *image.evidence)
            if item.score is not None and item.score >= cutoff
        ]
        cross = features.get("cross_modal_score")
        if cross is not None and cross >= cutoff:
            items.append(
                EvidenceItem(
                    source="cross_modal",
                    description="用户文字与图像文字形成风险协同或可疑差异",
                    score=cross,
                )
            )
        if not items:
            strongest_name, strongest_value = max(
                (
                    (name, value)
                    for name, value in features.items()
                    if name != "text_overlap" and value is not None
                ),
                key=lambda pair: float(pair[1]),
            )
            items.append(
                EvidenceItem(
                    source=strongest_name,
                    description="该分量对融合风险分数贡献最高",
                    score=float(strongest_value),
                )
            )
        return items


@dataclass(slots=True, frozen=True)
class TuningExample:
    sample_id: str
    label: int
    features: dict[str, float | None]


def _integer_compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in _integer_compositions(total - first, parts - 1):
            yield (first, *tail)


def _weighted_score(
    features: dict[str, float | None], weights: dict[str, float]
) -> float | None:
    usable = {
        name: weight
        for name, weight in weights.items()
        if weight > 0 and features.get(name) is not None
    }
    total = sum(usable.values())
    if total <= 0:
        return None
    return _clip(
        sum(weight * float(features[name]) for name, weight in usable.items()) / total
    )


def _metrics(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return {
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def tune_mode(
    examples: list[TuningExample],
    mode: str,
    *,
    weight_step: float = 0.1,
    thresholds: Iterable[float] | None = None,
    visual_weight_cap: float = 0.2,
) -> tuple[ModeConfig, dict[str, Any]]:
    """在验证集手写枚举权重和阈值，不依赖自动调参库。"""

    if mode not in MODE_FEATURES:
        raise ValueError(f"未知融合模式：{mode}")
    if not 0 < weight_step <= 1 or not math.isclose(1 / weight_step, round(1 / weight_step)):
        raise ValueError("weight_step 必须能整除 1，例如 0.1、0.2、0.25")
    labels_present = {example.label for example in examples}
    if labels_present != {0, 1}:
        raise ValueError("验证集必须同时包含正常与风险样本")

    features = MODE_FEATURES[mode]
    units = round(1 / weight_step)
    threshold_values = list(thresholds or (x / 20 for x in range(4, 17)))
    if mode == "user_text_only":
        candidates = [(units,)]
    else:
        candidates = _integer_compositions(units, len(features))

    best_key: tuple[float, ...] | None = None
    best_config: ModeConfig | None = None
    best_metrics: dict[str, Any] | None = None
    evaluated = 0

    for parts in candidates:
        weights = {name: part / units for name, part in zip(features, parts)}
        if mode == "full" and weights.get("visual_score", 0.0) > visual_weight_cap + 1e-9:
            continue
        if mode != "image_only" and not any(weights.get(name, 0.0) > 0 for name in TEXT_FEATURES):
            continue
        sample_scores: list[tuple[int, float]] = []
        for example in examples:
            score = _weighted_score(example.features, weights)
            if score is not None and (mode == "image_only" or any(
                example.features.get(name) is not None for name in TEXT_FEATURES
            )):
                sample_scores.append((example.label, score))
        if len(sample_scores) != len(examples):
            continue
        labels = [label for label, _ in sample_scores]
        for threshold in threshold_values:
            predictions = [int(score >= threshold) for _, score in sample_scores]
            metrics = _metrics(labels, predictions)
            key = (
                float(metrics["balanced_accuracy"]),
                float(metrics["f1"]),
                float(metrics["accuracy"]),
                -abs(float(threshold) - 0.5),
                -weights.get("visual_score", 0.0),
            )
            evaluated += 1
            if best_key is None or key > best_key:
                best_key = key
                best_config = ModeConfig(float(threshold), weights)
                best_metrics = metrics

    if best_config is None or best_metrics is None:
        raise ValueError(f"{mode} 没有可用的权重组合；请检查缺失证据")
    return best_config, {**best_metrics, "candidates_evaluated": evaluated}


def fit_bundle(
    examples: list[TuningExample],
    *,
    weight_step: float = 0.1,
    visual_weight_cap: float = 0.2,
) -> FusionBundle:
    if not examples:
        raise ValueError("验证集为空")
    modes: dict[str, ModeConfig] = {}
    mode_metrics: dict[str, Any] = {}
    for mode in MODE_FEATURES:
        if mode == "image_only":
            # 图像扰动标签与整体风险标签不同，禁止用整体 label 调图像阈值。
            modes[mode] = ModeConfig(0.5, {"visual_score": 1.0})
            mode_metrics[mode] = {"status": "unvalidated_use_B_threshold"}
            continue
        modes[mode], mode_metrics[mode] = tune_mode(
            examples,
            mode,
            weight_step=weight_step,
            visual_weight_cap=visual_weight_cap,
        )
    created_at = datetime.now(timezone.utc).isoformat()
    fingerprint_source = json.dumps(
        {name: config.as_dict() for name, config in modes.items()},
        ensure_ascii=False,
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:12]
    return FusionBundle(
        config_id=f"validation-{fingerprint}",
        trained_on_split="validation",
        algorithm="manual_weighted_late_fusion",
        visual_weight_cap=visual_weight_cap,
        modes=modes,
        selection={
            "status": "fitted",
            "created_at_utc": created_at,
            "sample_count": len(examples),
            "weight_step": weight_step,
            "objective": "balanced_accuracy_then_f1_then_accuracy",
            "mode_metrics": mode_metrics,
            "test_set_used": False,
        },
    )
