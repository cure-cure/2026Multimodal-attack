# src/mm_jailbreak_guard/rule_classifier_provider.py
from __future__ import annotations
from pathlib import Path
import os
import joblib

from .contracts import DetectionRequest, EvidenceItem, ImageEvidence, TextEvidence
from .rule_baseline import scan_with_rules
from .text_utils import tokenize

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "text_classifier_v2.joblib"


class RuleClassifierTextProvider:
    name = "C Rule + sklearn TF-IDF"
    version = "rule-baseline+tfidf-nb/v2"

    def __init__(self) -> None:
        # 读取环境变量，决定输出模式：combined (默认) / pure_rule / pure_classifier
        self.mode = os.environ.get("C_PROVIDER_MODE", "combined").lower()

        if self.mode in ("combined", "pure_classifier"):
            if not _MODEL_PATH.is_file():
                raise FileNotFoundError(f"模型不存在：{_MODEL_PATH}")
            self._pipeline = joblib.load(_MODEL_PATH)

        if self.mode == "pure_rule":
            self.version = "rule-baseline/v1"
        elif self.mode == "pure_classifier":
            self.version = "tfidf-nb/v2"

    def _score_text(self, text: str) -> tuple[float, list[EvidenceItem]]:
        text = text.strip()
        if not text:
            return 0.0, []

        evidence = []

        # 纯分类器模式
        if self.mode == "pure_classifier":
            clf_proba = float(self._pipeline.predict_proba([text])[0][1])
            evidence.append(
                EvidenceItem(source="text_classifier", description=f"TF-IDF 概率 {clf_proba:.3f}", score=clf_proba))
            return clf_proba, evidence

        # 纯规则模式
        if self.mode == "pure_rule":
            rule_score, rule_hits = scan_with_rules(text)
            for h in rule_hits:
                evidence.append(
                    EvidenceItem(source="rule_baseline", description=f"规则命中：{h.rule_name}", score=h.score))
            return rule_score, evidence

        # 混合模式（默认）
        rule_score, rule_hits = scan_with_rules(text)
        clf_proba = float(self._pipeline.predict_proba([text])[0][1])
        for h in rule_hits:
            evidence.append(EvidenceItem(source="rule_baseline", description=f"规则命中：{h.rule_name}", score=h.score))
        evidence.append(
            EvidenceItem(source="text_classifier", description=f"TF-IDF 概率 {clf_proba:.3f}", score=clf_proba))
        return max(rule_score, clf_proba), evidence

    def analyze(self, request: DetectionRequest, image_evidence: ImageEvidence) -> TextEvidence:
        user_score, user_evidence = self._score_text(request.user_text)

        ocr_text = image_evidence.ocr_text if image_evidence.available else ""
        ocr_ready = (image_evidence.available and image_evidence.ocr_status == "success" and bool(ocr_text.strip()))

        ocr_score = None
        combined_score = None
        evidence = list(user_evidence)

        if ocr_ready:
            ocr_score, ocr_evidence = self._score_text(ocr_text)
            evidence.extend(ocr_evidence)
            combined = request.user_text.strip() + "\n[OCR]\n" + ocr_text.strip()
            combined_score, combined_evidence = self._score_text(combined)
            evidence.extend(combined_evidence)

        return TextEvidence(
            sample_id=request.sample_id,
            available=True,
            user_text_score=user_score,
            ocr_text_score=ocr_score,
            combined_text_score=combined_score,
            evidence=tuple(evidence),
            warnings=() if ocr_ready else ("OCR 不可用或未成功，OCR/组合分数设为 null",),
            module_version=self.version,
        )


def create_provider() -> RuleClassifierTextProvider:
    return RuleClassifierTextProvider()