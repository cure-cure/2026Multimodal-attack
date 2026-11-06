from __future__ import annotations
from pathlib import Path
import joblib

from .contracts import DetectionRequest, EvidenceItem, ImageEvidence, TextEvidence
from .rule_baseline import scan_with_rules
# 必须导入 tokenize，否则加载 joblib 模型时会报找不到该函数的错误
from .text_utils import tokenize

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "text_classifier_v1.joblib"

class RuleClassifierTextProvider:
    name = "C Rule + sklearn TF-IDF"
    version = "rule-baseline+tfidf-nb/v1"

    def __init__(self) -> None:
        if not _MODEL_PATH.is_file():
            raise FileNotFoundError(f"模型不存在：{_MODEL_PATH}。请先运行 python train_text_classifier.py")
        self._pipeline = joblib.load(_MODEL_PATH)

    def _score_text(self, text: str) -> tuple[float, list[EvidenceItem]]:
        text = text.strip()
        if not text:
            return 0.0, []

        rule_score, rule_hits = scan_with_rules(text)
        clf_proba = float(self._pipeline.predict_proba([text])[0][1])

        evidence = [
            EvidenceItem(
                source="rule_baseline",
                description=f"规则命中：{h.rule_name} → “{h.matched_text}”",
                score=h.score,
            ) for h in rule_hits
        ]
        evidence.append(
            EvidenceItem(
                source="text_classifier",
                description=f"自主 TF-IDF+NB 风险概率 {clf_proba:.3f}",
                score=clf_proba,
            )
        )
        return max(rule_score, clf_proba), evidence

    def analyze(self, request: DetectionRequest, image_evidence: ImageEvidence) -> TextEvidence:
        user_score, user_evidence = self._score_text(request.user_text)

        ocr_text = image_evidence.ocr_text if image_evidence.available else ""
        ocr_ready = (
                image_evidence.available
                and image_evidence.ocr_status == "success"
                and bool(ocr_text.strip())
        )

        ocr_score = None
        combined_score = None
        evidence = list(user_evidence)

        # 严格遵循接口规范：OCR 不可用时，相关分数保持 None
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