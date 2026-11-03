# src/mm_jailbreak_guard/qwen3guard_provider.py
"""C 模块：基于 Qwen3Guard-Gen-0.6B 的 TextEvidenceProvider 实现。"""
from __future__ import annotations

import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .contracts import DetectionRequest, EvidenceItem, ImageEvidence, TextEvidence

_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "Qwen3Guard-Gen-0.6B"
_SAFE_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)")
_SCORE_MAP = {"Safe": 0.0, "Controversial": 0.5, "Unsafe": 1.0}


class Qwen3GuardTextProvider:
    name = "C Qwen3Guard-Gen-0.6B"
    version = "Qwen3Guard-Gen-0.6B/transformers"

    def __init__(self) -> None:
        if not _MODEL_DIR.is_dir():
            raise FileNotFoundError(
                f"模型目录不存在：{_MODEL_DIR}。请先执行 hf download Qwen/Qwen3Guard-Gen-0.6B "
                f"--local-dir .\\models\\Qwen3Guard-Gen-0.6B（并设置 HF_HUB_DISABLE_XET=1）"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
        self._model = AutoModelForCausalLM.from_pretrained(
            str(_MODEL_DIR), torch_dtype="auto", device_map="auto"
        )
        self._model.eval()

    def _score(self, text: str) -> tuple[str, float]:
        text = text.strip()
        if not text:
            return "Safe", 0.0
        messages = [{"role": "user", "content": text}]
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=128, do_sample=False
            )
        content = self._tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        m = _SAFE_PATTERN.search(content)
        label = m.group(1) if m else "Controversial"
        return label, _SCORE_MAP.get(label, 0.5)

    def analyze(
        self, request: DetectionRequest, image_evidence: ImageEvidence
    ) -> TextEvidence:
        warnings: list[str] = []
        user_label, user_score = self._score(request.user_text)

        ocr_text = image_evidence.ocr_text if image_evidence.available else ""
        ocr_ready = (
            image_evidence.available
            and image_evidence.ocr_status == "success"
            and bool(ocr_text.strip())
        )
        ocr_label = ocr_score = combined_score = None
        if ocr_ready:
            ocr_label, ocr_score = self._score(ocr_text)
            combined_text = request.user_text.strip() + "\n[OCR]\n" + ocr_text.strip()
            _, combined_score = self._score(combined_text)
        else:
            warnings.append("OCR 不可用或未成功，OCR/组合分数设为 null")

        evidence = [
            EvidenceItem(
                source="user_text",
                description=f"Qwen3Guard 判定用户文字为 {user_label}",
                score=user_score,
            )
        ]
        if ocr_score is not None:
            evidence.append(
                EvidenceItem(
                    source="ocr_text",
                    description=f"Qwen3Guard 判定 OCR 文字为 {ocr_label}",
                    score=ocr_score,
                )
            )

        return TextEvidence(
            sample_id=request.sample_id,
            available=True,
            user_text_score=user_score,
            ocr_text_score=ocr_score,
            combined_text_score=combined_score,
            evidence=tuple(evidence),
            warnings=tuple(warnings),
            module_version=self.version,
        )


def create_provider() -> Qwen3GuardTextProvider:
    return Qwen3GuardTextProvider()