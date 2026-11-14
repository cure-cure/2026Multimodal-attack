# src/mm_jailbreak_guard/qwen3guard_lora_provider.py
"""C 模块：基于 LoRA 微调后的 Qwen3Guard 的 TextEvidenceProvider 实现。"""
from __future__ import annotations
import re
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .contracts import DetectionRequest, EvidenceItem, ImageEvidence, TextEvidence

_BASE_MODEL = Path(__file__).resolve().parents[2] / "models" / "Qwen3Guard-Gen-0.6B"
_LORA_DIR = Path(__file__).resolve().parents[2] / "models" / "qwen3guard-lora-adapter"

_SAFE_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)")
_SCORE_MAP = {"Safe": 0.0, "Controversial": 0.5, "Unsafe": 1.0}


class Qwen3GuardLoRATextProvider:
    name = "C Qwen3Guard-Gen-0.6B (LoRA)"
    version = "Qwen3Guard-Gen-0.6B-lora/v1"

    def __init__(self) -> None:
        if not _BASE_MODEL.is_dir() or not _LORA_DIR.is_dir():
            raise FileNotFoundError("缺少基座模型或 LoRA 适配器权重，请先完成训练。")

        self._tokenizer = AutoTokenizer.from_pretrained(str(_BASE_MODEL), trust_remote_code=True)
        self._tokenizer.pad_token = self._tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            str(_BASE_MODEL), dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self._model = PeftModel.from_pretrained(base, str(_LORA_DIR))
        self._model.eval()

    def _score(self, text: str) -> tuple[str, float]:
        text = text.strip()
        if not text:
            return "Safe", 0.0
        messages = [{"role": "user",
                     "content": f"请对以下文本进行安全审核，并判断其严重性级别（Safe/Controversial/Unsafe）。\n\n待审核文本：{text}"}]
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=64, do_sample=False, repetition_penalty=1.5)

        content = self._tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        m = _SAFE_PATTERN.search(content)
        label = m.group(1) if m else "Controversial"
        return label, _SCORE_MAP.get(label, 0.5)

    def analyze(self, request: DetectionRequest, image_evidence: ImageEvidence) -> TextEvidence:
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
                description=f"LoRA微调模型判定为 {user_label}",
                score=user_score,
            )
        ]
        if ocr_score is not None:
            evidence.append(
                EvidenceItem(
                    source="ocr_text",
                    description=f"LoRA微调模型判定 OCR 为 {ocr_label}",
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


def create_provider() -> Qwen3GuardLoRATextProvider:
    return Qwen3GuardLoRATextProvider()