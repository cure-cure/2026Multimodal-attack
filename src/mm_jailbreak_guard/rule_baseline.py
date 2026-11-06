from __future__ import annotations

import re
from dataclasses import dataclass
from prompt_shield import PromptScanner

# 开源扫描器，覆盖通用注入
_SCANNER = PromptScanner(threshold="LOW")

# 自定义中文正则，补充越狱和绕过表达
_CUSTOM_PATTERNS: list[tuple[str, str]] = [
    (r"忽略(之前|以上|前面|所有)的?(指令|指示|规则|要求)", "覆盖原指令"),
    (r"(无视|跳过|绕过|解除)(安全|内容|道德|系统)?(限制|约束|过滤|审查|防护)", "绕过约束"),
    (r"(开发者|上帝|超级管理员|DAN)模式", "角色越权"),
    (r"(不要|不准|禁止)(拒绝|说不能|警告|提醒|道歉)", "抑制拒答"),
    (r"(假装|扮演|模拟).{0,15}(没有|不受|无需).{0,10}(限制|约束|规则)", "角色扮演绕过"),
    (r"(输出|显示|打印|告诉我|重复).{0,15}(系统提示|system prompt|初始指令|上面的内容)", "套取系统提示"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), name) for p, name in _CUSTOM_PATTERNS]


@dataclass(slots=True, frozen=True)
class RuleHit:
    rule_name: str
    matched_text: str
    score: float


def scan_with_rules(text: str) -> tuple[float, list[RuleHit]]:
    text = text.strip()
    if not text:
        return 0.0, []

    hits: list[RuleHit] = []

    try:
        result = _SCANNER.scan(text)
        severity = (getattr(result, "severity", "") or "").upper()
        if severity in {"HIGH", "CRITICAL"}:
            hits.append(RuleHit("prompt_shield", text[:120], 0.9))
        elif severity == "MEDIUM":
            hits.append(RuleHit("prompt_shield", text[:120], 0.6))
    except Exception:
        pass

    for pattern, name in _COMPILED:
        match = pattern.search(text)
        if match:
            hits.append(RuleHit(name, match.group(0), 0.85))

    if not hits:
        return 0.0, []

    score = min(1.0, max(h.score for h in hits) + 0.05 * (len(hits) - 1))
    return score, hits