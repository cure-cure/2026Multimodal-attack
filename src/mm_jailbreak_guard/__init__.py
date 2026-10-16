"""D 模块：多证据融合与系统集成。"""

from .contracts import (
    DetectionRequest,
    EvidenceItem,
    FusionResult,
    ImageEvidence,
    TextEvidence,
)
from .fusion import FusionBundle, FusionEngine

__all__ = [
    "DetectionRequest",
    "EvidenceItem",
    "FusionBundle",
    "FusionEngine",
    "FusionResult",
    "ImageEvidence",
    "TextEvidence",
]

__version__ = "0.1.0"
