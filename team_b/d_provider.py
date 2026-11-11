"""D-compatible online adapter backed by Task-B offline ImageEvidence JSONL."""
from __future__ import annotations

import json
import os
from pathlib import Path

from mm_jailbreak_guard.contracts import DetectionRequest, ImageEvidence


class CachedImageEvidenceProvider:
    name = "team_b_cached_evidence"
    version = "B-1.0.0"

    def __init__(self, evidence_path):
        self.evidence_path = Path(evidence_path).resolve()
        self.records = {}
        for number, line in enumerate(self.evidence_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            evidence = ImageEvidence.from_dict(json.loads(line))
            if evidence.sample_id in self.records:
                raise ValueError(f"Duplicate sample_id at line {number}: {evidence.sample_id}")
            self.records[evidence.sample_id] = evidence

    def analyze(self, request: DetectionRequest) -> ImageEvidence:
        return self.records.get(
            request.sample_id,
            ImageEvidence.unavailable(request.sample_id, "B evidence not found for sample_id"),
        )


def create_provider(evidence_path=None):
    path = evidence_path or os.environ.get("TEAM_B_EVIDENCE_PATH")
    if not path:
        raise ValueError("Set TEAM_B_EVIDENCE_PATH or pass evidence_path")
    return CachedImageEvidenceProvider(path)
