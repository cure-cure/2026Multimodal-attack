from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path, *, unique_key: str | None = None) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number} 不是合法 JSON：{exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_number} 必须是 JSON 对象")
            if unique_key:
                if unique_key not in record:
                    raise ValueError(f"{source}:{line_number} 缺少 {unique_key}")
                value = str(record[unique_key])
                if value in seen:
                    raise ValueError(f"{source}:{line_number} 出现重复 {unique_key}={value}")
                seen.add(value)
            records.append(record)
    return records


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    _atomic_text_write(target, payload)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n"
        for record in records
    )
    _atomic_text_write(target, payload)


def _atomic_text_write(target: Path, payload: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=target.parent
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, target)


class JsonlAuditLog:
    """线程安全的检测记录；默认不保存原始输入文字和图片。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        entry = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            records = load_jsonl(self.path)
        return list(reversed(records[-max(1, min(limit, 100)) :]))
