"""Validate a week-5 live CIDER batch without changing its output."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def rows(path: Path) -> list[dict]:
    result, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row.get("sample_id")
        if not sample_id or sample_id in seen:
            raise ValueError(f"Invalid/duplicate sample_id in {path}:{number}")
        seen.add(sample_id)
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    source, output = rows(args.manifest), rows(args.output)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if [row["sample_id"] for row in source] != [row["sample_id"] for row in output]:
        raise ValueError("Output IDs/order differ from the manifest")
    threshold = float(config["threshold"])
    failures = 0
    for row in output:
        if row.get("status") != "success":
            failures += 1
            continue
        similarities = row.get("similarities", [])
        if len(similarities) != len(config["stages"]) + 1 or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                for value in similarities):
            raise ValueError(f"Invalid similarities: {row['sample_id']}")
        if row.get("threshold") != threshold or row.get("comparison") != "gt":
            raise ValueError(f"Threshold/config mismatch: {row['sample_id']}")
        if row.get("prediction") != int(float(row["statistic"]) > threshold):
            raise ValueError(f"Strict comparison mismatch: {row['sample_id']}")
    summary = {"status": "valid" if not failures else "partial", "samples": len(output),
               "successes": len(output) - failures, "failures": failures,
               "threshold": threshold, "config_id": config["config_id"]}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
