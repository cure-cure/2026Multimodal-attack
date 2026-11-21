"""Compare the live week-5 demo with the frozen cached expectations."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row.get("sample_id")
        if not sample_id or sample_id in rows:
            raise ValueError(f"Invalid/duplicate sample_id in {path}:{number}")
        rows[sample_id] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    expected, live = load(args.expected), load(args.live)
    if list(expected) != list(live):
        raise ValueError("Live demo IDs/order differ from frozen expectations")
    comparisons = []
    for sample_id in expected:
        frozen, actual = expected[sample_id], live[sample_id]
        delta = float(actual["statistic"]) - float(frozen["expected_statistic"])
        prediction_match = actual["prediction"] == frozen["expected_prediction"]
        if not math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=args.absolute_tolerance):
            raise ValueError(f"Statistic mismatch for {sample_id}: {delta}")
        if not prediction_match:
            raise ValueError(f"Prediction mismatch for {sample_id}")
        comparisons.append({
            "sample_id": sample_id, "category": frozen["demo_category"],
            "expected_statistic": frozen["expected_statistic"],
            "live_statistic": actual["statistic"], "absolute_delta": abs(delta),
            "prediction": actual["prediction"], "prediction_match": prediction_match,
            "elapsed_ms": actual.get("elapsed_ms"), "module_version": actual.get("module_version"),
        })
    report = {
        "status": "exact_match", "samples": len(comparisons),
        "max_absolute_statistic_delta": max(row["absolute_delta"] for row in comparisons),
        "prediction_matches": sum(row["prediction_match"] for row in comparisons),
        "elapsed_ms": {
            "min": min(row["elapsed_ms"] for row in comparisons),
            "max": max(row["elapsed_ms"] for row in comparisons),
            "mean": sum(row["elapsed_ms"] for row in comparisons) / len(comparisons),
        },
        "comparisons": comparisons,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
