"""Build D-compatible ImageEvidence for the labeled CIDER calibration package."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_jsonl(path):
    rows = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("sample_id")
        if not sid or sid in rows:
            raise ValueError(f"Invalid/duplicate sample_id in {path}:{number}")
        rows[sid] = row
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "data/cider_calibration_v1/manifest.inputs.jsonl")
    parser.add_argument("--features", type=Path, default=root / "artifacts/features.cider_calibration_v1.jsonl")
    parser.add_argument("--ocr", type=Path, default=root / "artifacts/ocr.cider_calibration_v1.jsonl")
    parser.add_argument("--cider-raw", type=Path, default=root / "artifacts/cider.cider_calibration_v1.raw.jsonl")
    parser.add_argument("--cider-scored", type=Path, default=root / "artifacts/cider.cider_calibration_v1.scored.jsonl")
    parser.add_argument("--output", type=Path, default=root / "artifacts/image_evidence.cider_calibration_v1.jsonl")
    parser.add_argument("--include-splits", nargs="+", choices=["validation", "test"],
                        default=["validation"],
                        help="Build one D-valid split at a time; official adversarial images overlap across splits")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = load_jsonl(args.manifest)
    features, ocr = load_jsonl(args.features), load_jsonl(args.ocr)
    raw, scored = load_jsonl(args.cider_raw), load_jsonl(args.cider_scored)
    all_ids = set(manifest)
    if not (all_ids == set(features) == set(ocr) == set(raw) == set(scored)):
        raise ValueError("Input artifacts do not have identical sample IDs")
    ids = {sid for sid, row in manifest.items() if row["split"] in args.include_splits}
    if not ids:
        raise ValueError("No records selected by --include-splits")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"samples": 0, "predicted_adversarial": 0, "ocr_success": 0, "ocr_no_text": 0}
    with args.output.open("x", encoding="utf-8") as stream:
        for sid in manifest:
            if sid not in ids:
                continue
            feature, text, cider, decision = features[sid], ocr[sid], raw[sid], scored[sid]
            if feature.get("status") != "success" or cider.get("status") != "success":
                raise ValueError(f"Required B branch failed: {sid}")
            strict_threshold = float(decision["threshold"])
            # D v2 accepts ge/le only. The next representable float makes
            # statistic >= nextafter(tau,+inf) exactly encode statistic > tau.
            d_threshold = math.nextafter(strict_threshold, math.inf)
            prediction = int(decision["prediction"])
            relation = "exceeded" if prediction else "did not exceed"
            record = {
                "sample_id": sid,
                "available": True,
                "ocr_text": text.get("ocr_text", ""),
                "ocr_status": text.get("ocr_status", "unavailable"),
                "visual_score": None,
                "visual_features": feature["visual_features"],
                "regions": text.get("regions", []),
                "evidence": [{
                    "source": "cider",
                    "description": (
                        f"CIDER statistic {cider['statistic']:.8f} {relation} the frozen "
                        f"strict threshold {strict_threshold:.8f}; prediction={prediction}."
                    ),
                }],
                "warnings": [
                    "visual_score remains null because the raw statistic is not a calibrated probability.",
                    "D ge comparison encodes B strict gt using nextafter(threshold, +inf).",
                    "Threshold was selected on CIDER official threshold_clean data, not HoliSafe labels.",
                ],
                "module_version": "B-1.1.0-cider-calibrated",
                "cider": {
                    "similarities": cider["similarities"],
                    "statistic": cider["statistic"],
                    "statistic_name": cider["statistic_name"],
                    "threshold": d_threshold,
                    "comparison": "ge",
                    "config_id": cider["config_id"] + ":tau95-clean",
                    "reproduction": cider["reproduction"],
                },
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            counts["samples"] += 1
            counts["predicted_adversarial"] += prediction
            key = "ocr_success" if record["ocr_status"] == "success" else "ocr_no_text"
            counts[key] = counts.get(key, 0) + 1
    print(json.dumps({**counts, "included_splits": args.include_splits, "output": str(args.output),
                      "strict_threshold": strict_threshold, "d_ge_threshold": d_threshold},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
