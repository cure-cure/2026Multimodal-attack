"""Build final D ImageEvidence while preserving partial failures and reasons."""
import argparse
import json
from pathlib import Path

from b_pipeline import VERSION, ZipDataset, load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-zip", required=True)
    parser.add_argument("--manifest-name", default="manifest.inputs.jsonl")
    parser.add_argument("--features", required=True)
    parser.add_argument("--ocr", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    features, texts = load_jsonl(Path(args.features)), load_jsonl(Path(args.ocr))
    dataset = ZipDataset(Path(args.dataset_zip), args.manifest_name)
    counts = {"samples": 0, "available": 0, "unavailable": 0}
    try:
        with output.open("x", encoding="utf-8") as stream:
            for source in dataset.rows():
                sid = source["sample_id"]
                feature, text = features.get(sid), texts.get(sid)
                feature_ok = bool(feature and feature.get("status") == "success")
                status = text.get("ocr_status", "unavailable") if text else "unavailable"
                ocr_ok = status in {"success", "no_text"}
                available = feature_ok or ocr_ok
                warnings = ["CIDER threshold is not calibrated; cider and visual_score are intentionally null."]
                if not feature:
                    warnings.append("Visual feature record is missing.")
                elif not feature_ok:
                    warnings.append(f"Visual feature extraction failed: {feature.get('error', 'unknown error')}")
                if not text:
                    warnings.append("OCR record is missing.")
                elif not ocr_ok:
                    warnings.append(f"OCR failed: {text.get('error', 'unknown error')}")
                record = {
                    "sample_id": sid, "available": available,
                    "ocr_text": text.get("ocr_text", "") if available and text else "",
                    "ocr_status": status if available else ("failed" if text else "unavailable"),
                    "visual_score": None,
                    "visual_features": feature.get("visual_features", {}) if feature_ok else {},
                    "regions": text.get("regions", []) if available and text else [],
                    "evidence": [], "warnings": warnings, "module_version": VERSION,
                    "cider": None,
                }
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                counts["samples"] += 1
                counts["available" if available else "unavailable"] += 1
    finally:
        dataset.close()
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
