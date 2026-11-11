"""Filter low-confidence EasyOCR regions before handing text to C/D."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.20)
    args = parser.parse_args()
    source, destination = Path(args.input), Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("min-confidence must be in [0,1]")
    counts = {"records": 0, "success": 0, "no_text": 0, "failed": 0,
              "raw_regions": 0, "kept_regions": 0}
    with destination.open("x", encoding="utf-8") as stream:
        for line in source.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            raw = list(row.get("regions", []))
            kept = [region for region in raw
                    if float(region.get("confidence", 0.0)) >= args.min_confidence
                    and str(region.get("text", "")).strip()]
            if row.get("ocr_status") != "failed":
                row["regions"] = kept
                row["ocr_text"] = " ".join(region["text"].strip() for region in kept)
                row["ocr_status"] = "success" if row["ocr_text"] else "no_text"
            row["raw_region_count"] = len(raw)
            row["min_confidence"] = args.min_confidence
            counts["records"] += 1
            counts["raw_regions"] += len(raw)
            counts["kept_regions"] += len(kept)
            counts[row["ocr_status"]] += 1
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
