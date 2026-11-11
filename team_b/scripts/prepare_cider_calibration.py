"""Build a traceable Task-B clean/adversarial calibration package.

The upstream CIDER repository uses ``combine`` mode: every selected HarmBench
``standard`` query is paired with every image in a split.  Labels come only
from the upstream filename convention documented by CIDER; detector outputs
are never used to infer truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


UPSTREAM_COMMIT = "3c08d7e2d52e27b47d1e428a030e8ffda5efae3e"
SOURCE_URL = "https://github.com/PandragonXIII/CIDER"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_id(*parts: str) -> str:
    value = "\0".join(parts).encode("utf-8")
    return "BC" + sha256_bytes(value)[:14].upper()


def read_standard_queries(path: Path) -> list[dict]:
    queries = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row_number, row in enumerate(csv.reader(stream), 1):
            if len(row) < 2:
                raise ValueError(f"Malformed CIDER text row: {path}:{row_number}")
            if row[1].strip() != "standard":
                continue
            text = row[0].strip()
            if not text:
                raise ValueError(f"Blank standard query: {path}:{row_number}")
            behavior_id = row[6].strip() if len(row) > 6 and row[6].strip() else f"row-{row_number}"
            queries.append({
                "text": text,
                "behavior_id": behavior_id,
                "source_row": row_number,
            })
    if not queries:
        raise ValueError(f"No standard queries found in {path}")
    return queries


def image_label(name: str) -> int:
    if name.startswith("clean_"):
        return 0
    if name.startswith("prompt_constrained_"):
        return 1
    raise ValueError(f"Unrecognized CIDER image label convention: {name}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_package(source_repo: Path, output_parent: Path, package_name: str) -> dict:
    source_repo = source_repo.resolve()
    source_data = source_repo / "data"
    required = [
        source_repo / "LICENSE",
        source_data / "img" / "valset",
        source_data / "img" / "testset",
        source_data / "img" / "clean",
        source_data / "text" / "valset.csv",
        source_data / "text" / "testset.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required upstream CIDER files: " + ", ".join(missing))

    root = output_parent.resolve() / package_name
    archive = output_parent.resolve() / f"{package_name}.zip"
    if root.exists() or archive.exists():
        raise FileExistsError(f"Refusing to overwrite {root} or {archive}")
    (root / "images").mkdir(parents=True)

    val_queries = read_standard_queries(source_data / "text" / "valset.csv")
    test_queries = read_standard_queries(source_data / "text" / "testset.csv")
    manifests: list[dict] = []
    inputs: list[dict] = []
    labels: list[dict] = []
    provenance: list[dict] = []
    copied: dict[str, dict] = {}

    def register_image(path: Path, upstream_split: str, forced_label: int | None = None) -> dict:
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        suffix = path.suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {path}")
        label = forced_label if forced_label is not None else image_label(path.name)
        relative = f"images/{digest[:20]}{suffix}"
        if digest not in copied:
            shutil.copy2(path, root / relative)
            copied[digest] = {
                "image_path": relative,
                "image_sha256": digest,
                "image_adv_label": label,
                "upstream_image": path.relative_to(source_repo).as_posix(),
                "upstream_split": upstream_split,
            }
        elif copied[digest]["image_adv_label"] != label:
            raise ValueError(f"Same image bytes receive conflicting labels: {path}")
        return copied[digest]

    def add_pairs(image_dir: Path, split: str, queries: list[dict], forced_label: int | None = None) -> None:
        image_paths = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            raise ValueError(f"No images in {image_dir}")
        for image_path in image_paths:
            image = register_image(image_path, split, forced_label)
            for query in queries:
                sid = stable_id(split, image["image_sha256"], query["behavior_id"], query["text"])
                manifest = {
                    "sample_id": sid,
                    "image_path": image["image_path"],
                    "user_text": query["text"],
                    "split": split,
                    "image_group_id": "CIDER_IMAGE_" + image["image_sha256"][:16],
                    "request_group_id": "CIDER_REQUEST_" + sha256_bytes(query["text"].encode("utf-8"))[:16],
                    "source_id": "SRC_CIDER_OFFICIAL",
                }
                manifests.append(manifest)
                inputs.append(dict(manifest))
                labels.append({
                    "sample_id": sid,
                    "image_adv_label": image["image_adv_label"],
                    "split": split,
                    "label_status": "verified_from_upstream_filename",
                    "attack_type": "prompt_constrained" if image["image_adv_label"] else "clean",
                })
                provenance.append({
                    "sample_id": sid,
                    "upstream_image": image["upstream_image"],
                    "upstream_text_file": f"data/text/{'testset' if split == 'test' else 'valset'}.csv",
                    "upstream_text_row": query["source_row"],
                    "behavior_id": query["behavior_id"],
                    "pair_mode": "combine",
                    "label_evidence": "CIDER documented clean_/prompt_constrained_ filename convention",
                    "source_commit": UPSTREAM_COMMIT,
                })

    # Upstream threshold selection combines ten clean images with valset queries.
    clean_paths = sorted(
        path for path in (source_data / "img" / "clean").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not clean_paths:
        raise ValueError("No threshold-selection clean images found")
    for path in clean_paths:
        image = register_image(path, "threshold_clean", forced_label=0)
        for query in val_queries:
            sid = stable_id("threshold_clean", image["image_sha256"], query["behavior_id"], query["text"])
            manifest = {
                "sample_id": sid,
                "image_path": image["image_path"],
                "user_text": query["text"],
                "split": "threshold_clean",
                "image_group_id": "CIDER_IMAGE_" + image["image_sha256"][:16],
                "request_group_id": "CIDER_REQUEST_" + sha256_bytes(query["text"].encode("utf-8"))[:16],
                "source_id": "SRC_CIDER_OFFICIAL",
            }
            manifests.append(manifest)
            inputs.append(dict(manifest))
            labels.append({
                "sample_id": sid, "image_adv_label": 0, "split": "threshold_clean",
                "label_status": "verified_from_upstream_directory", "attack_type": "clean",
            })
            provenance.append({
                "sample_id": sid,
                "upstream_image": image["upstream_image"],
                "upstream_text_file": "data/text/valset.csv",
                "upstream_text_row": query["source_row"],
                "behavior_id": query["behavior_id"], "pair_mode": "combine",
                "label_evidence": "CIDER official threshold-selection clean directory",
                "source_commit": UPSTREAM_COMMIT,
            })

    add_pairs(source_data / "img" / "valset", "validation", val_queries)
    add_pairs(source_data / "img" / "testset", "test", test_queries)
    manifests.sort(key=lambda row: (row["split"], row["sample_id"]))
    input_by_id = {row["sample_id"]: row for row in inputs}
    label_by_id = {row["sample_id"]: row for row in labels}
    provenance_by_id = {row["sample_id"]: row for row in provenance}
    inputs = [input_by_id[row["sample_id"]] for row in manifests]
    labels = [label_by_id[row["sample_id"]] for row in manifests]
    provenance = [provenance_by_id[row["sample_id"]] for row in manifests]

    write_jsonl(root / "manifest.inputs.jsonl", manifests)
    for split in ("threshold_clean", "validation", "test"):
        write_jsonl(root / f"manifest.{split}.jsonl", [row for row in manifests if row["split"] == split])
    input_fields = [
        "sample_id", "image_path", "user_text", "split", "image_group_id",
        "request_group_id", "source_id",
    ]
    write_csv(root / "inputs.csv", inputs, input_fields)
    write_csv(root / "splits.csv", [
        {"sample_id": row["sample_id"], "split": row["split"]} for row in manifests
    ], ["sample_id", "split"])
    write_csv(root / "labels_for_B.csv", labels, [
        "sample_id", "image_adv_label", "split", "label_status", "attack_type",
    ])
    write_csv(root / "provenance.csv", provenance, [
        "sample_id", "upstream_image", "upstream_text_file", "upstream_text_row",
        "behavior_id", "pair_mode", "label_evidence", "source_commit",
    ])
    write_csv(root / "sources.csv", [{
        "source_id": "SRC_CIDER_OFFICIAL", "source_name": "CIDER official repository data",
        "source_url": SOURCE_URL, "source_commit": UPSTREAM_COMMIT,
        "label_rule": "clean_=0; prompt_constrained_=1; data/img/clean=0",
    }], ["source_id", "source_name", "source_url", "source_commit", "label_rule"])
    shutil.copy2(source_repo / "LICENSE", root / "UPSTREAM_LICENSE")

    split_counts = {}
    label_counts = {}
    for row in labels:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        key = f"{row['split']}:{row['image_adv_label']}"
        label_counts[key] = label_counts.get(key, 0) + 1
    metadata = {
        "package_name": package_name,
        "source_url": SOURCE_URL,
        "source_commit": UPSTREAM_COMMIT,
        "pair_mode": "combine",
        "query_filter": "column 2 equals standard",
        "sample_count": len(manifests),
        "unique_image_count": len(copied),
        "standard_query_counts": {"validation": len(val_queries), "test": len(test_queries)},
        "split_counts": split_counts,
        "label_counts": label_counts,
        "excluded": [
            "data/img/adv4 (no paired clean images in the documented val/test protocol)",
            "non-standard HarmBench query rows",
        ],
        "usage": {
            "threshold_clean": "select clean passing-rate threshold only",
            "validation": "compare clean/adversarial distributions and select configuration",
            "test": "evaluate once after freezing configuration and threshold",
        },
    }
    (root / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "README.md").write_text(
        "# CIDER official Task-B calibration package\n\n"
        "This package reproduces CIDER's documented `combine` pairing protocol. "
        "Labels are derived only from official paths/names, never from detector scores.\n\n"
        "- `threshold_clean`: clean-only threshold selection.\n"
        "- `validation`: configuration/threshold analysis.\n"
        "- `test`: one-time evaluation after freezing.\n"
        "- Do not merge these labels with HoliSafe `overall_risk_label`; report the two axes separately.\n",
        encoding="utf-8",
    )

    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                bundle.write(path, f"{package_name}/{path.relative_to(root).as_posix()}")
    metadata["output_dir"] = str(root)
    metadata["output_zip"] = str(archive)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    parser.add_argument("--package-name", default="cider_calibration_v1")
    args = parser.parse_args()
    args.output_parent.mkdir(parents=True, exist_ok=True)
    result = build_package(args.source_repo, args.output_parent, args.package_name)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
