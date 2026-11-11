"""Task-B data audit, OCR, handcrafted features, and D evidence pipeline."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import time
import zipfile

import numpy as np
from PIL import Image, ImageOps

VERSION = "B-1.0.0"


def json_line(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"


def gaussian_kernel(size=5, sigma=1.0):
    """Construct a normalized 2-D Gaussian kernel without a filter library."""
    if type(size) is not int or size < 3 or size % 2 == 0:
        raise ValueError("Gaussian kernel size must be an odd integer >= 3")
    if not isinstance(sigma, (int, float)) or sigma <= 0 or not math.isfinite(sigma):
        raise ValueError("Gaussian sigma must be finite and positive")
    axis = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * float(sigma) ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


def convolve_reflect(image, kernel):
    """Apply a 2-D kernel with reflect padding using NumPy window algebra."""
    if image.ndim not in (2, 3):
        raise ValueError("Expected HxW or HxWxC image")
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 == 0:
        raise ValueError("Expected a square odd-sized kernel")
    radius = kernel.shape[0] // 2
    working = image.astype(np.float32, copy=False)
    if working.ndim == 2:
        padded = np.pad(working, ((radius, radius), (radius, radius)), mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
        return np.einsum("ijxy,xy->ij", windows, kernel, optimize=True)
    padded = np.pad(working, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape, axis=(0, 1))
    return np.einsum("ijcxy,xy->ijc", windows, kernel, optimize=True)


def handcrafted_features(image, analysis_size=256, kernel_size=5, sigma=1.0,
                         high_frequency_cutoff=0.25):
    """Compute bounded auxiliary features; these are not calibrated attack scores."""
    if analysis_size < 32 or not 0 < high_frequency_cutoff < 1:
        raise ValueError("Invalid analysis parameters")
    rgb_image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = rgb_image.size
    resized = rgb_image.resize((analysis_size, analysis_size), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32)
    blurred = convolve_reflect(rgb, gaussian_kernel(kernel_size, sigma))
    residual = rgb - blurred
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
    power = np.abs(np.fft.fftshift(np.fft.fft2(gray - float(gray.mean())))) ** 2
    yy, xx = np.indices(power.shape, dtype=np.float32)
    center = (analysis_size - 1) / 2.0
    radius = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    radius /= max(float(radius.max()), 1.0)
    total = float(power.sum())
    high_ratio = centroid = 0.0
    if total > np.finfo(np.float64).eps:
        high_ratio = float(power[radius >= high_frequency_cutoff].sum() / total)
        centroid = float((power * radius).sum() / total)
    values = {
        "gaussian_residual_mae": float(np.mean(np.abs(residual)) / 255.0),
        "gaussian_residual_rmse": float(np.sqrt(np.mean(residual * residual)) / 255.0),
        "high_frequency_ratio": high_ratio,
        "spectral_centroid": centroid,
    }
    values = {key: min(max(value, 0.0), 1.0) for key, value in values.items()}
    return {
        "visual_features": values,
        "feature_parameters": {
            "analysis_size": analysis_size, "gaussian_kernel_size": kernel_size,
            "gaussian_sigma": float(sigma),
            "fft_high_frequency_cutoff": float(high_frequency_cutoff),
            "padding": "reflect", "resize": "Pillow bilinear",
        },
        "image_width": width, "image_height": height,
    }


@dataclass
class ZipDataset:
    path: Path
    manifest_name: str = "manifest.inputs.jsonl"

    def __post_init__(self):
        self.path = self.path.resolve()
        self.archive = zipfile.ZipFile(self.path)
        matches = [name for name in self.archive.namelist() if name.endswith("/" + self.manifest_name)]
        if len(matches) != 1:
            raise ValueError(f"Expected one {self.manifest_name}, found {len(matches)}")
        self.manifest_member = matches[0]
        self.root = str(PurePosixPath(self.manifest_member).parent)
        self.members = set(self.archive.namelist())

    def close(self):
        self.archive.close()

    def read_text(self, basename):
        return self.archive.read(f"{self.root}/{basename}").decode("utf-8-sig")

    def rows(self):
        result, seen = [], set()
        for number, line in enumerate(self.read_text(self.manifest_name).splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("sample_id")
            if not isinstance(sid, str) or not sid or sid in seen:
                raise ValueError(f"Invalid/duplicate sample_id at line {number}")
            seen.add(sid)
            if not isinstance(row.get("image_path"), str) or not row["image_path"]:
                raise ValueError(f"Invalid image_path: {sid}")
            if not isinstance(row.get("user_text"), str):
                raise ValueError(f"Invalid user_text: {sid}")
            result.append(row)
        return result

    def image_member(self, row):
        path = PurePosixPath(row["image_path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe image path: {path}")
        return f"{self.root}/{path.as_posix()}"

    def image_bytes(self, row):
        return self.archive.read(self.image_member(row))


def csv_rows(text):
    return list(csv.DictReader(StringIO(text)))


def audit_dataset(dataset):
    rows = dataset.rows()
    paths = [row["image_path"] for row in rows]
    inputs = csv_rows(dataset.read_text("inputs.csv"))
    labels = csv_rows(dataset.read_text("labels_for_B.csv"))
    populated = [row for row in labels if str(row.get("image_adv_label", "")).strip()]
    ids = {row["sample_id"] for row in rows}
    input_ids = {row.get("sample_id") for row in inputs}
    missing = [row["sample_id"] for row in rows if dataset.image_member(row) not in dataset.members]
    return {
        "audit_version": VERSION, "dataset_zip": str(dataset.path),
        "manifest_member": dataset.manifest_member, "sample_count": len(rows),
        "unique_sample_ids": len(ids), "unique_image_paths": len(set(paths)),
        "split_counts": dict(sorted(Counter(row.get("split") for row in rows).items())),
        "missing_image_count": len(missing), "missing_image_sample_ids": missing,
        "blank_user_text_count": sum(not row["user_text"].strip() for row in rows),
        "replacement_character_count": sum("\ufffd" in row["user_text"] for row in rows),
        "inputs_csv_count": len(inputs), "inputs_csv_id_match": input_ids == ids,
        "manifest_has_group_fields": all(row.get("image_group_id") and row.get("request_group_id") for row in rows),
        "inputs_csv_has_group_fields": all(row.get("image_group_id") and row.get("request_group_id") for row in inputs),
        "b_label_row_count": len(labels), "b_label_populated_count": len(populated),
        "image_adversarial_threshold_calibration_allowed": bool(populated),
    }


def adapt_manifest(dataset, output):
    rows = dataset.rows()
    inputs = {row["sample_id"]: row for row in csv_rows(dataset.read_text("inputs.csv"))}
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    missing = []
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            adapted, source = dict(row), inputs.get(row["sample_id"], {})
            for key in ("image_group_id", "request_group_id"):
                if source.get(key):
                    adapted[key] = source[key]
                else:
                    missing.append([row["sample_id"], key])
            stream.write(json_line(adapted))
    return {"rows": len(rows), "missing_group_fields": missing, "output": str(output)}


def selected(rows, limit):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    return rows if limit is None else rows[:limit]


def run_features(dataset, output, limit=None):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, cache = selected(dataset.rows(), limit), {}
    successes = failures = 0
    started = time.perf_counter()
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            try:
                raw = dataset.image_bytes(row)
                digest = hashlib.sha256(raw).hexdigest()
                if digest not in cache:
                    with Image.open(BytesIO(raw)) as image:
                        item = handcrafted_features(image)
                    item.update(image_sha256=digest, status="success")
                    cache[digest] = item
                record = {"sample_id": row["sample_id"], **cache[digest], "module_version": VERSION}
                successes += 1
            except Exception as exc:
                failures += 1
                record = {"sample_id": row["sample_id"], "status": "failed",
                          "error": f"{type(exc).__name__}: {exc}", "visual_features": {},
                          "module_version": VERSION}
            stream.write(json_line(record))
    return {"samples": len(rows), "unique_images_processed": len(cache), "successes": successes,
            "failures": failures, "elapsed_seconds": time.perf_counter() - started,
            "output": str(output)}


def ocr_reader(cpu):
    import easyocr
    return easyocr.Reader(["ch_sim", "en"], gpu=not cpu, verbose=True)


def ocr_image(reader, raw):
    with Image.open(BytesIO(raw)) as source:
        image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    regions = []
    for box, text, confidence in reader.readtext(image, detail=1, paragraph=False):
        points = np.asarray(box, dtype=np.float64)
        regions.append({"text": str(text), "confidence": min(max(float(confidence), 0.0), 1.0),
                        "box": [float(points[:, 0].min()), float(points[:, 1].min()),
                                float(points[:, 0].max()), float(points[:, 1].max())]})
    text = " ".join(region["text"].strip() for region in regions if region["text"].strip())
    return {"ocr_text": text, "ocr_status": "success" if text else "no_text", "regions": regions}


def run_ocr(dataset, output, cache_dir, limit=None, cpu=False):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows, reader = selected(dataset.rows(), limit), ocr_reader(cpu)
    counts = Counter()
    started = time.perf_counter()
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            try:
                raw = dataset.image_bytes(row)
                digest = hashlib.sha256(raw).hexdigest()
                cache_path = cache_dir / f"{digest}.json"
                if cache_path.exists():
                    item = json.loads(cache_path.read_text(encoding="utf-8"))
                    counts["cache_hits"] += 1
                else:
                    item = ocr_image(reader, raw)
                    item.update(image_sha256=digest, engine="EasyOCR", engine_version="1.7.2",
                                languages=["ch_sim", "en"])
                    cache_path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
                record = {"sample_id": row["sample_id"], **item, "module_version": VERSION}
                counts[item["ocr_status"]] += 1
            except Exception as exc:
                counts["failed"] += 1
                record = {"sample_id": row["sample_id"], "ocr_text": "", "ocr_status": "failed",
                          "regions": [], "error": f"{type(exc).__name__}: {exc}",
                          "module_version": VERSION}
            stream.write(json_line(record))
    return {"samples": len(rows), **counts, "elapsed_seconds": time.perf_counter() - started,
            "output": str(output), "cache_dir": str(cache_dir)}


def load_jsonl(path):
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("sample_id") or row["sample_id"] in result:
            raise ValueError(f"Invalid/duplicate sample_id in {path}:{number}")
        result[row["sample_id"]] = row
    return result


def build_evidence(rows, feature_path, ocr_path, output):
    if output.exists():
        raise FileExistsError(output)
    features, texts = load_jsonl(feature_path), load_jsonl(ocr_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with output.open("x", encoding="utf-8") as stream:
        for source in rows:
            sid = source["sample_id"]
            feature, text = features.get(sid), texts.get(sid)
            feature_ok = bool(feature and feature.get("status") == "success")
            status = text.get("ocr_status", "unavailable") if text else "unavailable"
            ocr_ok = status in {"success", "no_text"}
            available = feature_ok or ocr_ok
            counts["available" if available else "unavailable"] += 1
            warnings = ["CIDER threshold is not calibrated; cider and visual_score are intentionally null."]
            record = {
                "sample_id": sid, "available": available,
                "ocr_text": text.get("ocr_text", "") if available and text else "",
                "ocr_status": status if available else ("failed" if text else "unavailable"),
                "visual_score": None,
                "visual_features": feature.get("visual_features", {}) if feature_ok else {},
                "regions": text.get("regions", []) if available and text else [],
                "evidence": [], "warnings": warnings, "module_version": VERSION, "cider": None,
            }
            stream.write(json_line(record))
    return {"samples": len(rows), **counts, "output": str(output)}


def write_audit_chart(audit, output_dir):
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "actual_split_counts.png"
    labels, values = list(audit["split_counts"]), list(audit["split_counts"].values())
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#3563a9", "#5e91cf", "#e29d45", "#7d5ba6"])
    ax.set(title="Actual sample counts by fixed split", ylabel="Samples")
    ax.grid(axis="y", alpha=0.2)
    ax.bar_label(bars, padding=3)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description="Task-B OCR, visual feature, and evidence pipeline")
    parser.add_argument("command", choices=["audit", "adapt-manifest", "features", "ocr", "evidence"])
    parser.add_argument("--dataset-zip", required=True)
    parser.add_argument("--manifest-name", default="manifest.inputs.jsonl")
    parser.add_argument("--output")
    parser.add_argument("--output-dir")
    parser.add_argument("--feature-file")
    parser.add_argument("--ocr-file")
    parser.add_argument("--cache-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    dataset = ZipDataset(Path(args.dataset_zip), args.manifest_name)
    try:
        if args.command == "audit":
            if not args.output_dir:
                parser.error("audit requires --output-dir")
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            report = audit_dataset(dataset)
            report["chart"] = write_audit_chart(report, output_dir)
            (output_dir / "dataset_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "adapt-manifest":
            report = adapt_manifest(dataset, Path(args.output)) if args.output else parser.error("--output required")
        elif args.command == "features":
            report = run_features(dataset, Path(args.output), args.limit) if args.output else parser.error("--output required")
        elif args.command == "ocr":
            if not args.output or not args.cache_dir:
                parser.error("ocr requires --output and --cache-dir")
            report = run_ocr(dataset, Path(args.output), Path(args.cache_dir), args.limit, args.cpu)
        else:
            if not args.output or not args.feature_file or not args.ocr_file:
                parser.error("evidence requires --output, --feature-file, and --ocr-file")
            report = build_evidence(selected(dataset.rows(), args.limit), Path(args.feature_file),
                                    Path(args.ocr_file), Path(args.output))
        report.update(python=platform.python_version(), numpy=np.__version__,
                      pillow=Image.__version__, pipeline_version=VERSION)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
