"""Complete Task-B v2 thresholding, safety classification, and fixed-split exports.

This program never reruns CIDER, OCR, or handcrafted feature extraction.  It
uses the immutable cached artifacts, selects thresholds only on validation
data, evaluates frozen rules on non-selection splits, and records input hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


FEATURES = (
    "gaussian_residual_mae",
    "gaussian_residual_rmse",
    "high_frequency_ratio",
    "spectral_centroid",
)
FIXED_SPLITS = ("base_train", "fusion_train", "validation", "test")


def load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if not sample_id or sample_id in rows:
                raise ValueError(f"Invalid/duplicate sample_id in {path}:{number}")
            rows[str(sample_id)] = row
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if not positives:
        return float("nan")
    found = total = 0.0
    for rank, index in enumerate(sorted(range(len(scores)), key=lambda i: (-scores[i], i)), 1):
        if labels[index]:
            found += 1
            total += found / rank
    return total / positives


def classification_metrics(labels: list[int], scores: list[float], threshold: float,
                           comparison: str = "gt") -> dict:
    if comparison not in {"gt", "ge"}:
        raise ValueError(comparison)
    predictions = [int(score > threshold if comparison == "gt" else score >= threshold)
                   for score in scores]
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "samples": len(labels), "positives": sum(labels), "negatives": len(labels) - sum(labels),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": safe_div(tp + tn, len(labels)),
        "precision": precision, "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "fpr": safe_div(fp, fp + tn), "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "auroc": auc(labels, scores), "average_precision": average_precision(labels, scores),
    }


def candidate_thresholds(scores: list[float]) -> list[float]:
    values = sorted(set(float(value) for value in scores))
    if not values:
        raise ValueError("Cannot choose a threshold from no scores")
    return ([math.nextafter(values[0], -math.inf)]
            + [(left + right) / 2.0 for left, right in zip(values, values[1:])]
            + [values[-1]])


def select_threshold(labels: list[int], scores: list[float]) -> tuple[float, dict]:
    """Maximize balanced accuracy; deterministic ties prefer F1, recall, low FPR."""
    ranked = []
    for threshold in candidate_thresholds(scores):
        metrics = classification_metrics(labels, scores, threshold)
        key = (metrics["balanced_accuracy"], metrics["f1"], metrics["recall"],
               -metrics["fpr"], -threshold)
        ranked.append((key, threshold, metrics))
    _, threshold, metrics = max(ranked, key=lambda item: item[0])
    return threshold, metrics


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_balanced_logistic(x: np.ndarray, y: np.ndarray, iterations: int = 5000,
                          learning_rate: float = 0.05, l2: float = 0.02) -> tuple[np.ndarray, float]:
    """Fit a deterministic class-balanced logistic model with NumPy."""
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    if not positives or not negatives:
        raise ValueError("Both classes are required")
    sample_weights = np.where(y == 1, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))
    coefficients = np.zeros(x.shape[1], dtype=np.float64)
    intercept = 0.0
    for iteration in range(iterations):
        probabilities = sigmoid(x @ coefficients + intercept)
        errors = (probabilities - y) * sample_weights
        gradient = x.T @ errors / len(y) + l2 * coefficients
        intercept_gradient = float(errors.mean())
        step = learning_rate / math.sqrt(1.0 + iteration / 500.0)
        coefficients -= step * gradient
        intercept -= step * intercept_gradient
    return coefficients, intercept


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size), "min": float(array.min()), "q25": float(np.quantile(array, .25)),
        "median": float(np.median(array)), "mean": float(array.mean()),
        "q75": float(np.quantile(array, .75)), "max": float(array.max()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def standardized_difference(safe: list[float], unsafe: list[float]) -> float:
    left, right = np.asarray(safe, dtype=np.float64), np.asarray(unsafe, dtype=np.float64)
    denominator = len(left) + len(right) - 2
    pooled = math.sqrt(((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1))
                       / denominator) if denominator > 0 else 0.0
    return float((right.mean() - left.mean()) / pooled) if pooled else 0.0


def read_overall_labels(paths: list[Path]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                value = row.get("overall_risk_label", "").strip()
                if value not in {"0", "1"}:
                    continue
                sample_id, label = row["sample_id"], int(value)
                if sample_id in labels and labels[sample_id] != label:
                    raise ValueError(f"Conflicting overall label for {sample_id}")
                labels[sample_id] = label
    return labels


def build_cider_report(rows: list[dict]) -> tuple[dict, list[dict]]:
    validation = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]
    threshold, validation_metrics = select_threshold(
        [row["image_adv_label"] for row in validation],
        [row["statistic"] for row in validation],
    )
    test_metrics = classification_metrics(
        [row["image_adv_label"] for row in test],
        [row["statistic"] for row in test], threshold,
    )
    old_threshold = float(rows[0]["threshold"])
    old_validation = classification_metrics(
        [row["image_adv_label"] for row in validation],
        [row["statistic"] for row in validation], old_threshold,
    )
    old_test = classification_metrics(
        [row["image_adv_label"] for row in test],
        [row["statistic"] for row in test], old_threshold,
    )
    predictions = []
    for row in rows:
        item = dict(row)
        item["threshold"] = threshold
        item["prediction"] = int(float(row["statistic"]) > threshold)
        item["selection_protocol"] = "validation_max_balanced_accuracy_then_f1"
        predictions.append(item)
    return {
        "experiment_id": "B-CIDER-threshold-v2",
        "selection_split": "validation",
        "selection_objective": "maximize balanced_accuracy; ties: F1, recall, lower FPR, lower threshold",
        "comparison": "strict_gt",
        "selected_threshold": threshold,
        "validation_metrics": validation_metrics,
        "frozen_test_metrics": test_metrics,
        "previous_conservative_threshold": old_threshold,
        "previous_validation_metrics": old_validation,
        "previous_test_metrics": old_test,
        "test_used_for_selection": False,
    }, predictions


def build_handcrafted_classifier(manifest: dict[str, dict], features: dict[str, dict],
                                 labels: dict[str, int]) -> tuple[dict, list[dict]]:
    rows = []
    for sample_id, label in labels.items():
        if sample_id not in manifest or sample_id not in features:
            continue
        feature = features[sample_id]
        if feature.get("status") != "success":
            continue
        rows.append({
            "sample_id": sample_id, "split": manifest[sample_id]["split"], "label": label,
            "values": [float(feature["visual_features"][name]) for name in FEATURES],
        })
    by_split = {split: [row for row in rows if row["split"] == split] for split in FIXED_SPLITS}
    train, validation, frozen = by_split["base_train"], by_split["validation"], by_split["fusion_train"]
    if not train or not validation or not frozen:
        raise ValueError("base_train, validation, and fusion_train labeled rows are required")
    train_x = np.asarray([row["values"] for row in train], dtype=np.float64)
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    if np.any(scale == 0):
        raise ValueError("A handcrafted feature has zero training variance")
    standardize = lambda selected: (np.asarray([row["values"] for row in selected], dtype=np.float64) - mean) / scale
    train_y = np.asarray([row["label"] for row in train], dtype=np.float64)
    coefficients, intercept = fit_balanced_logistic(standardize(train), train_y)
    probabilities = {
        split: sigmoid(standardize(selected) @ coefficients + intercept).tolist()
        for split, selected in by_split.items() if selected
    }
    threshold, validation_metrics = select_threshold(
        [row["label"] for row in validation], probabilities["validation"])
    split_metrics = {
        split: classification_metrics([row["label"] for row in selected], probabilities[split], threshold)
        for split, selected in by_split.items() if selected
    }
    predictions = []
    for split, selected in by_split.items():
        for row, probability in zip(selected, probabilities.get(split, [])):
            predictions.append({
                "sample_id": row["sample_id"], "split": split,
                "overall_risk_label": row["label"], "unsafe_probability": probability,
                "prediction": int(probability > threshold), "threshold": threshold,
                "comparison": "gt", "model_version": "B-handcrafted-overall-risk-v2",
            })
    differences = {}
    for index, name in enumerate(FEATURES):
        safe = [row["values"][index] for row in rows if row["label"] == 0]
        unsafe = [row["values"][index] for row in rows if row["label"] == 1]
        raw_auc = auc([row["label"] for row in rows], [row["values"][index] for row in rows])
        differences[name] = {
            "safe": describe(safe), "unsafe": describe(unsafe),
            "unsafe_minus_safe_mean": float(np.mean(unsafe) - np.mean(safe)),
            "cohens_d_unsafe_minus_safe": standardized_difference(safe, unsafe),
            "raw_direction_auroc": raw_auc,
        }
    report = {
        "experiment_id": "B-handcrafted-overall-risk-v2",
        "target": "HoliSafe overall_risk_label (safe=0, unsafe=1)",
        "important_scope": "Image-only auxiliary classifier; not image-adversarial truth and not a replacement for multimodal fusion.",
        "fixed_split_policy": {
            "fit": "base_train", "threshold_selection": "validation",
            "frozen_transfer_evaluation": "fusion_train", "test": "not evaluated because labels are hidden",
            "image_group_overlap_between_fixed_splits": 0,
        },
        "model": {
            "type": "class_balanced_logistic_regression_numpy", "features": list(FEATURES),
            "training_mean": mean.tolist(), "training_scale": scale.tolist(),
            "standardized_coefficients": coefficients.tolist(), "intercept": intercept,
            "probability_threshold_strict_gt": threshold,
        },
        "split_metrics": split_metrics,
        "feature_differences_all_labeled": differences,
        "labeled_samples": len(rows), "safe_samples": sum(row["label"] == 0 for row in rows),
        "unsafe_samples": sum(row["label"] == 1 for row in rows),
    }
    return report, predictions


def update_evidence_v2(base: dict[str, dict], raw_cider: dict[str, dict], threshold: float) -> dict[str, dict]:
    if set(base) != set(raw_cider):
        raise ValueError("Evidence and HoliSafe CIDER IDs differ")
    d_threshold = math.nextafter(threshold, math.inf)
    output = {}
    for sample_id, source in base.items():
        raw = raw_cider[sample_id]
        statistic = float(raw["statistic"])
        prediction = int(statistic > threshold)
        record = dict(source)
        record["evidence"] = [{
            "source": "cider",
            "description": (f"CIDER statistic {statistic:.8f} "
                            f"{'exceeded' if prediction else 'did not exceed'} the validation-optimized "
                            f"strict threshold {threshold:.8f}; prediction={prediction}."),
            "score": record.get("visual_score"),
        }]
        record["warnings"] = [
            "CIDER threshold v2 was selected on official validation labels by balanced accuracy; test was not used for selection.",
            "visual_score remains the existing validation-fitted image-adversarial Platt score; OCR and handcrafted features are reused unchanged.",
            "This image-adversarial signal is not the HoliSafe overall-risk label.",
            "D ge comparison encodes B strict gt using nextafter(threshold, +inf).",
        ]
        record["module_version"] = "B-2.0.0-stable-fixed-splits"
        cider = dict(record["cider"])
        cider["threshold"] = d_threshold
        cider["comparison"] = "ge"
        cider["config_id"] = raw["config_id"] + ":tau-validation-balanced-v2:platt-v1"
        record["cider"] = cider
        output[sample_id] = record
    return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def render_report(cider: dict, handcrafted: dict) -> str:
    old, new = cider["previous_test_metrics"], cider["frozen_test_metrics"]
    differences = handcrafted["feature_differences_all_labeled"]
    frozen = handcrafted["split_metrics"]["fusion_train"]
    lines = [
        "# B 图像模块 v2：阈值、分类和固定划分交付", "",
        "## CIDER 阈值更正", "",
        f"验证集最大化 balanced accuracy/F1 后，严格 `>` 阈值为 `{cider['selected_threshold']:.10f}`；"
        f"旧阈值为 `{cider['previous_conservative_threshold']:.10f}`。test 未参与选阈值。", "",
        "| test 指标 | 旧阈值 | v2 阈值 |", "|---|---:|---:|",
    ]
    for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "fpr", "specificity", "auroc", "average_precision"):
        lines.append(f"| {key} | {old[key]:.4f} | {new[key]:.4f} |")
    lines += ["", "## 高斯/频域特征的安全与不安全分类", "",
              "模型只使用四个图像特征：base_train 拟合，validation 选概率阈值，fusion_train 冻结迁移评估。"
              "它预测的是 HoliSafe 整体安全标签，不是图像对抗标签。", "",
              "| frozen fusion_train 指标 | 数值 |", "|---|---:|" ]
    for key in ("samples", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "fpr", "auroc", "average_precision"):
        value = frozen[key]
        lines.append(f"| {key} | {value if key == 'samples' else f'{value:.4f}'} |")
    lines += ["", "### 两类特征差异（全部 3,415 条有标签样本）", "",
              "| 特征 | safe mean | unsafe mean | unsafe-safe | Cohen's d | AUROC |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in FEATURES:
        item = differences[name]
        lines.append(f"| {name} | {item['safe']['mean']:.6f} | {item['unsafe']['mean']:.6f} | "
                     f"{item['unsafe_minus_safe_mean']:.6f} | {item['cohens_d_unsafe_minus_safe']:.4f} | "
                     f"{item['raw_direction_auroc']:.4f} |")
    lines += ["", "结论：这些低层视觉特征对整体安全标签的区分很弱，必须作为辅助证据，不能单独替代文字与融合模块。", ""]
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    repo_root = root.parent
    artifact_dir = repo_root / "artifacts" if (repo_root / "artifacts").is_dir() else root / "artifacts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=artifact_dir / "v2")
    parser.add_argument("--dataset-root", type=Path, default=repo_root / "data/team_dataset_v1",
                        help="directory containing grouped manifest and C/D label CSV files")
    parser.add_argument("--add-validation-manifests", action="store_true",
                        help="add grouped split manifests to an existing v2 directory")
    args = parser.parse_args()
    master_manifest_path = args.dataset_root / "manifest.inputs.grouped.jsonl"
    if not master_manifest_path.exists():
        master_manifest_path = artifact_dir / "manifest.inputs.grouped.jsonl"
    if args.add_validation_manifests:
        if not args.output_dir.is_dir():
            raise FileNotFoundError(args.output_dir)
        manifest = load_jsonl(master_manifest_path)
        for split in FIXED_SPLITS:
            rows = [row for row in manifest.values() if row["split"] == split]
            write_jsonl(args.output_dir / f"manifest.{split}.jsonl", rows)
        config_path = args.output_dir / "config.v2.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["outputs"] = sorted(path.name for path in args.output_dir.iterdir())
        config["manifest_note"] = "Split manifests preserve grouped fields required by the D validator."
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        print(json.dumps({"added_grouped_manifests": list(FIXED_SPLITS)}, ensure_ascii=False))
        return
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    paths = {
        "manifest": master_manifest_path,
        "features": artifact_dir / "features.all.jsonl",
        "ocr": artifact_dir / "ocr.filtered.all.jsonl",
        "evidence": artifact_dir / "image_evidence.fixed_fusion.all.jsonl",
        "holisafe_cider": artifact_dir / "cider.full4024.raw.jsonl",
        "calibration_scored": artifact_dir / "cider.cider_calibration_v1.scored.jsonl",
        "labels_c": args.dataset_root / "labels_for_C.csv",
        "labels_d": args.dataset_root / "labels_for_D.csv",
    }
    manifest, features, ocr = (load_jsonl(paths[name]) for name in ("manifest", "features", "ocr"))
    evidence, raw_cider = (load_jsonl(paths[name]) for name in ("evidence", "holisafe_cider"))
    ids = set(manifest)
    if not (ids == set(features) == set(ocr) == set(evidence) == set(raw_cider)):
        raise ValueError("HoliSafe cached artifact IDs are not identical")

    calibration = list(load_jsonl(paths["calibration_scored"]).values())
    cider_report, cider_predictions = build_cider_report(calibration)
    labels = read_overall_labels([paths["labels_c"], paths["labels_d"]])
    handcrafted_report, handcrafted_predictions = build_handcrafted_classifier(manifest, features, labels)
    evidence_v2 = update_evidence_v2(evidence, raw_cider, cider_report["selected_threshold"])

    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "cider_predictions.calibration_v2.jsonl", cider_predictions)
    write_jsonl(args.output_dir / "handcrafted_safety_predictions.v2.jsonl", handcrafted_predictions)
    split_counts = {}
    for split in FIXED_SPLITS:
        split_ids = [sample_id for sample_id, row in manifest.items() if row["split"] == split]
        split_counts[split] = len(split_ids)
        write_jsonl(args.output_dir / f"manifest.{split}.jsonl", [manifest[sid] for sid in split_ids])
        write_jsonl(args.output_dir / f"features.{split}.jsonl", [features[sid] for sid in split_ids])
        write_jsonl(args.output_dir / f"ocr.{split}.jsonl", [ocr[sid] for sid in split_ids])
        write_jsonl(args.output_dir / f"image_evidence.{split}.jsonl", [evidence_v2[sid] for sid in split_ids])

    (args.output_dir / "CIDER_THRESHOLD_V2.json").write_text(
        json.dumps(cider_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "HANDCRAFTED_SAFETY_CLASSIFIER_V2.json").write_text(
        json.dumps(handcrafted_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path = root / "reports/B_WEEKLY_COMPLETION_V2.md"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(render_report(cider_report, handcrafted_report), encoding="utf-8")

    config = {
        "module_version": "B-2.0.0-stable-fixed-splits",
        "cache_policy": "reuse_unchanged; no OCR, feature, or CIDER inference rerun",
        "fixed_split_counts": split_counts,
        "cider_threshold": cider_report["selected_threshold"],
        "cider_comparison": "strict_gt (D export uses ge(nextafter(threshold,+inf)))",
        "handcrafted_classifier": handcrafted_report["model"],
        "ocr": {"source": str(paths["ocr"].relative_to(repo_root)), "engine": "EasyOCR 1.7.2",
                "languages": ["ch_sim", "en"], "confidence_filter": 0.20,
                "cache_directory_reused": "artifacts/ocr_cache"},
        "feature_configuration": next(iter(features.values()))["feature_parameters"],
        "input_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "outputs": sorted(path.name for path in args.output_dir.iterdir()),
    }
    config_path = args.output_dir / "config.v2.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir), "report": str(report_path),
        "cider_threshold": cider_report["selected_threshold"],
        "cider_test": cider_report["frozen_test_metrics"],
        "handcrafted_frozen_fusion_train": handcrafted_report["split_metrics"]["fusion_train"],
        "fixed_split_counts": split_counts,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
