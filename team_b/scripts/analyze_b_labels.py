"""Analyze image-adversarial and HoliSafe overall-risk labels separately."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

FEATURES = ("gaussian_residual_mae", "gaussian_residual_rmse",
            "high_frequency_ratio", "spectral_centroid")


def jsonl(path):
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


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def higher_quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No values")
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def auc(labels, scores):
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks, start = [0.0] * len(scores), 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(labels, scores):
    positives = sum(labels)
    if not positives:
        return None
    found = total = 0
    for rank, index in enumerate(sorted(range(len(scores)), key=lambda i: (-scores[i], i)), 1):
        if labels[index]:
            found += 1
            total += found / rank
    return total / positives


def metrics(labels, scores, threshold):
    predictions = [int(score > threshold) for score in scores]
    tp = sum(y and p for y, p in zip(labels, predictions))
    tn = sum(not y and not p for y, p in zip(labels, predictions))
    fp = sum(not y and p for y, p in zip(labels, predictions))
    fn = sum(y and not p for y, p in zip(labels, predictions))
    div = lambda a, b: a / b if b else None
    precision, recall, specificity = div(tp, tp + fp), div(tp, tp + fn), div(tn, tn + fp)
    return {
        "samples": len(labels), "positives": sum(labels), "negatives": len(labels) - sum(labels),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": div(tp + tn, len(labels)), "precision": precision, "recall": recall,
        "f1": div(2 * precision * recall, precision + recall), "fpr": div(fp, fp + tn),
        "specificity": specificity, "balanced_accuracy": (recall + specificity) / 2,
        "auroc": auc(labels, scores), "average_precision": average_precision(labels, scores),
    }


def describe(values):
    ordered = sorted(values)
    return {"n": len(ordered), "min": ordered[0], "q25": higher_quantile(ordered, .25),
            "median": median(ordered), "mean": sum(ordered) / len(ordered),
            "q75": higher_quantile(ordered, .75), "max": ordered[-1]}


def compare(rows, names):
    labels = [row["label"] for row in rows]
    result = {}
    for name in names:
        scores = [row[name] for row in rows]
        value = auc(labels, scores)
        result[name] = {
            "label_0": describe([score for score, label in zip(scores, labels) if not label]),
            "label_1": describe([score for score, label in zip(scores, labels) if label]),
            "auroc_raw_direction": value,
            "auroc_best_orientation_exploratory": max(value, 1 - value),
        }
    return result


def adversarial_axis(args):
    manifest, cider, features = jsonl(args.calibration_manifest), jsonl(args.calibration_cider), jsonl(args.calibration_features)
    labels = {row["sample_id"]: int(row["image_adv_label"]) for row in csv_rows(args.calibration_labels)}
    ids = set(manifest)
    if not (ids == set(labels) == set(cider) == set(features)):
        raise ValueError("Calibration artifacts do not have identical IDs")
    rows = []
    for sid, source in manifest.items():
        if cider[sid].get("status") != "success" or features[sid].get("status") != "success":
            raise ValueError(f"Failed record: {sid}")
        row = {"sample_id": sid, "split": source["split"], "label": labels[sid],
               "image_sha256": cider[sid]["image_sha256"],
               "cider_statistic": cider[sid]["statistic"], **features[sid]["visual_features"]}
        rows.append(row)
    clean = [row["cider_statistic"] for row in rows if row["split"] == "threshold_clean"]
    threshold = higher_quantile(clean, args.clean_pass_rate)
    report = {"target": "image_adv_label",
              "meaning": "optimization-based clean (0) versus adversarial (1) image",
              "threshold_protocol": {"source_split": "threshold_clean", "comparison": "gt",
                                     "clean_pass_rate_target": args.clean_pass_rate,
                                     "threshold": threshold, "samples": len(clean),
                                     "observed_clean_pass_rate": sum(x <= threshold for x in clean) / len(clean)}}
    for split in ("validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        unique = {}
        for row in selected:
            unique.setdefault(row["image_sha256"], row)
        report[split] = {
            "cider_pair_level": metrics([row["label"] for row in selected],
                                        [row["cider_statistic"] for row in selected], threshold),
            "cider_distributions": compare(selected, ("cider_statistic",)),
            "handcrafted_unique_image_level": compare(list(unique.values()), FEATURES),
            "unique_images": len(unique),
        }
    scored = [{"sample_id": row["sample_id"], "split": row["split"],
               "image_adv_label": row["label"], "statistic": row["cider_statistic"],
               "threshold": threshold, "prediction": int(row["cider_statistic"] > threshold),
               "comparison": "gt"} for row in rows]
    return report, scored


def holisafe_axis(args):
    truths = {}
    for path in (args.holisafe_labels_c, args.holisafe_labels_d):
        for row in csv_rows(path):
            value = row.get("overall_risk_label", "").strip()
            if value not in {"0", "1"}:
                continue
            sid, label = row["sample_id"], int(value)
            if sid in truths and truths[sid] != label:
                raise ValueError(f"Conflicting overall label: {sid}")
            truths[sid] = label
    manifest, cider, features = jsonl(args.holisafe_manifest), jsonl(args.holisafe_cider), jsonl(args.holisafe_features)
    rows = []
    for sid, label in truths.items():
        if sid not in manifest or sid not in cider or sid not in features:
            continue
        if cider[sid].get("status") != "success" or features[sid].get("status") != "success":
            continue
        rows.append({"sample_id": sid, "split": manifest[sid]["split"], "label": label,
                     "cider_statistic": cider[sid]["statistic"], **features[sid]["visual_features"]})
    return {"target": "overall_risk_label",
            "meaning": "broad multimodal input safety; not image-adversarial truth",
            "samples": len(rows), "positives": sum(row["label"] for row in rows),
            "negatives": sum(not row["label"] for row in rows),
            "split_counts": {split: sum(row["split"] == split for row in rows)
                             for split in sorted({row["split"] for row in rows})},
            "exploratory_signal_comparison": compare(rows, ("cider_statistic",) + FEATURES),
            "warning": "Do not report these AUROCs as image-adversarial detection performance."}


def render(report):
    attack, broad = report["image_adversarial_axis"], report["holisafe_overall_risk_axis"]
    protocol = attack["threshold_protocol"]
    lines = ["# Task B dual-axis label analysis", "",
             "The two truth definitions are analyzed separately and are never pooled.", "",
             "## Image-adversarial axis (CIDER official data)", "",
             f"Threshold `{protocol['threshold']:.8f}` with strict `>`; observed clean pass rate "
             f"{protocol['observed_clean_pass_rate']:.2%} on {protocol['samples']} clean pairs.", "",
             "| split | n | AUROC | AP | accuracy | recall | FPR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("validation", "test"):
        item = attack[split]["cider_pair_level"]
        lines.append(f"| {split} | {item['samples']} | {item['auroc']:.4f} | {item['average_precision']:.4f} | "
                     f"{item['accuracy']:.4f} | {item['recall']:.4f} | {item['fpr']:.4f} |")
    lines += ["", "## Overall-risk axis (HoliSafe)", "",
              f"Exploratory comparison on {broad['samples']} labeled pairs "
              f"({broad['positives']} unsafe, {broad['negatives']} safe).", "",
              "| signal | safe mean | unsafe mean | raw AUROC |", "|---|---:|---:|---:|"]
    for name, item in broad["exploratory_signal_comparison"].items():
        lines.append(f"| {name} | {item['label_0']['mean']:.6f} | {item['label_1']['mean']:.6f} | "
                     f"{item['auroc_raw_direction']:.4f} |")
    lines += ["", "HoliSafe results are broad safety associations, not adversarial-image accuracy.", ""]
    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-pass-rate", type=float, default=.95)
    parser.add_argument("--calibration-manifest", type=Path, default=root / "data/cider_calibration_v1/manifest.inputs.jsonl")
    parser.add_argument("--calibration-labels", type=Path, default=root / "data/cider_calibration_v1/labels_for_B.csv")
    parser.add_argument("--calibration-cider", type=Path, default=root / "artifacts/cider.cider_calibration_v1.raw.jsonl")
    parser.add_argument("--calibration-features", type=Path, default=root / "artifacts/features.cider_calibration_v1.jsonl")
    parser.add_argument("--holisafe-manifest", type=Path, default=root.parent / "data/team_dataset_v1/manifest.inputs.grouped.jsonl")
    parser.add_argument("--holisafe-labels-c", type=Path, default=root.parent / "data/team_dataset_v1/labels_for_C.csv")
    parser.add_argument("--holisafe-labels-d", type=Path, default=root.parent / "data/team_dataset_v1/labels_for_D.csv")
    parser.add_argument("--holisafe-cider", type=Path, default=root / "artifacts/cider.full4024.raw.jsonl")
    parser.add_argument("--holisafe-features", type=Path, default=root / "artifacts/features.all.jsonl")
    parser.add_argument("--output-json", type=Path, default=root / "reports/B_DUAL_AXIS_ANALYSIS.json")
    parser.add_argument("--output-md", type=Path, default=root / "reports/B_DUAL_AXIS_ANALYSIS.md")
    parser.add_argument("--scored-output", type=Path, default=root / "artifacts/cider.cider_calibration_v1.scored.jsonl")
    args = parser.parse_args()
    if not 0 < args.clean_pass_rate < 1:
        raise ValueError("clean-pass-rate must be between zero and one")
    for path in (args.output_json, args.output_md, args.scored_output):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    attack, scored = adversarial_axis(args)
    report = {"analysis_version": "B-dual-axis-1.0", "image_adversarial_axis": attack,
              "holisafe_overall_risk_axis": holisafe_axis(args)}
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    args.output_md.write_text(render(report), encoding="utf-8")
    with args.scored_output.open("x", encoding="utf-8") as stream:
        for row in scored:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md),
                      "scored_output": str(args.scored_output),
                      "threshold": attack["threshold_protocol"]["threshold"],
                      "validation": attack["validation"]["cider_pair_level"],
                      "test": attack["test"]["cider_pair_level"],
                      "holisafe_samples": report["holisafe_overall_risk_axis"]["samples"]},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
