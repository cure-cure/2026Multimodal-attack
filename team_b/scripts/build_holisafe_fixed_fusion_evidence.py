"""Build full HoliSafe ImageEvidence with frozen CIDER and visual scores.

The CIDER decision threshold is selected only from the official clean threshold
split.  The [0, 1] visual score is a Platt-scaled image-adversarial probability
fit only on the official CIDER validation split.  HoliSafe labels are never
read by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


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


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def fit_platt(scores: list[float], labels: list[int], max_iterations: int = 100) -> dict:
    """Fit deterministic two-parameter Platt scaling with smoothed targets."""
    if len(scores) != len(labels) or len(scores) < 2:
        raise ValueError("Platt calibration requires aligned non-empty scores and labels")
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Platt calibration requires both classes")
    mean = sum(scores) / len(scores)
    scale = math.sqrt(sum((value - mean) ** 2 for value in scores) / len(scores))
    if not scale or not math.isfinite(scale):
        raise ValueError("Calibration scores have zero or invalid variance")
    standardized = [(value - mean) / scale for value in scores]
    high = (positives + 1.0) / (positives + 2.0)
    low = 1.0 / (negatives + 2.0)
    targets = [high if label else low for label in labels]
    coefficient = 0.0
    intercept = math.log((positives + 1.0) / (negatives + 1.0))
    converged = False
    for iteration in range(1, max_iterations + 1):
        probabilities = [sigmoid(coefficient * value + intercept) for value in standardized]
        gradient_a = sum((probability - target) * value for probability, target, value
                         in zip(probabilities, targets, standardized))
        gradient_b = sum(probability - target for probability, target in zip(probabilities, targets))
        weights = [max(probability * (1.0 - probability), 1e-12)
                   for probability in probabilities]
        hessian_aa = sum(weight * value * value for weight, value in zip(weights, standardized)) + 1e-12
        hessian_ab = sum(weight * value for weight, value in zip(weights, standardized))
        hessian_bb = sum(weights) + 1e-12
        determinant = hessian_aa * hessian_bb - hessian_ab * hessian_ab
        if determinant <= 0 or not math.isfinite(determinant):
            raise ValueError("Singular Platt calibration Hessian")
        step_a = (hessian_bb * gradient_a - hessian_ab * gradient_b) / determinant
        step_b = (-hessian_ab * gradient_a + hessian_aa * gradient_b) / determinant
        coefficient -= step_a
        intercept -= step_b
        if max(abs(step_a), abs(step_b)) < 1e-10:
            converged = True
            break
    if not converged:
        raise ValueError("Platt calibration did not converge")
    if coefficient <= 0:
        raise ValueError("Fitted visual score is not monotonic in the expected CIDER direction")
    return {
        "method": "platt_logistic_smoothed_targets",
        "mean": mean,
        "scale": scale,
        "coefficient": coefficient,
        "intercept": intercept,
        "positive_target": high,
        "negative_target": low,
        "iterations": iteration,
    }


def platt_predict(statistic: float, model: dict) -> float:
    standardized = (float(statistic) - model["mean"]) / model["scale"]
    return sigmoid(model["coefficient"] * standardized + model["intercept"])


def auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
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
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def score_metrics(labels: list[int], scores: list[float]) -> dict:
    epsilon = 1e-15
    predictions = [int(score >= 0.5) for score in scores]
    return {
        "samples": len(labels),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "auroc": auc(labels, scores),
        "accuracy_at_0_5": sum(a == b for a, b in zip(labels, predictions)) / len(labels),
        "brier": sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(labels),
        "log_loss": -sum(label * math.log(min(1 - epsilon, max(epsilon, score)))
                         + (1 - label) * math.log(min(1 - epsilon, max(epsilon, 1 - score)))
                         for label, score in zip(labels, scores)) / len(labels),
        "score_min": min(scores),
        "score_max": max(scores),
        "score_mean": sum(scores) / len(scores),
    }


def calibration_rows(scored: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for sample_id, record in scored.items():
        split = record.get("split")
        label = record.get("image_adv_label")
        statistic = record.get("statistic")
        if split not in {"threshold_clean", "validation", "test"}:
            raise ValueError(f"Invalid calibration split: {sample_id}")
        if label not in {0, 1} or isinstance(label, bool):
            raise ValueError(f"Invalid calibration label: {sample_id}")
        if isinstance(statistic, bool) or not isinstance(statistic, (int, float)) or not math.isfinite(statistic):
            raise ValueError(f"Invalid calibration statistic: {sample_id}")
        rows.append({"sample_id": sample_id, "split": split, "label": label, "statistic": float(statistic)})
    return rows


def build(args: argparse.Namespace) -> dict:
    outputs = (args.output, args.report)
    if any(path.exists() for path in outputs):
        raise FileExistsError("Output already exists: " + ", ".join(str(path) for path in outputs if path.exists()))
    rows = calibration_rows(load_jsonl(args.calibration_scored))
    validation = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]
    threshold_clean = [row for row in rows if row["split"] == "threshold_clean"]
    if not validation or not test or not threshold_clean:
        raise ValueError("Calibration data must contain threshold_clean, validation, and test")
    threshold = sorted(row["statistic"] for row in threshold_clean)[
        max(0, math.ceil(args.clean_pass_rate * len(threshold_clean)) - 1)
    ]
    model = fit_platt([row["statistic"] for row in validation],
                      [row["label"] for row in validation])
    validation_scores = [platt_predict(row["statistic"], model) for row in validation]
    test_scores = [platt_predict(row["statistic"], model) for row in test]

    base = load_jsonl(args.base_evidence)
    raw = load_jsonl(args.holisafe_raw)
    if set(base) != set(raw):
        raise ValueError("HoliSafe base evidence and raw CIDER IDs differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predicted_adversarial = 0
    score_sum = 0.0
    d_threshold = math.nextafter(threshold, math.inf)
    with args.output.open("x", encoding="utf-8") as stream:
        for sample_id, record in base.items():
            cider = raw[sample_id]
            if cider.get("status") != "success":
                raise ValueError(f"HoliSafe CIDER result failed: {sample_id}")
            statistic = float(cider["statistic"])
            visual_score = platt_predict(statistic, model)
            prediction = int(statistic > threshold)
            predicted_adversarial += prediction
            score_sum += visual_score
            relation = "exceeded" if prediction else "did not exceed"
            output = dict(record)
            output["visual_score"] = visual_score
            output["evidence"] = [{
                "source": "cider",
                "description": (f"CIDER statistic {statistic:.8f} {relation} the frozen strict "
                                f"threshold {threshold:.8f}; prediction={prediction}."),
                "score": visual_score,
            }]
            output["warnings"] = [
                "visual_score is a Platt-calibrated image-adversarial probability; it is not HoliSafe overall risk.",
                "Calibration used only CIDER official validation labels; HoliSafe labels were not read.",
                "D ge comparison encodes B strict gt using nextafter(threshold, +inf).",
            ]
            output["module_version"] = "B-1.2.0-cider-platt-v1"
            output["cider"] = {
                "similarities": cider["similarities"],
                "statistic": statistic,
                "statistic_name": cider["statistic_name"],
                "threshold": d_threshold,
                "comparison": "ge",
                "config_id": cider["config_id"] + ":tau95-clean:platt-v1",
                "reproduction": cider["reproduction"],
            }
            stream.write(json.dumps(output, ensure_ascii=False, allow_nan=False) + "\n")

    report = {
        "experiment_id": "B-visual-score-platt-v1",
        "target": "image_adversarial_probability",
        "not_target": "HoliSafe overall risk",
        "data_policy": {
            "threshold_fit_split": "CIDER official threshold_clean",
            "score_fit_split": "CIDER official validation",
            "evaluation_split": "CIDER official test",
            "holisafe_labels_used": False,
        },
        "cider_decision": {
            "clean_pass_rate_target": args.clean_pass_rate,
            "threshold_strict_gt": threshold,
            "d_threshold_ge": d_threshold,
            "threshold_clean_samples": len(threshold_clean),
            "observed_clean_pass_rate": sum(row["statistic"] <= threshold for row in threshold_clean)
                                        / len(threshold_clean),
        },
        "visual_score_model": model,
        "validation_fit_metrics": score_metrics([row["label"] for row in validation], validation_scores),
        "frozen_test_metrics": score_metrics([row["label"] for row in test], test_scores),
        "holisafe_output": {
            "samples": len(base),
            "visual_score_non_null": len(base),
            "cider_non_null": len(base),
            "predicted_adversarial": predicted_adversarial,
            "visual_score_mean": score_sum / len(base),
            "path": str(Path("artifacts") / args.output.name),
        },
        "input_sha256": {
            "calibration_scored": file_sha256(args.calibration_scored),
            "holisafe_base_evidence": file_sha256(args.base_evidence),
            "holisafe_raw_cider": file_sha256(args.holisafe_raw),
        },
        "output_sha256": file_sha256(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    artifact_root = (root if (root / "artifacts/cider.full4024.raw.jsonl").is_file()
                     else root.parent)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-scored", type=Path,
                        default=artifact_root / "artifacts/cider.cider_calibration_v1.scored.jsonl")
    parser.add_argument("--base-evidence", type=Path,
                        default=artifact_root / "artifacts/image_evidence.final.all.jsonl")
    parser.add_argument("--holisafe-raw", type=Path,
                        default=artifact_root / "artifacts/cider.full4024.raw.jsonl")
    parser.add_argument("--output", type=Path,
                        default=artifact_root / "artifacts/image_evidence.fixed_fusion.all.jsonl")
    parser.add_argument("--report", type=Path,
                        default=root / "reports/B_VISUAL_SCORE_CALIBRATION.json")
    parser.add_argument("--clean-pass-rate", type=float, default=0.95)
    args = parser.parse_args()
    if not 0 < args.clean_pass_rate < 1:
        parser.error("--clean-pass-rate must be between 0 and 1")
    return args


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps({"output": result["holisafe_output"],
                      "frozen_test_metrics": result["frozen_test_metrics"]},
                     ensure_ascii=False, indent=2))
