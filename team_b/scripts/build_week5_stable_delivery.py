"""Build the week-5 stable image-side demo and anomaly-analysis delivery.

Only cached outputs are read.  The script does not rerun OCR/CIDER/features and
does not add or alter any ImageEvidence field consumed by fusion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median


FEATURES = (
    "gaussian_residual_mae",
    "gaussian_residual_rmse",
    "high_frequency_ratio",
    "spectral_centroid",
)
CONFUSION_ORDER = ("TP", "TN", "FP", "FN")


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confusion_name(label: int, prediction: int) -> str:
    return ("T" if label == prediction else "F") + ("P" if prediction else "N")


def representative(rows: list[dict]) -> dict:
    """Choose the observed statistic nearest the category median."""
    center = median(row["statistic"] for row in rows)
    return min(rows, key=lambda row: (abs(row["statistic"] - center), row["sample_id"]))


def describe(values: list[float]) -> dict:
    ordered = sorted(values)
    size = len(ordered)
    quantile = lambda probability: ordered[max(0, math.ceil(probability * size) - 1)]
    return {
        "n": size, "min": ordered[0], "q25": quantile(.25), "median": median(ordered),
        "mean": sum(ordered) / size, "q75": quantile(.75), "max": ordered[-1],
    }


def select_demo(calibration_manifest: dict[str, dict], predictions: dict[str, dict],
                features: dict[str, dict], ocr: dict[str, dict], output_dir: Path,
                calibration_data_dir: Path) -> list[dict]:
    test = [row for row in predictions.values() if row["split"] == "test"]
    groups = {name: [] for name in CONFUSION_ORDER}
    for row in test:
        groups[confusion_name(int(row["image_adv_label"]), int(row["prediction"]))].append(row)
    if any(not rows for rows in groups.values()):
        raise ValueError("All TP/TN/FP/FN groups are required for the demo")
    chosen = []
    for name in CONFUSION_ORDER:
        prediction = representative(groups[name])
        sample_id = prediction["sample_id"]
        source = calibration_manifest[sample_id]
        relative_image = Path(os.path.relpath(
            calibration_data_dir / source["image_path"], output_dir))
        chosen.append({
            "sample_id": sample_id,
            "image_path": relative_image.as_posix(),
            "user_text": source["user_text"],
            "demo_category": name,
            "image_adv_label": prediction["image_adv_label"],
            "expected_prediction": prediction["prediction"],
            "expected_statistic": prediction["statistic"],
            "threshold": prediction["threshold"],
            "visual_features": features[sample_id]["visual_features"],
            "ocr_status": ocr[sample_id]["ocr_status"],
            "ocr_text": ocr[sample_id].get("ocr_text", ""),
            "explanation": {
                "TP": "adversarial image correctly detected",
                "TN": "clean image correctly accepted",
                "FP": "clean image statistic exceeds the threshold",
                "FN": "adversarial image statistic remains below the threshold",
            }[name],
        })
    manifest_path = output_dir / "demo_manifest.jsonl"
    expected_path = output_dir / "demo_expected.jsonl"
    with manifest_path.open("x", encoding="utf-8") as stream:
        for row in chosen:
            stream.write(json.dumps({key: row[key] for key in ("sample_id", "image_path", "user_text")},
                                    ensure_ascii=False, allow_nan=False) + "\n")
    with expected_path.open("x", encoding="utf-8") as stream:
        for row in chosen:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return chosen


def analyze_anomalies(predictions: dict[str, dict], holisafe_features: dict[str, dict],
                      holisafe_ocr: dict[str, dict], robustness: dict) -> dict:
    threshold = next(iter(predictions.values()))["threshold"]
    splits = {}
    errors = []
    for split in ("validation", "test"):
        selected = [row for row in predictions.values() if row["split"] == split]
        counts = Counter(confusion_name(int(row["image_adv_label"]), int(row["prediction"]))
                         for row in selected)
        splits[split] = {"samples": len(selected), **{name: counts[name] for name in CONFUSION_ORDER}}
        for row in selected:
            category = confusion_name(int(row["image_adv_label"]), int(row["prediction"]))
            if category in {"FP", "FN"}:
                errors.append({
                    "sample_id": row["sample_id"], "split": split, "category": category,
                    "statistic": row["statistic"], "threshold": threshold,
                    "signed_margin": row["statistic"] - threshold,
                })
    nearest_errors = sorted(errors, key=lambda row: (abs(row["signed_margin"]), row["sample_id"]))[:20]
    farthest_errors = sorted(errors, key=lambda row: (-abs(row["signed_margin"]), row["sample_id"]))[:20]

    feature_report = {}
    successful = [row for row in holisafe_features.values() if row.get("status") == "success"]
    for name in FEATURES:
        values = [float(row["visual_features"][name]) for row in successful]
        low = sorted(successful, key=lambda row: (row["visual_features"][name], row["sample_id"]))[:5]
        high = sorted(successful, key=lambda row: (-row["visual_features"][name], row["sample_id"]))[:5]
        feature_report[name] = {
            "distribution": describe(values),
            "lowest": [{"sample_id": row["sample_id"], "value": row["visual_features"][name]} for row in low],
            "highest": [{"sample_id": row["sample_id"], "value": row["visual_features"][name]} for row in high],
            "interpretation": "Extreme texture/frequency values are review cues, not attack labels.",
        }
    ocr_counts = Counter(row.get("ocr_status", "unavailable") for row in holisafe_ocr.values())
    return {
        "analysis_version": "B-week5-anomaly-v1",
        "cider_threshold": threshold,
        "cider_confusion_by_split": splits,
        "cider_error_count": len(errors),
        "nearest_threshold_errors": nearest_errors,
        "largest_margin_errors": farthest_errors,
        "handcrafted_feature_extremes": feature_report,
        "ocr_status": dict(sorted(ocr_counts.items())),
        "ordinary_transform_sensitivity": robustness["summary"],
        "interpretation_boundaries": [
            "CIDER FP/FN is defined by image-adversarial truth, not whether user text is harmful.",
            "Gaussian/frequency extremes can be caused by natural texture, blur, resizing, or compression.",
            "OCR no_text means successful OCR with no retained region; it is not a processing failure.",
            "The handcrafted overall-risk classifier is diagnostic only because its frozen balanced accuracy is near chance.",
        ],
    }


def render_report(demo: list[dict], anomaly: dict, config: dict, hashes: dict) -> str:
    test = anomaly["cider_confusion_by_split"]["test"]
    lines = [
        "# 第五周 B 图像侧稳定版：演示、异常分析与复现说明", "",
        "> 版本：`B-2.0.0-stable-fixed-splits`。本周冻结融合接口，只修阻断问题；没有新增 `ImageEvidence` 字段或视觉特征。", "",
        "## 1. 独立图像侧演示", "",
        "演示顺序固定为 TP、TN、FP、FN。四条样例均来自 CIDER 官方 test，阈值只在 validation 上选择。", "",
        "| 类别 | sample_id | statistic | threshold | 标签/预测 | OCR |", "|---|---|---:|---:|---|---|",
    ]
    for row in demo:
        lines.append(f"| {row['demo_category']} | `{row['sample_id']}` | {row['expected_statistic']:.8f} | "
                     f"{row['threshold']:.8f} | {row['image_adv_label']}/{row['expected_prediction']} | "
                     f"{row['ocr_status']} |")
    lines += [
        "", "运行缓存演示（无需 GPU）：", "",
        "```bash", "python scripts/build_week5_stable_delivery.py --verify", "```", "",
        "运行实时四样例 CIDER（需要 CUDA 和本地权重）：", "",
        "```bash",
        "bash scripts/run_week5_batch.sh artifacts/week5/demo_manifest.jsonl outputs/week5_demo.jsonl",
        "```", "",
        "## 2. 图像异常分析", "",
        f"v2 test 混淆计数：TP={test['TP']}、TN={test['TN']}、FP={test['FP']}、FN={test['FN']}。"
        "误差清单同时保存阈值距离，便于优先检查边界样本和大间隔异常。", "",
        "普通变换的均值绝对变化：", "",
        "| 变换 | Gaussian MAE | Gaussian RMSE | 高频比例 | 频谱质心 |", "|---|---:|---:|---:|---:|",
    ]
    for transform, item in anomaly["ordinary_transform_sensitivity"].items():
        lines.append(f"| {transform} | {item['gaussian_residual_mae']['mean_absolute_delta']:.6f} | "
                     f"{item['gaussian_residual_rmse']['mean_absolute_delta']:.6f} | "
                     f"{item['high_frequency_ratio']['mean_absolute_delta']:.6f} | "
                     f"{item['spectral_centroid']['mean_absolute_delta']:.6f} |")
    lines += [
        "", "异常解释边界：高频或高斯残差极值不是攻击判定；模糊会系统性降低相关特征，JPEG Q50 在这 12 张校园图上的变化较小；"
        "OCR `no_text` 与 `failed` 必须分开。", "",
        "## 3. 复现差异", "",
        "- 复用 CIDER 的 guided diffusion 参数和 LLaVA 视觉层/投影层/输入词嵌入，不运行回答生成或 HarmBench。",
        "- 阶段 50–350 是时间索引；每个阶段从原图重新加噪并做一次 `pred_xstart`，不是 350 步连续逆扩散。",
        "- 图片统一 224×224；阶段图保存在内存，使用 Pillow 双线性缩放，不经过作者 JPEG 中间文件，因此属于 adapted 复现。",
        "- 统计量为 `max(sim_original - sim_denoised)`，保留负值；判定为严格 `>`。D 的 `ge` 用 `nextafter` 等价编码。",
        "- 固定 seed=2026 并结合图片 SHA-256 派生逐图 seed；相同图片复用视觉向量，但不同文字仍单独编码。",
        "- v2 阈值在官方 validation 上最大化 balanced accuracy，test 只做冻结评估；不使用 HoliSafe 整体风险标签选 CIDER 阈值。",
        "- OCR 和自主高斯/频域特征沿用已冻结缓存，本周没有重算或修改融合接口。", "",
        "## 4. 批量测试入口", "",
        "输入 JSONL 每行必须有 `sample_id`、`image_path`、`user_text`。脚本先执行环境和缺图检查，再运行常驻 GPU 的 `run_b_fast.py`，"
        "最后校验行数、ID、有限数值、阈值及严格比较结果。输出已存在时拒绝覆盖。", "",
        "```bash",
        "bash scripts/run_week5_batch.sh /path/manifest.jsonl /path/output.jsonl",
        "```", "",
        "固定配置：`config.week5_stable_v2.json`；阈值："
        f"`{config['threshold']:.10f}`；阶段：`{config['stages']}`；配置 ID：`{config['config_id']}`。", "",
        "## 5. 接口冻结与交付", "",
        "`visual_features` 仍只有四项：Gaussian residual MAE/RMSE、high-frequency ratio、spectral centroid。"
        "`ImageEvidence` 合同、OCR region 结构和 CIDER 字段均未增加。四个固定划分证据已通过 D validator。", "",
        "输入哈希：", "",
    ]
    lines.extend(f"- `{name}`: `{value}`" for name, value in hashes.items())
    lines.append("")
    return "\n".join(lines)


def validate_cached_delivery(root: Path) -> dict:
    repo_root = root.parent
    artifact_dir = repo_root / "artifacts" if (repo_root / "artifacts").is_dir() else root / "artifacts"
    output_dir = artifact_dir / "week5"
    demo = load_jsonl(output_dir / "demo_expected.jsonl")
    config = json.loads((root / "config.week5_stable_v2.json").read_text(encoding="utf-8"))
    threshold = float(config["threshold"])
    for sample_id, row in demo.items():
        expected = int(float(row["expected_statistic"]) > threshold)
        if expected != row["expected_prediction"]:
            raise ValueError(f"Demo threshold mismatch: {sample_id}")
        if set(row["visual_features"]) != set(FEATURES):
            raise ValueError(f"Demo feature contract changed: {sample_id}")
    anomaly = json.loads((output_dir / "ANOMALY_ANALYSIS.json").read_text(encoding="utf-8"))
    if anomaly["cider_confusion_by_split"]["test"] != {
        "samples": 1280, "TP": 553, "TN": 606, "FP": 34, "FN": 87,
    }:
        raise ValueError("Frozen test confusion counts changed")
    return {"status": "valid", "demo_samples": len(demo), "threshold": threshold,
            "fusion_interface_features": list(FEATURES)}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    repo_root = root.parent
    artifact_dir = repo_root / "artifacts" if (repo_root / "artifacts").is_dir() else root / "artifacts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(validate_cached_delivery(root), ensure_ascii=False, indent=2))
        return
    output_dir = artifact_dir / "week5"
    report_path = root / "reports/B_WEEK5_IMAGE_CHAPTER.md"
    if output_dir.exists() or report_path.exists():
        raise FileExistsError("Week-5 output already exists; use --verify")
    output_dir.mkdir(parents=True)
    paths = {
        "calibration_manifest": root / "data/cider_calibration_v1/manifest.inputs.jsonl",
        "calibration_predictions": artifact_dir / "v2/cider_predictions.calibration_v2.jsonl",
        "calibration_features": artifact_dir / "features.cider_calibration_v1.jsonl",
        "calibration_ocr": artifact_dir / "ocr.cider_calibration_v1.jsonl",
        "holisafe_features": artifact_dir / "features.all.jsonl",
        "holisafe_ocr": artifact_dir / "ocr.filtered.all.jsonl",
        "robustness": artifact_dir / "robustness.campus12.json",
        "config": root / "config.week5_stable_v2.json",
    }
    loaded = {name: load_jsonl(path) for name, path in paths.items()
              if name not in {"robustness", "config"}}
    predictions = loaded["calibration_predictions"]
    demo = select_demo(loaded["calibration_manifest"], predictions,
                       loaded["calibration_features"], loaded["calibration_ocr"], output_dir,
                       paths["calibration_manifest"].parent)
    robustness = json.loads(paths["robustness"].read_text(encoding="utf-8"))
    anomaly = analyze_anomalies(predictions, loaded["holisafe_features"],
                                loaded["holisafe_ocr"], robustness)
    (output_dir / "ANOMALY_ANALYSIS.json").write_text(
        json.dumps(anomaly, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    hashes = {name: sha256(path) for name, path in paths.items()}
    delivery = {
        "version": "B-week5-stable-v1", "interface_changed": False,
        "new_fusion_features": [], "demo_categories": list(CONFUSION_ORDER),
        "input_sha256": hashes,
    }
    (output_dir / "DELIVERY_MANIFEST.json").write_text(
        json.dumps(delivery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(demo, anomaly, config, hashes), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "report": str(report_path),
                      "demo": [row["sample_id"] for row in demo],
                      "test_confusion": anomaly["cider_confusion_by_split"]["test"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
