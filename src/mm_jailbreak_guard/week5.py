"""D 第五周：用同一批 B/C 证据运行固定融合并生成交接记录。"""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from . import __version__
from .contracts import DetectionRequest
from .fusion import PRIMARY_MODES, TEXT_FEATURES, FusionBundle, FusionEngine
from .handoff import validate_handoff
from .io import load_jsonl, write_json, write_jsonl
from .pipeline import DetectionPipeline, image_decision
from .providers import make_image_provider, make_text_provider
from .training_io import file_digest


def add_week5_batch_command(subparsers):
    parser = subparsers.add_parser("week5-batch", help="第五周固定融合、分支输出和交接日志")
    parser.add_argument("--manifest", required=True, help="A 的同一批清单；真实运行用验证集")
    parser.add_argument("--image-evidence", required=True, help="B 按同一清单交付的 ImageEvidence JSONL")
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--text-evidence", help="C 按同一清单交付的 TextEvidence JSONL")
    text.add_argument("--text-provider", help="C 可运行入口 package.module:factory")
    parser.add_argument("--config", required=True, help="预先确定的固定融合配置")
    parser.add_argument("--data-kind", choices=("real", "synthetic"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, help="未使用过的本地输出目录")
    parser.set_defaults(handler=run_week5_batch)


def _request(row, manifest):
    image_path = (manifest.parent / row["image_path"]).resolve() if row["image_path"] else None
    return DetectionRequest(row["sample_id"], row["user_text"],
                            str(image_path) if image_path else None)


def _is_mock(value):
    return "mock" in value.casefold() or "synthetic" in value.casefold()


def run_week5_batch(args):
    manifest = Path(args.manifest).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"运行目录已存在，请使用新目录：{output}")
    if not args.run_id.strip():
        raise ValueError("run-id 不能为空")

    validate_handoff(manifest, args.image_evidence, args.text_evidence)
    rows = load_jsonl(manifest, unique_key="sample_id")
    real = args.data_kind == "real"
    if real and any(row["split"] not in {"val", "validation"} or
                    _is_mock(row["sample_id"]) or _is_mock(row["source"]) for row in rows):
        raise ValueError("第五周真实运行只接受 A 的主验证集，不接受模拟数据、训练集或测试集")

    bundle = FusionBundle.load(args.config)
    if (bundle.algorithm != "manual_weighted_late_fusion" or bundle.trained_on_split is not None
            or not set(PRIMARY_MODES).issubset(bundle.modes)
            or bundle.selection.get("require_both_sides") is not True):
        raise ValueError("第五周须使用事先确定、包含三个主模式并要求图文双侧证据的固定加权配置")
    if real and (bundle.modes["full"].weights.get("visual_score", 0) <= 0 or
                 not any(bundle.modes["full"].weights.get(name, 0) > 0
                         for name in TEXT_FEATURES)):
        raise ValueError("真实完整融合配置必须给图像分数及至少一项文本分数正权重")

    sources = {"manifest": args.manifest, "image_evidence": args.image_evidence,
               "text_evidence": args.text_evidence, "config": args.config}
    input_sha256 = {name: file_digest(path) for name, path in sources.items() if path}
    pipeline = DetectionPipeline(make_image_provider(evidence_path=args.image_evidence),
                                 make_text_provider(args.text_provider, args.text_evidence),
                                 FusionEngine(bundle))
    predictions = {mode: [] for mode in PRIMARY_MODES}
    logs, versions = [], {"B": set(), "C": set()}
    for row in rows:
        started = perf_counter()
        request = _request(row, manifest)
        image, text = pipeline.collect_evidence(request)
        module_versions = {"B": image.module_version, "C": text.module_version, "D": f"D-{__version__}"}
        if real and (any(_is_mock(version) for version in module_versions.values()) or
                     (image.cider and image.cider.reproduction == "mock")):
            raise ValueError(f"{request.sample_id}: 真实运行检测到模拟证据")
        for member, evidence in (("B", image), ("C", text)):
            if evidence.available:
                versions[member].add(evidence.module_version)

        results = {mode: pipeline.fusion_engine.fuse(request, image, text, mode)
                   for mode in PRIMARY_MODES}
        full = results["full"]
        both_sides = full.prediction is not None
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        for mode, result in results.items():
            cider_rule = image_decision(image) if mode == "image_only" else None
            predictions[mode].append({
                "sample_id": request.sample_id, "mode": mode,
                "prediction_target": "image_adversarial" if mode == "image_only" else "overall_risk",
                "prediction": result.prediction, "risk_score": result.risk_score,
                "label_name": result.label_name,
                "threshold": None if cider_rule else result.threshold, "module_versions": module_versions,
                "fusion_config_id": bundle.config_id, "ocr_status": image.ocr_status,
                "image_decision": cider_rule,
            })
        logs.append({
            "sample_id": request.sample_id, "split": row["split"], "elapsed_ms": elapsed_ms,
            "image_available": image.available, "cider_available": image.cider is not None,
            "visual_score_available": image.visual_score is not None, "ocr_status": image.ocr_status,
            "text_available": text.available, "user_text_score_available": text.user_text_score is not None,
            "image_warning_count": len(image.warnings), "text_warning_count": len(text.warnings),
            "ocr_text_score_used": full.effective_weights.get("ocr_text_score", 0) > 0,
            "combined_text_score_used": full.effective_weights.get("combined_text_score", 0) > 0,
            "image_only_decided": results["image_only"].prediction is not None,
            "text_only_decided": results["text_only"].prediction is not None,
            "full_used_both_sides": both_sides, "module_versions": module_versions,
        })

    if real and any(len(v) > 1 or "unknown" in v or "unavailable" in v for v in versions.values()):
        raise ValueError("真实运行要求每个可用分支只使用一个已标注的模块版本")
    covered = {mode: sum(row["prediction"] is not None for row in result_rows)
               for mode, result_rows in predictions.items()}
    status = "synthetic_only" if not real else (
        "complete_coverage" if all(count == len(rows) for count in covered.values()) else "partial_coverage"
    )
    output.mkdir(parents=True, exist_ok=False)
    for mode, result_rows in predictions.items():
        write_jsonl(output / f"predictions.{mode}.jsonl", result_rows)
    write_jsonl(output / "run_log.jsonl", logs)
    write_json(output / "run_summary.json", {
        "run_id": args.run_id, "data_kind": args.data_kind, "status": status,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "sample_count": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "decision_counts": covered, "abstained_counts": {m: len(rows)-n for m, n in covered.items()},
        "ocr_status_counts": dict(Counter(row["ocr_status"] for row in logs)),
        "module_versions": {member: sorted(values) for member, values in versions.items()},
        "text_provider": args.text_provider, "fusion_config": bundle.as_dict(), "input_sha256": input_sha256,
    })
    print(f"{status}：写入 {output}。这些是逐条预测与覆盖日志，指标由 A 计算。")
    return 0
