from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .contracts import DetectionRequest
from .fusion import FusionBundle, FusionEngine, TuningExample, build_feature_vector, fit_bundle, MODE_FEATURES, PRIMARY_MODES
from .io import load_jsonl, write_json, write_jsonl
from .pipeline import DetectionPipeline
from .providers import make_image_provider, make_text_provider
from .handoff import validate_handoff
from .pipeline import image_decision
from .learned_fusion import load_fusion_engine, train_bundle
from .training_io import adapt_manifest, file_digest, validate_training_manifests


def _add_upstream_arguments(parser: argparse.ArgumentParser) -> None:
    image = parser.add_mutually_exclusive_group()
    image.add_argument("--image-provider", help="B 在线提供方 package.module:factory")
    image.add_argument("--image-evidence", help="B 离线 JSONL")
    text = parser.add_mutually_exclusive_group()
    text.add_argument("--text-provider", help="C 在线提供方 package.module:factory")
    text.add_argument("--text-evidence", help="C 离线 JSONL")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mm-guard", description="第十题 D 模块：多证据融合与系统集成"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect-evidence", help="调用B/C提供方，冻结可追溯的证据JSONL供后续融合")
    collect.add_argument("--manifest", required=True)
    _add_upstream_arguments(collect)
    collect.add_argument("--config", required=True, help="系统配置；本命令只收集证据，不运行融合")
    collect.add_argument("--image-output", required=True)
    collect.add_argument("--text-output", required=True)
    collect.set_defaults(handler=_collect)
    adapter = subparsers.add_parser("adapt-manifest", help="从A分组CSV补充清单字段，生成新文件")
    adapter.add_argument("--manifest", required=True)
    adapter.add_argument("--groups-csv", required=True)
    adapter.add_argument("--output", required=True)
    adapter.set_defaults(handler=_adapt)
    train = subparsers.add_parser("train-fusion", help="D手写逻辑回归：fusion_train训练，validation/val选阈值")
    train.add_argument("--train-manifest", required=True)
    train.add_argument("--validation-manifest", required=True)
    _add_upstream_arguments(train)
    train.add_argument("--validation-image-evidence", help="验证集单独的B结果；省略则共用训练提供方")
    train.add_argument("--validation-text-evidence", help="验证集单独的C结果；省略则共用训练提供方")
    train.add_argument("--baseline-config", required=True)
    train.add_argument("--epochs", type=int, default=600)
    train.add_argument("--learning-rate", type=float, default=0.2)
    train.add_argument("--l2", type=float, default=0.01)
    train.add_argument("--output", required=True)
    train.set_defaults(handler=_train_fusion)
    check_parser = subparsers.add_parser("validate", help="校验四类划分、分组隔离及 B/C 离线交接文件")
    check_parser.add_argument("--manifest", required=True)
    check_parser.add_argument("--image-evidence")
    check_parser.add_argument("--text-evidence")
    check_parser.set_defaults(handler=_validate)

    fit_parser = subparsers.add_parser("fit", help="旧原型：验证集网格搜索，仅供模拟或明确记录的基线实验")
    fit_parser.add_argument("--manifest", required=True)
    _add_upstream_arguments(fit_parser)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--weight-step", type=float, default=0.1)
    fit_parser.add_argument("--visual-weight-cap", type=float, default=0.2)
    fit_parser.set_defaults(handler=_fit)

    batch_parser = subparsers.add_parser("batch", help="批量生成一种模式的预测")
    batch_parser.add_argument("--manifest", required=True)
    _add_upstream_arguments(batch_parser)
    batch_parser.add_argument("--config", required=True)
    batch_parser.add_argument("--mode", choices=tuple(MODE_FEATURES), default="full")
    batch_parser.add_argument("--output", required=True)
    batch_parser.set_defaults(handler=_batch)

    compare_parser = subparsers.add_parser("compare", help="逐样本输出三种方案")
    compare_parser.add_argument("--manifest", required=True)
    _add_upstream_arguments(compare_parser)
    compare_parser.add_argument("--config", required=True)
    compare_parser.add_argument("--baseline-config", help="另外输出同证据的固定融合对照")
    compare_parser.add_argument("--output", required=True)
    compare_parser.set_defaults(handler=_compare)

    web_parser = subparsers.add_parser("web", help="启动本地演示页面")
    _add_upstream_arguments(web_parser)
    web_parser.add_argument("--config", required=True)
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8000)
    web_parser.add_argument("--records", default="runtime/detections.jsonl")
    web_parser.set_defaults(handler=_web)
    return parser


def _request_from_record(record: dict[str, Any], manifest_path: Path) -> DetectionRequest:
    image_path = record.get("image_path")
    if image_path:
        candidate = Path(str(image_path))
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        image_path = str(candidate.resolve())
    return DetectionRequest(
        sample_id=str(record["sample_id"]),
        user_text=str(record.get("user_text", "")),
        image_path=image_path,
        image_name=Path(image_path).name if image_path else None,
    )


def _validate(args: argparse.Namespace) -> int:
    result = validate_handoff(args.manifest, args.image_evidence, args.text_evidence)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _adapt(args):
    count = adapt_manifest(args.manifest, args.groups_csv, args.output)
    print(f"已适配{count}条记录，A源文件未修改：{args.output}")
    return 0


def _collect(args):
    path, rows = _load_manifest(args.manifest)
    outputs = [Path(args.image_output).resolve(), Path(args.text_output).resolve()]
    sources = {Path(p).resolve() for p in (args.manifest, args.config, args.image_evidence, args.text_evidence) if p}
    if len(set(outputs)) != 2 or any(p in sources for p in outputs):
        raise ValueError("证据输出须为两个不同的新路径，不得覆盖输入")
    pipeline = _pipeline(args)
    images, texts = [], []
    for row in rows:
        image, text = pipeline.collect_evidence(_request_from_record(row, path))
        images.append(image.as_dict())
        texts.append(text.as_dict())
    write_jsonl(args.image_output, images)
    write_jsonl(args.text_output, texts)
    print(f"已缓存{len(rows)}条B/C证据；失败项保留不可用状态，不代表检测成绩")
    return 0


def _train_fusion(args):
    training_rows, validation_rows = validate_training_manifests(args.train_manifest, args.validation_manifest)
    sources = [args.train_manifest, args.validation_manifest, args.baseline_config,
               args.image_evidence, args.text_evidence,
               args.validation_image_evidence, args.validation_text_evidence]
    if Path(args.output).resolve() in {Path(p).resolve() for p in sources if p}:
        raise ValueError("训练输出不得覆盖源清单、证据或固定配置")
    image_provider, text_provider = _providers(args)
    val_image = make_image_provider(evidence_path=args.validation_image_evidence) if args.validation_image_evidence else image_provider
    val_text = make_text_provider(evidence_path=args.validation_text_evidence) if args.validation_text_evidence else text_provider
    versions = {"B": set(), "C": set()}
    examples = []
    for path, records, image_provider, text_provider in (
        (args.train_manifest, training_rows, image_provider, text_provider),
        (args.validation_manifest, validation_rows, val_image, val_text)):
        current = []
        for row in records:
            request = _request_from_record(row, Path(path).resolve())
            image = image_provider.analyze(request)
            text = text_provider.analyze(request, image)
            if image.sample_id != request.sample_id or text.sample_id != request.sample_id:
                raise ValueError("训练证据sample_id与清单不一致")
            for name, evidence in (("B", image), ("C", text)):
                if evidence.available:
                    versions[name].add(evidence.module_version)
            current.append(TuningExample(request.sample_id, row['label'], build_feature_vector(request, image, text)))
        examples.append(current)
    if any(len(v) > 1 for v in versions.values()):
        raise ValueError("训练/验证的B/C版本不一致，须冻结同一版本再训练")
    provenance = {"train_split": "fusion_train", "threshold_split": "validation",
                  "train_count": len(training_rows), "validation_count": len(validation_rows),
                  "test_set_used": False, "module_versions": {k: sorted(v) for k, v in versions.items()},
                  "input_sha256": {str(index)+":"+Path(p).name: file_digest(p) for index, p in enumerate(sources) if p},
                  "note": "只记录软件训练；真实效果须由A用冻结测试协议评测"}
    bundle = train_bundle(*examples, FusionBundle.load(args.baseline_config), provenance=provenance,
                          epochs=args.epochs, learning_rate=args.learning_rate, l2=args.l2)
    write_json(args.output, bundle)
    print(f"已生成自主融合配置：{args.output}；课件对应仍待核验")
    return 0


def _load_manifest(path: str) -> tuple[Path, list[dict[str, Any]]]:
    manifest = Path(path).resolve()
    return manifest, load_jsonl(manifest, unique_key="sample_id")


def _providers(args: argparse.Namespace):
    return (
        make_image_provider(args.image_provider, args.image_evidence),
        make_text_provider(args.text_provider, args.text_evidence),
    )


def _fit(args: argparse.Namespace) -> int:
    print("提示：fit 是旧原型的验证集网格搜索，不代替新计划的 fusion_train 训练及 validation 选阈值。", file=sys.stderr)
    manifest_path, records = _load_manifest(args.manifest)
    if not records:
        raise ValueError("验证集清单为空")
    invalid_splits = {
        str(record.get("split", "<missing>")).casefold()
        for record in records
        if str(record.get("split", "<missing>")).casefold() != "validation"
    }
    if invalid_splits:
        raise ValueError(
            "fit 只允许 validation 划分；检测到：" + ", ".join(sorted(invalid_splits))
        )
    image_provider, text_provider = _providers(args)
    examples: list[TuningExample] = []
    for record in records:
        if "label" not in record or int(record["label"]) not in {0, 1}:
            raise ValueError(f"{record.get('sample_id')} 缺少二分类 label")
        request = _request_from_record(record, manifest_path)
        image = image_provider.analyze(request)
        if image.sample_id != request.sample_id:
            raise ValueError(f"B 返回的 sample_id 与请求不一致：{request.sample_id}")
        text = text_provider.analyze(request, image)
        if text.sample_id != request.sample_id:
            raise ValueError(f"C 返回的 sample_id 与请求不一致：{request.sample_id}")
        examples.append(
            TuningExample(
                sample_id=request.sample_id,
                label=int(record["label"]),
                features=build_feature_vector(request, image, text),
            )
        )
    bundle = fit_bundle(
        examples,
        weight_step=args.weight_step,
        visual_weight_cap=args.visual_weight_cap,
    )
    write_json(args.output, bundle.as_dict())
    print(f"已生成验证集融合配置：{args.output}")
    return 0


def _pipeline(args: argparse.Namespace) -> DetectionPipeline:
    image_provider, text_provider = _providers(args)
    return DetectionPipeline(image_provider, text_provider, load_fusion_engine(args.config))


def _batch(args: argparse.Namespace) -> int:
    manifest_path, records = _load_manifest(args.manifest)
    pipeline = _pipeline(args)
    output = []
    for record in records:
        report = pipeline.detect(_request_from_record(record, manifest_path), args.mode)
        output.append(report.as_prediction_record())
    write_jsonl(args.output, output)
    print(f"已写入 {len(output)} 条预测：{args.output}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    manifest_path, records = _load_manifest(args.manifest)
    pipeline = _pipeline(args)
    baseline_engine = load_fusion_engine(args.baseline_config) if args.baseline_config else None
    output: list[dict[str, Any]] = []
    for record in records:
        request = _request_from_record(record, manifest_path)
        image, text = pipeline.collect_evidence(request)
        schemes = {}
        for mode in PRIMARY_MODES:
            fusion = pipeline.fusion_engine.fuse(request, image, text, mode)
            schemes[mode] = {
                "prediction_target": "image_adversarial" if mode == "image_only" else "overall_risk",
                "image_decision": image_decision(image) if mode == "image_only" else None,
                "prediction": fusion.prediction,
                "label_name": fusion.label_name,
                "risk_score": fusion.risk_score,
                "threshold": fusion.threshold,
                "component_scores": fusion.component_scores,
                "warnings": list(fusion.warnings),
                "decision_rule": fusion.decision_rule,
            }
        baseline_schemes = {}
        if baseline_engine:
            for mode in PRIMARY_MODES:
                base = baseline_engine.fuse(request, image, text, mode)
                baseline_schemes[mode] = base.as_dict()
        output.append(
            {
                "sample_id": request.sample_id,
                "schemes": schemes,
                "fixed_baseline_schemes": baseline_schemes,
                "fusion_config_id": pipeline.fusion_engine.bundle.config_id,
                "module_versions": {
                    "B": image.module_version,
                    "C": text.module_version,
                    "D": f"D-{__version__}",
                },
            }
        )
    write_jsonl(args.output, output)
    print(f"已写入 {len(output)} 条三方案对比：{args.output}")
    return 0


def _web(args: argparse.Namespace) -> int:
    from .web import run_server

    run_server(_pipeline(args), args.host, args.port, args.records)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError, TypeError, ImportError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
