from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .contracts import DetectionRequest
from .fusion import FusionBundle, FusionEngine, TuningExample, build_feature_vector, fit_bundle
from .io import load_jsonl, write_json, write_jsonl
from .pipeline import DetectionPipeline
from .providers import make_image_provider, make_text_provider


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

    fit_parser = subparsers.add_parser("fit", help="只在验证集拟合三种融合模式")
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
    batch_parser.add_argument("--mode", choices=("user_text_only", "user_text_ocr", "full"), default="full")
    batch_parser.add_argument("--output", required=True)
    batch_parser.set_defaults(handler=_batch)

    compare_parser = subparsers.add_parser("compare", help="逐样本输出三种方案")
    compare_parser.add_argument("--manifest", required=True)
    _add_upstream_arguments(compare_parser)
    compare_parser.add_argument("--config", required=True)
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


def _load_manifest(path: str) -> tuple[Path, list[dict[str, Any]]]:
    manifest = Path(path).resolve()
    return manifest, load_jsonl(manifest, unique_key="sample_id")


def _providers(args: argparse.Namespace):
    return (
        make_image_provider(args.image_provider, args.image_evidence),
        make_text_provider(args.text_provider, args.text_evidence),
    )


def _fit(args: argparse.Namespace) -> int:
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
    bundle = FusionBundle.load(args.config)
    return DetectionPipeline(image_provider, text_provider, FusionEngine(bundle))


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
    output: list[dict[str, Any]] = []
    for record in records:
        request = _request_from_record(record, manifest_path)
        image, text = pipeline.collect_evidence(request)
        schemes = {}
        for mode in ("user_text_only", "user_text_ocr", "full"):
            fusion = pipeline.fusion_engine.fuse(request, image, text, mode)
            schemes[mode] = {
                "prediction": fusion.prediction,
                "label_name": fusion.label_name,
                "risk_score": fusion.risk_score,
                "threshold": fusion.threshold,
                "component_scores": fusion.component_scores,
                "warnings": list(fusion.warnings),
            }
        output.append(
            {
                "sample_id": request.sample_id,
                "schemes": schemes,
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
