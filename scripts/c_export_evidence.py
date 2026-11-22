"""C 模块：批量导出 TextEvidence（三类分数）与 D 可直接消费的纯 C 特征。

用法：
  python scripts/c_export_evidence.py --manifest <清单> --image-evidence <B证据> \
      --mode combined --tag val --out-dir outputs/C_week6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mm_jailbreak_guard.contracts import DetectionRequest, ImageEvidence  # noqa: E402
from mm_jailbreak_guard.io import load_jsonl, write_jsonl  # noqa: E402
from mm_jailbreak_guard.providers import make_text_provider  # noqa: E402

MODES = {
    "combined": ("mm_jailbreak_guard.rule_classifier_provider:create_provider", "combined"),
    "pure_rule": ("mm_jailbreak_guard.rule_classifier_provider:create_provider", "pure_rule"),
    "pure_classifier": ("mm_jailbreak_guard.rule_classifier_provider:create_provider", "pure_classifier"),
    "qwen3guard": ("mm_jailbreak_guard.qwen3guard_provider:create_provider", None),
    "lora": ("mm_jailbreak_guard.qwen3guard_lora_provider:create_provider", None),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request(record: dict, manifest: Path) -> DetectionRequest:
    image_path = record.get("image_path")
    if image_path:
        candidate = Path(str(image_path))
        if not candidate.is_absolute():
            candidate = manifest.parent / candidate
        image_path = str(candidate.resolve())
    return DetectionRequest(
        sample_id=str(record["sample_id"]),
        user_text=str(record.get("user_text", "")),
        image_path=image_path,
        image_name=Path(image_path).name if image_path else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-evidence", required=True)
    parser.add_argument("--mode", choices=tuple(MODES), default="combined")
    parser.add_argument("--tag", default="val")
    parser.add_argument("--out-dir", default="outputs/C_week6")
    args = parser.parse_args()

    spec, env_mode = MODES[args.mode]
    if env_mode:
        os.environ["C_PROVIDER_MODE"] = env_mode
    else:
        os.environ.pop("C_PROVIDER_MODE", None)

    manifest = Path(args.manifest).resolve()
    records = load_jsonl(manifest, unique_key="sample_id")
    images = {str(r["sample_id"]): ImageEvidence.from_dict(r)
              for r in load_jsonl(Path(args.image_evidence).resolve(), unique_key="sample_id")}

    provider = make_text_provider(spec, None)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence_rows, pure_rows = [], []
    started = time.perf_counter()
    for record in records:
        request = _request(record, manifest)
        image = images.get(request.sample_id) or ImageEvidence.unavailable(
            request.sample_id, "B 证据文件中缺少该样本")
        text = provider.analyze(request, image)
        evidence_rows.append(text.as_dict())
        pure_rows.append({
            "sample_id": text.sample_id,
            "available": text.available,
            "user_text_score": text.user_text_score,
            "ocr_text_score": text.ocr_text_score,
            "combined_text_score": text.combined_text_score,
            "module_version": text.module_version,
        })

    elapsed = time.perf_counter() - started
    ev_path = out_dir / f"{args.tag}.text_evidence.{args.mode}.jsonl"
    pc_path = out_dir / f"{args.tag}.pure_c.{args.mode}.jsonl"
    write_jsonl(ev_path, evidence_rows)
    write_jsonl(pc_path, pure_rows)

    def stat(field: str) -> str:
        values = [r[field] for r in evidence_rows if r[field] is not None]
        if not values:
            return "null=100.0%"
        return (f"非空={len(values)}/{len(evidence_rows)} "
                f"均值={statistics.fmean(values):.4f} 中位={statistics.median(values):.4f} "
                f"最小={min(values):.3f} 最大={max(values):.3f}")

    # 从 provider 读实际模型路径，别在这里写死文件名——升版本时极易漏改，
    # 会导致输出的 sha256 与实际打分用的模型不一致（v3 时就踩过一次）。
    from mm_jailbreak_guard.rule_classifier_provider import MODEL_PATH as model_path

    print(f"模式          : {args.mode}")
    print(f"模块版本      : {evidence_rows[0]['module_version'] if evidence_rows else 'n/a'}")
    print(f"样本数        : {len(evidence_rows)}  用时 {elapsed:.2f}s "
          f"({elapsed / max(1, len(evidence_rows)) * 1000:.1f} ms/条)")
    print(f"user_text_score    : {stat('user_text_score')}")
    print(f"ocr_text_score     : {stat('ocr_text_score')}")
    print(f"combined_text_score: {stat('combined_text_score')}")
    if args.mode in {"combined", "pure_classifier"} and model_path.is_file():
        print(f"模型 sha256   : {_sha256(model_path)}")
    print(f"证据文件      : {ev_path}")
    print(f"纯 C 特征     : {pc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
