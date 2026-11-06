"""D 数据适配与训练边界校验；不修改A的源文件、划分或真值。"""
import csv
import hashlib
import os
from pathlib import Path

from .handoff import validate_manifest_records
from .io import load_jsonl, write_jsonl


def normalized_split(value):
    return "validation" if value == "val" else value


def adapt_manifest(manifest, groups_csv, output):
    source, target = Path(manifest).resolve(), Path(output).resolve()
    if target in {source, Path(groups_csv).resolve()}:
        raise ValueError("适配输出必须是新文件，不得覆盖A源文件")
    with Path(groups_csv).open(encoding="utf-8-sig", newline="") as handle:
        groups = {}
        for row in csv.DictReader(handle):
            sid = row.get("sample_id")
            if not sid or sid in groups:
                raise ValueError("分组CSV的sample_id缺失或重复")
            groups[sid] = row
    rows = load_jsonl(source, unique_key="sample_id")
    result = []
    for row in rows:
        row = dict(row)
        sid = row["sample_id"]
        if sid not in groups:
            raise ValueError(f"{sid}: A分组CSV中不存在此样本")
        for key in ("image_group_id", "request_group_id"):
            value = groups[sid].get(key)
            if not value or (key in row and row[key] != value):
                raise ValueError(f"{sid}: {key}缺失或与A记录冲突")
            row[key] = value
        row["split"] = normalized_split(row.get("split"))
        if row.get("image_path"):
            image = Path(row["image_path"])
            if not image.is_absolute():
                image = source.parent / image
            row["image_path"] = os.path.relpath(image.resolve(), target.parent)
        result.append(row)
    # 写出前校验，避免失败后留下看似可用的适配文件。
    validate_manifest_records(result)
    write_jsonl(target, result)
    return len(result)


def validate_training_manifests(train_path, validation_path):
    """所有检查在加载B/C之前进行；未知标签拒绝，不强制改为正常。"""
    train, validation = (load_jsonl(p, unique_key="sample_id") for p in (train_path, validation_path))
    for path, rows, expected in ((train_path, train, "fusion_train"),
                                  (validation_path, validation, "validation")):
        if not rows or any(normalized_split(r.get("split")) != expected for r in rows):
            raise ValueError(f"{expected}清单为空或包含错误划分；禁止base_train/test参与融合训练或选阈值")
        validate_manifest_records(rows)
        for row in rows:
            if type(row.get("label")) is not int or row["label"] not in {0, 1}:
                raise ValueError(f"{row['sample_id']}: 训练/验证label须是0/1，未知标签不得强行补0")
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"{expected}必须有正常与风险两类")
    for key in ("sample_id", "image_group_id", "request_group_id"):
        overlap = {r[key] for r in train} & {r[key] for r in validation}
        if overlap:
            raise ValueError(f"训练/验证泄漏：{key}重复")
    texts = [{''.join(r['user_text'].casefold().split()) for r in rows if r['user_text'].strip()}
             for rows in (train, validation)]
    if texts[0] & texts[1]:
        raise ValueError("训练/验证存在完全相同的规范化请求，须由A确认分组")
    paths = []
    for path, rows in ((train_path, train), (validation_path, validation)):
        paths.append({str((Path(path).resolve().parent / r['image_path']).resolve()).casefold()
                      for r in rows if r.get('image_path')})
    if paths[0] & paths[1]:
        raise ValueError("训练/验证引用相同图片路径，须由A确认分组")
    return train, validation


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
