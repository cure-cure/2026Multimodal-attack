"""D 的交接校验；只读检查，不修改 A 的标签或划分。"""
from collections import Counter
from .contracts import ImageEvidence, TextEvidence
from .io import load_jsonl

SPLITS = {"base_train", "fusion_train", "validation", "test"}
REQUIRED_FIELDS = ("sample_id", "image_path", "user_text", "split", "source",
                   "image_group_id", "request_group_id")


def validate_handoff(manifest, image_path=None, text_path=None):
    records = load_jsonl(manifest, unique_key="sample_id")
    if not records:
        raise ValueError("交接清单为空")
    owners = {}
    for row in records:
        missing = [key for key in REQUIRED_FIELDS if key not in row]
        if missing:
            raise ValueError(f"{row.get('sample_id', '<unknown>')}: 缺少必需字段 {', '.join(missing)}")
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid.strip():
            raise ValueError("sample_id 必须是非空字符串")
        if not isinstance(row["user_text"], str):
            raise ValueError(f"{sid}: user_text 必须是字符串，允许空字符串")
        image = row["image_path"]
        if image is not None and (not isinstance(image, str) or not image.strip()):
            raise ValueError(f"{sid}: image_path 必须是非空路径字符串或 null")
        if not isinstance(row["source"], str) or not row["source"].strip():
            raise ValueError(f"{sid}: source 必须是非空字符串")
        split = row.get("split")
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError(f"{sid}: split 必须是 {sorted(SPLITS)}")
        for label in ("label", "harmful_label", "jailbreak_label", "image_adversarial_label"):
            value = row.get(label)
            if value is not None and (type(value) is not int or value not in {0, 1}):
                raise ValueError(f"{sid}: {label} 必须是 0/1/null")
        for key in ("image_group_id", "request_group_id"):
            group = row.get(key)
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f"{sid}: 缺少 {key}，由 A 提供分组")
            token = (key, group)
            if token in owners and owners[token] != split:
                raise ValueError(f"跨划分泄漏：{key}={group}")
            owners[token] = split
    ids = {r["sample_id"] for r in records}
    for path, cls in ((image_path, ImageEvidence), (text_path, TextEvidence)):
        if path is None:
            continue
        evidence = load_jsonl(path, unique_key="sample_id")
        found = {r["sample_id"] for r in evidence}
        if found != ids:
            raise ValueError(f"{path}: 编号不一致；缺少 {sorted(ids-found)}；多余 {sorted(found-ids)}")
        for row in evidence:
            cls.from_dict(row)
    return {"status": "valid", "sample_count": len(records),
            "split_counts": dict(Counter(r["split"] for r in records)),
            "note": "仅校验字段和已提供的分组；真实标签、相似样本分组和来源由 A 审核"}
