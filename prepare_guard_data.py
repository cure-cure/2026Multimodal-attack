# prepare_guard_data.py
"""从开源数据集自动构建 Qwen3Guard LoRA 微调数据。"""
import json
from pathlib import Path
from datasets import load_dataset

OUTPUT_PATH = Path("data/guard_train.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 1. 加载开源数据集，筛选中文/混合样本
ds = load_dataset("ynyg/Unified-Prompt-Guard", split="train")
ds = ds.filter(lambda x: x["lang"] in {"zh", "mixed"})
print(f"筛选后中文/混合样本数: {len(ds)}")


# 2. 标签映射与格式构建
def build_training_example(row):
    label = str(row["label"]).lower().strip()
    text = str(row["text"]).strip()

    if label in {"unsafe", "1", "true", "harmful", "jailbreak"}:
        # 对于二分类数据集，Unsafe 样本统一映射为 Unsafe
        # Categories 暂时留空，或根据数据源补充
        output = "Safety: Unsafe\nCategories: None"
    else:
        output = "Safety: Safe\nCategories: None"

    return {
        "instruction": "请对以下文本进行安全审核，并判断其严重性级别（Safe/Controversial/Unsafe）。",
        "input": text,
        "output": output,
    }


# 3. 构建并保存
records = [build_training_example(row) for row in ds]
# 打乱顺序
import random

random.seed(42)
random.shuffle(records)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"已生成 {len(records)} 条训练数据 -> {OUTPUT_PATH}")
print(f"其中 Unsafe 样本数: {sum(1 for r in records if 'Unsafe' in r['output'])}")
print(f"其中 Safe 样本数: {sum(1 for r in records if 'Safe' in r['output'])}")