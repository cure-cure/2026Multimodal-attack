# tune_text_classifier.py (最终修复版)
import json
from pathlib import Path
import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from mm_jailbreak_guard.text_utils import tokenize

VAL_MANIFEST = Path("outputs/A/team_dataset_v1/manifest.validation.jsonl")

print("正在加载开源数据作为独立训练集...")
# 如果你本地已经下载过，这里会非常快
ds = load_dataset("ynyg/Unified-Prompt-Guard", split="train")
ds = ds.filter(lambda x: x["lang"] in {"zh", "mixed"})
train_texts = [str(t) for t in ds["text"]]


def map_label(l):
    return 1 if str(l).lower().strip() in {"unsafe", "1", "true", "harmful", "jailbreak"} else 0


train_labels = [map_label(l) for l in ds["label"]]

print("使用最优参数在开源数据上训练 v2 模型...")
# 注意：这里将参数分开传，避免 TypeError
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        tokenizer=tokenize,
        lowercase=False,
        max_features=100000,
        ngram_range=(1, 2)
    )),
    ("clf", MultinomialNB(alpha=0.5)),
])

pipeline.fit(train_texts, train_labels)

Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/text_classifier_v2.joblib")
print("已保存真正的 v2 模型: models/text_classifier_v2.joblib")

# ================= 独立评估 =================
if VAL_MANIFEST.is_file():
    print(f"在 A 的独立验证集 {VAL_MANIFEST} 上评估...")
    val_texts, val_labels = [], []
    with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            # 注意：A 的 manifest 中，真值标签可能叫 label
            if "label" in row:
                val_texts.append(row["user_text"])
                val_labels.append(int(row["label"]))

    if val_texts:
        pred = pipeline.predict(val_texts)
        print("=== 独立验证集真实成绩 ===")
        print(classification_report(val_labels, pred, digits=4))

        # 将真实的独立评估配置写入 config_v2.json
        config = {
            "version": "v2",
            "trained_on": "open_source_Unified-Prompt-Guard",
            "eval_on": str(VAL_MANIFEST),
            "best_params": {"max_features": 100000, "ngram_range": [1, 2], "alpha": 0.5},
            "independent_val_f1": round(f1_score(val_labels, pred, pos_label=1), 4),
            "independent_val_recall": round(recall_score(val_labels, pred, pos_label=1), 4)
        }
        config_path = Path("outputs/C_package/config_v2.json")
        # outputs/ 不进 git，换机器时目录不存在，这里必须先建，否则模型存完才崩
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"真实的 {config_path} 已更新")
    else:
        print("警告：A 的验证集里没有找到 label 字段，无法进行独立评估。")
else:
    print("未找到 A 的验证集，跳过独立评估。")