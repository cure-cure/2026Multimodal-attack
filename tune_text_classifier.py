# tune_text_classifier.py
"""训练文本侧 TF-IDF + MultinomialNB 分类器。

v3 相对 v2 的唯一改动：分词器从 `tokenize` 换成 `tokenize_drop_space`。
jieba 对英文会吐出纯空格 token，导致 (1,2)-gram 里混进 'How  ' 这类噪声特征；
去掉后英文侧 bigram 才是有意义的词组。超参（max_features / ngram_range / alpha）
一个都没动 —— 在验证集上再调 alpha 属于拿测试集调参，会被质疑数据泄漏。
"""
import json
from pathlib import Path

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from mm_jailbreak_guard.text_utils import tokenize_drop_space

VAL_MANIFEST = Path("outputs/A/team_dataset_v1/manifest.validation.jsonl")

print("正在加载开源数据作为独立训练集...")
# 如果你本地已经下载过，这里会非常快
ds = load_dataset("ynyg/Unified-Prompt-Guard", split="train")
ds = ds.filter(lambda x: x["lang"] in {"zh", "mixed"})
train_texts = [str(t) for t in ds["text"]]


def map_label(l):
    return 1 if str(l).lower().strip() in {"unsafe", "1", "true", "harmful", "jailbreak"} else 0


train_labels = [map_label(l) for l in ds["label"]]

print("训练 v3 模型（drop-space 分词）...")
# 注意：这里将参数分开传，避免 TypeError
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        tokenizer=tokenize_drop_space,
        lowercase=False,
        max_features=100000,
        ngram_range=(1, 2)
    )),
    ("clf", MultinomialNB(alpha=0.5)),
])

pipeline.fit(train_texts, train_labels)

Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/text_classifier_v3.joblib")
print("已保存 v3 模型: models/text_classifier_v3.joblib")

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
        proba = pipeline.predict_proba(val_texts)[:, 1]
        print("=== 独立验证集真实成绩 ===")
        print(classification_report(val_labels, pred, digits=4))

        tn, fp, fn, tp = confusion_matrix(val_labels, pred, labels=[0, 1]).ravel()
        auc = roc_auc_score(val_labels, proba)
        fpr = fp / (fp + tn) if fp + tn else 0.0
        print(f"AUC = {auc:.4f}   FPR = {fpr:.4f}   (tp={tp} fp={fp} fn={fn} tn={tn})")

        # 将真实的独立评估配置写入 config_v3.json
        config = {
            "version": "v3",
            "trained_on": "open_source_Unified-Prompt-Guard",
            "eval_on": str(VAL_MANIFEST),
            "best_params": {"max_features": 100000, "ngram_range": [1, 2], "alpha": 0.5},
            "tokenizer": "tokenize_drop_space",
            "independent_val_auc": round(float(auc), 4),
            "independent_val_f1": round(f1_score(val_labels, pred, pos_label=1), 4),
            "independent_val_recall": round(recall_score(val_labels, pred, pos_label=1), 4),
            "independent_val_fpr": round(fpr, 4),
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        }
        config_path = Path("outputs/C_package/config_v3.json")
        # outputs/ 不进 git，换机器时目录不存在，这里必须先建，否则模型存完才崩
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"真实的 {config_path} 已更新")
    else:
        print("警告：A 的验证集里没有找到 label 字段，无法进行独立评估。")
else:
    print("未找到 A 的验证集，跳过独立评估。")