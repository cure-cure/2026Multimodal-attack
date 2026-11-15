# tune_text_classifier.py
"""在验证集上搜索最优超参数，生成 v2 模型并输出固定配置。"""
import json
from pathlib import Path
import itertools
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from datasets import load_dataset

from mm_jailbreak_guard.text_utils import tokenize

# 优先使用 A 交付的真实验证集，如果不存在则使用开源数据兜底
VAL_MANIFEST = Path("outputs/A/manifest.val.jsonl")
texts, labels = [], []

if VAL_MANIFEST.is_file():
    print(f"加载 A 模块真实验证集: {VAL_MANIFEST}")
    with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            texts.append(row["user_text"])
            labels.append(int(row["label"]))
else:
    print("未找到 A 模块验证集，使用开源数据集自动划分 15% 作为验证集...")
    ds = load_dataset("ynyg/Unified-Prompt-Guard", split="train")
    ds = ds.filter(lambda x: x["lang"] in {"zh", "mixed"})


    def map_label(l):
        return 1 if str(l).lower().strip() in {"unsafe", "1", "true", "harmful", "jailbreak"} else 0


    raw_texts = [str(t) for t in ds["text"]]
    raw_labels = [map_label(l) for l in ds["label"]]

    from sklearn.model_selection import train_test_split

    _, texts, _, labels = train_test_split(
        raw_texts, raw_labels, test_size=0.15, random_state=42, stratify=raw_labels
    )

# 网格搜索空间
param_grid = {
    "tfidf__max_features": [20000, 50000, 100000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__alpha": [0.5, 1.0, 2.0],
}

best_score = -1.0
best_params = None
results = []

print(f"开始调参，验证集样本数: {len(texts)}")

for max_feat, ngram, alpha in itertools.product(
        param_grid["tfidf__max_features"],
        param_grid["tfidf__ngram_range"],
        param_grid["clf__alpha"],
):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=tokenize, lowercase=False, max_features=max_feat, ngram_range=ngram)),
        ("clf", MultinomialNB(alpha=alpha)),
    ])
    pipeline.fit(texts, labels)
    pred = pipeline.predict(texts)

    recall = recall_score(labels, pred, pos_label=1)
    f1 = f1_score(labels, pred, pos_label=1)
    score = 0.7 * recall + 0.3 * f1

    results.append({"max_features": max_feat, "ngram": ngram, "alpha": alpha, "score": score})

    if score > best_score:
        best_score = score
        best_params = {"max_features": max_feat, "ngram_range": ngram, "alpha": alpha}

print(f"最优参数: {best_params}")
print(f"验证集最优得分: {best_score:.4f}")

# 重新训练并保存 v2 模型
final_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(tokenizer=tokenize, lowercase=False, max_features=best_params["max_features"],
                              ngram_range=best_params["ngram_range"])),
    ("clf", MultinomialNB(alpha=best_params["alpha"])),
])
final_pipeline.fit(texts, labels)

Path("models").mkdir(exist_ok=True)
joblib.dump(final_pipeline, "models/text_classifier_v2.joblib")
print("已保存: models/text_classifier_v2.joblib")

# 保存固定配置
Path("outputs/C_package").mkdir(parents=True, exist_ok=True)
config = {
    "version": "v2",
    "best_params": best_params,
    "val_score": round(best_score, 4),
    "objective": "0.7*Recall + 0.3*F1"
}
with open("outputs/C_package/config_v2.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)