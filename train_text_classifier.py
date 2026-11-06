"""使用开源中文安全数据，自主实现 TF-IDF + 朴素贝叶斯分类器。"""
from pathlib import Path
import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from mm_jailbreak_guard.text_utils import tokenize

# 加载数据集并筛选中文/混合样本
ds = load_dataset("ynyg/Unified-Prompt-Guard", split="train")
ds = ds.filter(lambda x: x["lang"] in {"zh", "mixed"})
texts = [str(t) for t in ds["text"]]

# 兼容数据集中 'safe' / 'unsafe' 等字符串标签
def map_label(l):
    l = str(l).lower().strip()
    return 1 if l in {"unsafe", "1", "true", "harmful", "jailbreak"} else 0

labels = [map_label(l) for l in ds["label"]]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.15, random_state=42, stratify=labels
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        tokenizer=tokenize,
        lowercase=False,
        max_features=50000,
        ngram_range=(1, 2),
    )),
    ("clf", MultinomialNB(alpha=1.0)),
])

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
print(classification_report(y_test, pred, digits=4))

Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/text_classifier_v1.joblib")
print("saved: models/text_classifier_v1.joblib")