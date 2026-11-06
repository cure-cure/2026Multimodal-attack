"""D 自主实现：逻辑回归、缺失特征处理及验证阈值选择（课件对应待核验）。"""
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .fusion import FusionBundle, FusionEngine, MODE_FEATURES, _metrics

ALGORITHM = "D_manual_logistic_fusion_v1"
LEARNED_MODES = ("text_only", "full")


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class LogisticModel:
    features: list[str]
    means: list[float]
    weights: list[float]
    bias: float
    threshold: float
    loss_history: list[float]
    observed_counts: list[int] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.features)
        if not n or len(set(self.features)) != n or len(self.means) != n or len(self.weights) != 2*n:
            raise ValueError("逻辑回归特征或参数长度无效")
        if not all(type(x) in (int, float) and math.isfinite(x)
                   for x in [*self.means, *self.weights, self.bias, self.threshold, *self.loss_history]):
            raise ValueError("逻辑回归参数必须是有限数值")
        if not 0 <= self.threshold <= 1 or any(not 0 <= x <= 1 for x in self.means):
            raise ValueError("阈值和填充值必须位于[0,1]")
        if len(self.observed_counts) != n or any(type(c) is not int or c < 0 for c in self.observed_counts):
            raise ValueError("模型缺少有效训练特征覆盖计数")

    def vector(self, features):
        values = [features.get(name) for name in self.features]
        return [mean if value is None else value for value, mean in zip(values, self.means)] + [
            float(value is None) for value in values]

    def score(self, features):
        return sigmoid(self.bias + sum(w*x for w, x in zip(self.weights, self.vector(features))))


def train_logistic(examples, names, *, epochs=600, learning_rate=0.2, l2=0.01):
    """全批量梯度下降；均值、类别权重和梯度只来自融合训练集。"""
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs必须为正整数")
    if not math.isfinite(learning_rate) or learning_rate <= 0 or not math.isfinite(l2) or l2 < 0:
        raise ValueError("learning_rate须正数，l2须非负有限数")
    if {e.label for e in examples} != {0, 1}:
        raise ValueError("融合训练集必须同时包含正常与风险标签")
    means, counts_observed = [], []
    for name in names:
        values = [e.features[name] for e in examples if e.features.get(name) is not None]
        means.append(sum(values)/len(values) if values else 0.0)
        counts_observed.append(len(values))
    model = LogisticModel(list(names), means, [0.0]*(2*len(names)), 0.0, 0.5, [], counts_observed)
    vectors = [model.vector(e.features) for e in examples]
    counts = {y: sum(e.label == y for e in examples) for y in (0, 1)}
    class_weights = {y: len(examples)/(2*counts[y]) for y in counts}
    for epoch in range(epochs):
        gradient = [0.0]*len(model.weights)
        bias_gradient = loss = 0.0
        for example, vector in zip(examples, vectors):
            z = model.bias + sum(w*x for w, x in zip(model.weights, vector))
            probability = sigmoid(z)
            weight = class_weights[example.label]
            error = weight*(probability-example.label)
            bias_gradient += error
            for j, value in enumerate(vector):
                gradient[j] += error*value
            # 数值稳定的二元交叉熵，不截断概率。
            loss += weight*(max(z, 0)-example.label*z+math.log1p(math.exp(-abs(z))))
        if epoch == 0 or (epoch+1) % 50 == 0 or epoch == epochs-1:
            model.loss_history.append(loss/len(examples) + l2*sum(w*w for w in model.weights)/2)
        model.weights = [w-learning_rate*(g/len(examples)+l2*w)
                         for w, g in zip(model.weights, gradient)]
        model.bias -= learning_rate*bias_gradient/len(examples)
    model.__post_init__()
    return model


def select_threshold(model, validation):
    """只选择阈值，不更新系数、偏置或填充值。"""
    if {e.label for e in validation} != {0, 1}:
        raise ValueError("验证集必须同时包含正常与风险标签")
    if any(e.features.get(name) is not None and count == 0
           for e in validation for name, count in zip(model.features, model.observed_counts)):
        raise ValueError("验证集出现训练从未覆盖的特征，须补齐训练证据后重新训练")
    scores = [model.score(e.features) for e in validation]
    candidates = sorted({0.0, 0.5, 1.0, *scores, *(
        (a+b)/2 for a, b in zip(sorted(set(scores)), sorted(set(scores))[1:]))})
    labels = [e.label for e in validation]
    def key(t):
        metrics = _metrics(labels, [int(s >= t) for s in scores])
        return metrics["balanced_accuracy"], metrics["f1"], -abs(t-0.5)
    return max(candidates, key=key)


def train_bundle(training, validation, baseline, *, provenance, **options):
    if {e.sample_id for e in training} & {e.sample_id for e in validation}:
        raise ValueError("融合训练和验证sample_id重复")
    models = {}
    for mode in LEARNED_MODES:
        names = MODE_FEATURES[mode]
        for example in [*training, *validation]:
            if not any(example.features.get(n) is not None for n in names if n.endswith("text_score")):
                raise ValueError(f"{example.sample_id}: 无有效文本证据，不能用于{mode}训练或验证")
        model = train_logistic(training, names, **options)
        model.threshold = select_threshold(model, validation)
        models[mode] = asdict(model)
    payload = {"algorithm": ALGORITHM, "schema_version": 1, "models": models,
               "baseline": baseline.as_dict(), "provenance": provenance,
               "training_options": options, "course_mapping": "pending_verification"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    return dict(payload, config_id="logistic-"+digest)


class LearnedFusionEngine:
    """三模式推理：图像沿用B规则；文字和完整融合使用独立训练的模型。"""
    def __init__(self, payload):
        if payload.get("algorithm") != ALGORITHM or payload.get("schema_version") != 1:
            raise ValueError("未知可训练融合配置")
        self.payload = payload
        self.bundle = SimpleNamespace(config_id=payload["config_id"])
        self.baseline = FusionEngine(FusionBundle.from_dict(payload["baseline"]))
        self.models = {mode: LogisticModel(**payload["models"][mode]) for mode in LEARNED_MODES}
        for mode, model in self.models.items():
            if model.features != list(MODE_FEATURES[mode]):
                raise ValueError(f"{mode}: 模型特征顺序不匹配")

    def fuse(self, request, image, text, mode="full"):
        base = self.baseline.fuse(request, image, text, mode)
        if mode not in self.models:
            return replace(base, fusion_config_id=self.bundle.config_id)
        model = self.models[mode]
        versions = self.payload["provenance"].get("module_versions", {})
        mismatches = [name for name, evidence in (("B", image), ("C", text))
                      if evidence.available and versions.get(name) and evidence.module_version not in versions[name]]
        unseen = [name for name, count in zip(model.features, model.observed_counts)
                  if count == 0 and base.component_scores.get(name) is not None]
        if base.prediction is None or mismatches or unseen:
            return replace(base, prediction=None, label_name="insufficient_evidence", risk_score=None,
                           threshold=model.threshold, effective_weights={}, risk_sources=(),
                           warnings=base.warnings + (("上游版本改变，须重新确认训练配置："+','.join(mismatches),) if mismatches else ())
                           + (("训练未覆盖此证据，须重新训练："+','.join(unseen),) if unseen else ()),
                           fusion_config_id=self.bundle.config_id)
        score = model.score(base.component_scores)
        prediction = int(score >= model.threshold)
        vector = model.vector(base.component_scores)
        names = model.features + [name+"_missing" for name in model.features]
        contributions = dict(zip(names, [w*x for w, x in zip(model.weights, vector)]))
        sources = self.baseline._select_sources(prediction, score, model.threshold,
                                               base.component_scores, image, text)
        if mode == "text_only":
            sources = [e for e in sources if e.source not in {"image", "visual_score", "cider"}]
        return replace(base, prediction=prediction, label_name="risk" if prediction else "normal",
                       risk_score=score, threshold=model.threshold, effective_weights={},
                       risk_sources=tuple(sources), fusion_config_id=self.bundle.config_id,
                       decision_rule={"algorithm": ALGORITHM, "bias": model.bias,
                                      "logit_contributions": contributions})


def load_fusion_engine(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("algorithm") == ALGORITHM:
        return LearnedFusionEngine(payload)
    return FusionEngine(FusionBundle.from_dict(payload))
