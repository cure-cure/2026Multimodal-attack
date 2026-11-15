# C 模块第四周实现结果与说明

## 1. 预测结果文件说明
本目录包含提供给 D 模块（融合模块）直接使用的 C 模块真实预测结果，基于项目自带的 `mock-001` 至 `mock-006` 样本。
- `predictions.rule_clf.jsonl`：全融合模式的预测结果，其中 `component_scores.user_text_score` 为 C 模块输出的真实分数。
- `compare.rule_clf.jsonl`：三种方案（仅图像、仅文字、完整融合）的对比结果。

## 2. 自主分类器训练指标
基于开源数据集 `ynyg/Unified-Prompt-Guard` 的中文/混合子集（测试集占 15%），模型使用自主实现的 TF-IDF + 朴素贝叶斯训练。
测试集评估结果如下：
- Precision (不安全类): 0.6980
- **Recall (不安全类): 0.8989 (89.89%)**
- F1-score (不安全类): 0.7858
- Overall Accuracy: 0.7417

*注：在安全检测任务中，Recall（召回率）是核心指标。89.89% 的召回率意味着 100 条风险样本中，C 模块能成功拦截 90 条。*

## 3. 给 D 同学的运行提示
如果你需要实时调用 C 模块或重新生成预测结果，请执行以下操作：
1. 安装依赖：`pip install ai-injection-guard datasets jieba scikit-learn joblib`
2. 生成本地模型：`$env:PYTHONPATH = "$PWD\src"; python train_text_classifier.py`
3. 批量运行：`python -m mm_jailbreak_guard batch ... --text-provider mm_jailbreak_guard.rule_classifier_provider:create_provider ...`

由于模型文件（`.joblib`）和输出目录（`outputs/`）在 `.gitignore` 中被排除，无法直接推送，因此将预测结果单独抽出放在本目录以供 D 联调。