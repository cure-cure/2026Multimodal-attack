# A 第四周评测程序与接口修复

本说明对应组长计划中 A 的“完成自主指标脚本”。实现文件为 `evaluate_predictions.py`，指标由 Python 标准库计数及公式计算，不调用 sklearn 等库的指标函数。代码由 AI 辅助编写；成员 A 需理解、复核并能解释计算过程。它不替代 B/C/D 必须完成的自主课件算法。

## 真值与任务必须一致

| 被评测结果 | target 参数 | 真值列 | 当前主包能否计算 |
| --- | --- | --- | --- |
| B 图像对抗、D image_only | image_adversarial | image_adv_label | 不能：全部未知 |
| C 用户文字有害内容 | harmful_content | harmful_content_label | 可以，但需对应目标的真实预测 |
| C 整体风险基线、D full/text_only | overall_risk | overall_risk_label | 可以，但需对应目标的真实预测 |
| 绕过约束意图 | jailbreak_intent | jailbreak_intent_label | 不能：全部未知 |
| 提示注入 | prompt_injection | prompt_injection_label | 不能：全部未知 |

C 的用户文字分支如果目标是 query safety，使用 harmful_content；如果明确训练为整体风险基线，可以用 overall_risk，但须说明缺少视觉上下文的限制。两种目标不能在同一列中直接比较优劣。B 的图像异常分数不能替代整体风险真值。

## 输入和对齐

- 默认读取 `work/inputs.csv`、`master_labels.csv`、`splits.csv`，按 sample_id 联结，不依赖文件顺序。
- B CSV 默认读取 `image_pred`，C CSV 默认读取 `text_pred`。
- D JSONL 读取 `prediction/risk_score/mode`；兼容旧 CSV 的 `fusion_pred/fusion_score/fusion_mode`。D `compare` 的嵌套 `schemes` 也支持，通过 `--mode` 选取一种模式。
- C 仅提交分数时，可用 `--score-field user_text_score --threshold 0.5`。这里的 0.5 只是命令示例，实际须使用已在训练/验证阶段确定的阈值。程序不会选阈值。D/CIDER 使用已提交的判定，不在 A 侧另选阈值。
- 重复编号、数据集外编号、非法标签/分数、组号跨划分、冲突字段会报错；输入可以包含其他 split 的预测，程序只评指定 split 并记录忽略数量。
- 空值/null 是未知或拒判，不是 0。没有有效标签时状态为 `no_valid_labels`；没有可用预测时为 `no_predictions`。两者都不能成为正式实验成绩。
- label_status 为 pending/review 的真值不参与正式指标。标签缺失、行缺失、明确拒判分别计数。OCR 失败不妨碍有效的用户文字分支，但不能用于伪造 OCR 分支预测。

## 指标口径

正类为 1。TP 为风险判风险，TN 为正常判正常，FP 为正常判风险，FN 为风险判正常。

| 指标 | 手写公式 |
| --- | --- |
| accuracy | (TP+TN)/(TP+TN+FP+FN) |
| precision | TP/(TP+FP) |
| recall | TP/(TP+FN) |
| F1 | 2TP/(2TP+FP+FN) |
| FPR | FP/(FP+TN) |
| FNR | FN/(FN+TP) |
| specificity | TN/(TN+FP) |
| balanced_accuracy | (recall+specificity)/2，两类分母均非零时计算 |

任何分母为零的指标写为 JSON null、CSV 空值，不补成 0 或 1。F1 直接用计数公式：存在正类而一个都没检测到时可以为 0；全为正确负例时分母为零，F1 未定义。

混淆矩阵仅涵盖“对应标签已核实且预测可用”的样本，因此必须同时公布预测覆盖率、各真实类别覆盖率、缺失数及拒判数。`correct_fraction_of_all_eligible` 是“全部有效真值中实际答对的比例”，用于展示覆盖损失，不冒充有效预测子集的 accuracy。类别不平衡时，必须连同多数类基线和 FPR/FNR 报告，不能只看 accuracy。

## PowerShell 使用示例

在项目根目录运行。以下真实预测文件目前尚未交付；文件名为约定示例，不应创建虚假内容来满足路径。

```powershell
# C 用户文字有害内容预测，使用 val 诊断
.\.venv\Scripts\python.exe -B evaluate_predictions.py --predictions "received\C_val.csv" --module C --target harmful_content --split val --run-id C-v1 --data-kind real --out "output\week4_evaluation\C_v1_val"

# D 普通 batch 或嵌套 compare JSONL，均按 full 模式评测
.\.venv\Scripts\python.exe -B evaluate_predictions.py --predictions "received\D_val.jsonl" --module D --target overall_risk --mode full --split val --run-id D-v1 --data-kind real --out "output\week4_evaluation\D_v1_full_val"

# 只收到 C 的用户文字分数时（阈值应替换成已确定的值）
.\.venv\Scripts\python.exe -B evaluate_predictions.py --predictions "received\C_val.jsonl" --module C --target harmful_content --score-field user_text_score --threshold 0.5 --split val --run-id C-v1 --data-kind real --out "output\week4_evaluation\C_score_v1_val"

# 回归测试只使用临时合成数据
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

`--split validation` 等价于本地 val。base_train/fusion_train 的结果被标为训练诊断，不能作为独立测试成绩。输出目录已存在时停止，换新运行目录以保留历史。

最终 test 评测必须另加 `--frozen-config 配置文件.json`。该文件需要真实填写以下结构，并与命令对应：

```json
{
  "frozen": true,
  "run_id": "C-v1",
  "module": "C",
  "target": "harmful_content",
  "mode": null,
  "score_field": null,
  "threshold": null,
  "model_version": "填写已冻结模型版本及配置位置",
  "frozen_at": "填写实际冻结时间",
  "test_used_for_selection": false
}
```

这是结构示例，不是已冻结声明。使用 score-field 时要填写其名字和阈值；D 要填写 mode。脚本校验字段并记录配置文件、输入、预测和评测代码 SHA-256，但无法证明成员训练过程中没有看过公开测试标签；仍需保留训练来源和配置记录。

## 输出与私有边界

`summary.json` 保存完整口径和版本指纹；`metrics.csv` 包含指标与覆盖率；`confusion_matrix.csv` 为二维矩阵。`sample_results_PRIVATE.csv`、`failure_cases_PRIVATE.csv` 含逐样本真值，A 本地保管，不能连同代码公开上传。真实失败原因由 B/C/D 分析后汇总。

`output/week4_evaluation/synthetic_smoke/` 中如有结果，只是合成数据的程序验证，不是本项目实验成绩。目前没有据此声称真实模型的准确率或 F1。

