# D 第六周本地候选系统

本包完成 808 条融合训练、605 条验证选阈值、固定与学习融合对照，以及冻结后 609 条测试预测。
运行版本为 B-2.0.0-stable-fixed-splits、C rule-baseline+tfidf-nb/v3。
本包保存图文样本的预计算证据，运行从证据回放开始。时间仅覆盖回放和融合。
模型编号为 logistic-dd04267d5f107864，详情见 candidate/FREEZE_RECORD.json。

## 启动演示

Windows 在包根目录运行：

    powershell -ExecutionPolicy Bypass -File start_demo.ps1

Python 未在 PATH 时：

    powershell -ExecutionPolicy Bypass -File start_demo.ps1 -PythonPath <你的python.exe绝对路径>

打开 http://127.0.0.1:8000，输入清单中的样本编号并切换三种模式。
S0259059B680F 有 OCR，S03CEC96B0D85 无 OCR。可查看图片侧 CIDER、用户文字、OCR 和融合贡献。
本包不含原图或大模型权重，新上传图文的实时 B/C 推理尚未验收。

## 验证结果

下表都是参与选择阈值的验证集结果，不能当作独立测试成绩。
标签来自 B 公开交付保留的 A labels_for_C.csv、labels_for_D.csv 整体风险标签。
来源、分组、输入指纹见 candidate/source_provenance.json，A 可使用自己的母表复核。

| 算法 | 模式 | 准确率 | 精确率 | 召回率 | F1 | 误报率 | 平衡准确率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | text_only | 0.5488 | 0.9170 | 0.5040 | 0.6504 | 0.2277 | 0.6381 |
| fixed | full | 0.4380 | 0.9316 | 0.3512 | 0.5101 | 0.1287 | 0.6112 |
| learned | text_only | 0.4413 | 0.9826 | 0.3353 | 0.5000 | 0.0297 | 0.6528 |
| learned | full | 0.5157 | 0.9414 | 0.4464 | 0.6057 | 0.1386 | 0.6539 |

验证集 504 条风险、101 条正常，多数类基线准确率为 504/605=0.8331。
完整学习融合 F1 高于固定融合，但低于仅文字固定融合。当前结果体现融合方法仍有局限。
训练超参沿用预先实现的默认值，600 轮、学习率 0.2、L2 0.01。
阈值使用现有规则先最大化平衡准确率，再比较 F1 及距 0.5 的距离，未为了 F1 反复换方案。
text_only 阈值为 0.5727544384264274，full 为 0.553103125411251。
固定融合是两种模型共同的对照，图像模式使用 B 的原始 CIDER 阈值。
结果和失败样本在 candidate/results/validation/ 与 candidate/results/validation_failure_cases.jsonl。

## 给 A 的最终评测输入

- candidate/results/test/learned.full.jsonl 对应 D 整体风险主结果。
- candidate/results/test/learned.text_only.jsonl 是学习文本对照。
- candidate/results/test/fixed.full.jsonl、fixed.text_only.jsonl 是固定融合对照。
- image_only 对应图像对抗扰动，HoliSafe 没有该轴真值，不使用整体标签评测该模式。
- candidate/results/test/comparison.jsonl 包含同证据各模式及固定/学习对照。
- 测试清单和证据没有整体风险真值。本次没有计算测试指标，也没有接触 A 的私有测试标签。

A 在包根目录运行以下示例，work-dir 替换为 A 的实际母表目录，输出须用新目录。

    python -B evaluate_predictions.py --work-dir <A的真实评测目录> --predictions candidate/results/test/learned.full.jsonl --module D --target overall_risk --mode full --split test --run-id D-week6-Cv3-learned --frozen-config candidate/evaluation_frozen.full.json --data-kind real --out outputs/A_D_week6_full

文字对照改用 learned.text_only.jsonl、--mode text_only、evaluation_frozen.text_only.json。
固定融合的验证结果和测试预测也已保存，评测时使用其对应配置和阈值。
A 最终评测后如发现程序错误，按分工记录修复原因，不以测试效果选择参数。

## 可复现训练与对照

在包根目录运行，将输出写入新目录避免覆盖冻结结果。

    $env:PYTHONPATH = "$PWD\src"
    New-Item -ItemType Directory -Path outputs/retrain
    python -B -m mm_jailbreak_guard train-fusion --train-manifest candidate/inputs/manifests/fusion_train.jsonl --validation-manifest candidate/inputs/manifests/validation.jsonl --image-evidence candidate/inputs/B/image_evidence.fusion_train.jsonl --text-evidence candidate/inputs/C/fusion_train.jsonl --validation-image-evidence candidate/inputs/B/image_evidence.validation.jsonl --validation-text-evidence candidate/inputs/C/validation.jsonl --baseline-config candidate/fixed_fusion.json --epochs 600 --learning-rate 0.2 --l2 0.01 --output outputs/retrain/learned.json
    python -B -m mm_jailbreak_guard compare --manifest candidate/inputs/manifests/validation.jsonl --image-evidence candidate/inputs/B/image_evidence.validation.jsonl --text-evidence candidate/inputs/C/validation.jsonl --config outputs/retrain/learned.json --baseline-config candidate/fixed_fusion.json --output outputs/retrain/validation-comparison.jsonl

算法配置哈希可能因输入路径记录变化而不同，同一输入特征和超参的模型数值应可重现。
单条 batch、compare 与 web 均加载同一学习配置。训练与验证分组、请求和图片路径隔离已检查。

## 自主代码与本次验证

D 手写逻辑回归位于 src/mm_jailbreak_guard/learned_fusion.py，核心为 train_logistic、select_threshold、train_bundle。
使用 Python 计算梯度、类别平衡损失和 L2，未调用现成分类器或自动微分。
训练集计算缺失均值并附缺失标记，验证集只选阈值。
配置记录每路覆盖和上游版本，版本变化、缺失或新出现未覆盖特征会提示证据不足。
课件标题、章节及页码仍需核对，candidate/learned_fusion.json 中保留 pending_verification。
本次 59 项回归测试通过，7284 条批量/比较记录一致，6 条真实 HTTP 模式请求一致。
这些检查验证工程行为。课程算法出处和真实图片实时推理仍需单独确认。

## 版本变更记录

旧第五周结果使用 B-1.2.0、C v2。本地 C v3 固定重跑保持旧 B 便于对照。
本次第六周候选改用 B 已冻结稳定版 B-2.0.0，所有训练、验证和测试 C 统一 v3。
C 修复英文空白 token 的原因由其交接材料说明。本包不修改 B/C 的模型算法。
本包对应 D 第六周候选交付。第五周结果保留用于追溯，第六周评测使用本包的冻结配置与版本。
