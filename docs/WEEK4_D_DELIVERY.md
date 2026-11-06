# D 第四周交付与复现说明

依据课程《2026多媒体大作业题目.docx》及工作区最新版《题目十分工与DDL计划.md》。第四周（11月2—8日）要求使用模拟结果完成界面、固定融合、批量预测、自主融合训练代码及缺失/异常处理。D负责系统集成，不替A整理数据、不替B实现CIDER/OCR、不替C实现文本分类。

本次D交付基于A/C已合并成果（基线提交9be28dc），版本以Git提交记录为准；A/C源文件保持原样。生成的模拟配置、预测和运行日志保留在本地，不包含在代码交付中。

## 已实现及依赖

| 要求 | 本地实现 | 仍需外部成果 |
| --- | --- | --- |
| 简单融合和自主可训练融合 | 固定加权＋手写逻辑回归，两套配置可对照 | 算法对应课件须核验；真实融合参数待B/C稳定分数 |
| 融合训练隔离 | fusion_train更新参数，validation/val只选阈值，test/base_train拒绝进入训练入口 | A确认分组及真值；近重复语义分组仍由A负责 |
| 接入A数据 | adapt-manifest读取A的分组CSV生成D副本，保留源标签/划分，调整相对图片路径 | A交数据包及同版work/inputs.csv或相同结构的分组CSV |
| 接入B/C | Python提供方、离线JSONL、collect-evidence缓存及版本检查 | B模块/权重/配置，C模块运行依赖与权重及真实结果 |
| 系统与批量 | 三模式、固定/学习对照、耗时、证据贡献、错误提示、日志 | 真实图文端到端验收待第五周v1成果 |

## 一键模拟流程

在仓库根目录，Python3.10+：

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/check_d_workflow.py
python -m unittest discover -s tests -v
```

脚本只用examples/handoff内六条模拟记录，将融合训练和验证分别提取到outputs/D_mock，然后生成learned.json、comparison.jsonl、predictions.jsonl。它不生成攻击内容。模拟数据少且标签、分数人工构造，验证的是软件流程，任何结果都不能作为课程检测指标。

使用训练配置启动同一页面：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1 -ConfigPath outputs/D_mock/learned.json
```

浏览器打开http://127.0.0.1:8000，输入mock-001至mock-006。页面支持三模式及证据贡献；mock-006展示证据不足。离线模式按sample_id回放证据，新上传图片不会重新计算结果。真实检测必须加载B/C在线提供方。

## A的数据接入

A当前export_d_manifests生成的清单未附带组号；D不猜测组号，不修改A导出器或原始包。将A同一版本的work/inputs.csv作为分组侧表：

```powershell
python -m mm_jailbreak_guard adapt-manifest --manifest data/A/manifest.fusion_train.jsonl --groups-csv data/A/work/inputs.csv --output outputs/D_inputs/fusion_train.jsonl
python -m mm_jailbreak_guard adapt-manifest --manifest data/A/manifest.validation.jsonl --groups-csv data/A/work/inputs.csv --output outputs/D_inputs/validation.jsonl
```

路径均为示例，以A实际交付位置为准。侧表必须含sample_id/image_group_id/request_group_id。未知样本、冲突组号会拒绝；只将val别名规范为validation，不改变训练用途；源图片解析位置保持不变。若A未交组号，先记录阻断，不能自行编造组号绕过校验。测试清单可独立适配，不能补测试标签给D。

## B/C证据缓存

接入提供方后，按每个划分分别收集，以下以融合训练为例：

```powershell
python -m mm_jailbreak_guard collect-evidence --manifest outputs/D_inputs/fusion_train.jsonl --config configs/fusion_config.example.json --image-provider team_b.adapter:create_provider --text-provider mm_jailbreak_guard.qwen3guard_provider:create_provider --image-output outputs/B/fusion_train.jsonl --text-output outputs/C/fusion_train.jsonl
```

team_b为待交付模块占位名。C也可用rule_classifier_provider；依赖和权重由C说明并准备，D不下载或训练C模型。缓存按sample_id保留模块版本、OCR状态和失败警告；加载失败不自动视为正常。缺少B/C时可用已有离线文件进行框架调试，但不宣称真实端到端已完成。

## 训练、验证和固定融合对比

```powershell
python -m mm_jailbreak_guard train-fusion --train-manifest outputs/D_inputs/fusion_train.jsonl --validation-manifest outputs/D_inputs/validation.jsonl --image-evidence outputs/B/fusion_train.jsonl --text-evidence outputs/C/fusion_train.jsonl --validation-image-evidence outputs/B/validation.jsonl --validation-text-evidence outputs/C/validation.jsonl --baseline-config configs/fusion_config.example.json --output outputs/D/fusion.json
python -m mm_jailbreak_guard compare --manifest outputs/D_inputs/validation.jsonl --image-evidence outputs/B/validation.jsonl --text-evidence outputs/C/validation.jsonl --config outputs/D/fusion.json --baseline-config configs/fusion_config.example.json --output outputs/D/validation-comparison.jsonl
```

同一提供方/合并证据文件可省略validation专用证据参数。训练入口在调用B/C之前检查划分、标签、编号、组号和完全相同请求/图片路径。它不检查真实图片近重复或理解标签含义，不能代替A的质量检查。输出禁止覆盖输入文件。

学习配置包含两个模型：text_only与full；image_only始终使用B的图像分支阈值，绝不用整体label训练图像扰动判定。模型保存缺失填充值、缺失指示、系数、偏置、特征顺序、训练覆盖数、损失记录、阈值、B/C版本及输入文件SHA256。新增训练未覆盖的特征或更换上游版本时拒绝静默使用旧学习配置，提示重新训练。outputs内文件默认不上传。

compare输出schemes与fixed_baseline_schemes，供A统一评测。image_only对应image_adversarial_label，另外两种对应整体label。HoliSafe中未知的图像对抗标签不能当负例。最终测试须由A使用冻结版本执行，不通过测试效果回改参数。

## 11月9日系统框架演示顺序

1. 说明D当前用模拟证据验证系统，课件对应和真实训练尚待交付。
2. 运行一键模拟流程，展示训练/验证分开，配置可保存与加载。
3. 打开页面，对mock-002切换三模式，展示图像原始规则、文本证据和逻辑回归贡献。
4. 用mock-003展示OCR失败时其他分支仍可用，mock-006展示证据不足。
5. 展示comparison.jsonl内固定/学习两组输出，说明正式指标由A统一计算。

第二次由B主讲，D提供框架与系统演示材料；不提前声称第五周真实接入或第六周最终训练已完成。

## 本地验证记录

2026-11-07：全量47项测试通过，包含A原有数据管线测试及D新增测试。D测试覆盖手算首步梯度、损失下降、确定性重训、验证不更新参数/填充值、训练及测试隔离、组号/请求重复、未知标签、参数合法性、配置重载、缺失/版本变化/新特征拒判、A清单适配、证据缓存、固定/学习比较与批量一致性，以及两种配置的三模式HTTP接口。`scripts/check_d_workflow.py`运行成功，生成本地模拟配置与逐条输出。此记录只证明对应测试场景的软件行为，不是正式模型检测指标。

## 仍需交付或核验

课件核验、A/B/C确认、真实B图像与OCR接入、C真实模型运行、稳定分数上的最终融合训练、真实指标及失败分析。代码和模拟复现已具备，不能推导真实算法效果。
