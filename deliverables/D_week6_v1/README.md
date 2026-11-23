# D 第六周：融合训练与冻结候选交付

本目录是第六周候选交付入口，人员按 A、B、C、D 指代。D 负责统一接口、证据融合、批量预测和演示；本次使用 B、C 已交付的真实预计算证据完成融合训练，不改动 B/C 算法。

**新成员先下载 [完整运行包](D_week6_candidate_for_A.zip) 并解压，再阅读包内 README。** 包含代码、输入证据、训练/验证标签、冻结配置、逐条预测、评测入口和启动脚本，无需再私下索取这部分文件。本目录提供同版关键结果供 GitHub 直接浏览。第五周文件保留用于追溯；本次评测使用第六周版本。

## 已完成什么

| 项目 | 本次结果 |
| --- | --- |
| 上游版本 | B-2.0.0-stable-fixed-splits；C rule-baseline+tfidf-nb/v3 |
| 融合训练 | 808 条 fusion_train；手写逻辑回归，分别训练 text_only 与 full |
| 阈值选择 | 605 条 validation；按既有规则优先平衡准确率，再比较 F1 和距 0.5 的距离 |
| 训练参数 | 600 轮、学习率 0.2、L2 0.01；沿用既有默认值 |
| 对照 | 同证据固定融合与学习融合；image_only、text_only、full 三种模式 |
| 配置冻结 | logistic-dd04267d5f107864；测试推理前冻结，测试未用于选参 |
| 测试交接 | 609 条 test 的逐条预测；未计算测试指标，交由 A 使用私有真值统一评测 |
| 工程检查 | 59 项回归测试通过；7284 条 batch/compare 决策一致；6 条真实 HTTP 请求与批量一致；独立运行包启动通过 |

`text_only` 使用 C 的文字证据；`full` 加入 B 的图像侧证据和关联特征。`image_only` 检测图像对抗扰动，和整体风险的评测目标不同。HoliSafe 没有图像对抗扰动轴真值，不能使用整体风险标签评价该模式。

## 文件从哪里看

| 文件 | 用途 |
| --- | --- |
| [D_week6_candidate_for_A.zip](D_week6_candidate_for_A.zip) | 自包含运行/评测/复现包；解压后从包根目录执行命令 |
| [learned_fusion.json](learned_fusion.json) / [fixed_fusion.json](fixed_fusion.json) | 学习融合与固定基线配置 |
| [FREEZE_RECORD.json](FREEZE_RECORD.json) | 冻结时间、配置指纹、阈值、版本及数据数量 |
| [results/test/](results/test/) | 给 A 的 609 条测试预测；learned.* 与 fixed.* 分别为学习和固定融合 |
| [results/validation/](results/validation/) | 605 条验证预测及模式对照 |
| [validation_metrics.csv](results/validation_metrics.csv) / [JSON](results/validation_metrics.json) | 验证指标，参与阈值选择，不能当作独立测试成绩 |
| [validation_failure_cases.jsonl](results/validation_failure_cases.jsonl) | 验证集错例，供分析误报与漏报 |
| [run_log.jsonl](results/run_log.jsonl) | 训练、预测与运行记录 |
| [source_provenance.json](source_provenance.json) | B 交付提交号、输入文件及 C 交接包指纹、标签来源 |
| [verification.json](verification.json) / [completion_status.json](completion_status.json) | 已完成验证及尚未验收项 |
| [D_REPORT_CHAPTER.md](D_REPORT_CHAPTER.md) | 方法、结果、困难和复现说明；其中 candidate/ 路径相对于解压后的运行包根目录 |
| [SHA256SUMS.txt](SHA256SUMS.txt) | 本目录文件校验值；ZIP 内另有包内文件校验表 |

自主算法代码见 [learned_fusion.py](../../src/mm_jailbreak_guard/learned_fusion.py)；统一输入输出见 [接口说明](../../docs/INTERFACE_SPEC.md)。代码使用 Python 自行计算梯度、类别平衡损失和 L2，未调用现成分类器或自动微分。缺失值的填补统计仅由融合训练集计算，并附缺失标记。

## A 如何使用

1. 下载并解压 ZIP。包根目录有 `src/`、`candidate/`、`evaluate_predictions.py`、`start_demo.ps1` 和详细 README。
2. 主结果为 `candidate/results/test/learned.full.jsonl`，文字对照为 `learned.text_only.jsonl`；固定融合对照为 `fixed.full.jsonl` 与 `fixed.text_only.jsonl`。每份包含 609 条；不要把第五周输出和第六周配置混用。
3. 使用 A 自己的母表目录评测。以下命令在解压后的包根目录运行，替换实际路径，并使用新的输出目录：

```powershell
python -B evaluate_predictions.py --work-dir "A的真实评测目录" --predictions candidate/results/test/learned.full.jsonl --module D --target overall_risk --mode full --split test --run-id D-week6-Cv3-learned --frozen-config candidate/evaluation_frozen.full.json --data-kind real --out outputs/A_D_week6_full
```

文字对照同时替换为 `learned.text_only.jsonl`、`--mode text_only` 和 `evaluation_frozen.text_only.json`。以上 `evaluation_frozen.*` 专用于本次 learned 结果，不能直接套到 fixed 对照。固定融合配置见 `fixed_fusion.json`，A 需为固定对照单独记录对应冻结信息。

预测文件已经带有冻结阈值下的判定。评测配置的 `score_field`、`threshold` 为 null，表示评测这些既定判定；实际学习阈值保存在学习配置和冻结记录中。测试真值不在本包中，不用测试指标重新选择参数。融合训练与验证标签来自 B 公开交付中保留的 A 整体风险标签，A 可按来源记录与母表核对。

## 演示与复现

Python 3.10+；D 融合与回放不需要第三方 Python 依赖。在解压后的包根目录启动：

```powershell
powershell -ExecutionPolicy Bypass -File start_demo.ps1
# Python 未在 PATH 中时：
powershell -ExecutionPolicy Bypass -File start_demo.ps1 -PythonPath "C:\path\to\python.exe"
```

访问 http://127.0.0.1:8000，输入清单已有编号并切换三种模式。`S0259059B680F` 是有 OCR 样例，`S03CEC96B0D85` 是无 OCR 样例。OCR 是从图片中识别出的文字；页面展示预计算证据与融合贡献，耗时只覆盖证据回放和融合。重训与固定/学习对照命令见 ZIP 内 README；输出写入新目录，避免覆盖冻结交付。

## 结果如何理解，哪些仍待确认

验证集 605 条，其中风险 504 条、正常 101 条。学习完整融合 F1 约 0.6057，固定完整融合约 0.5101，固定仅文字约 0.6504。学习完整融合优于固定完整融合，但低于固定仅文字，不能宣称融合全面领先。验证集用于选阈值，这些数值只作诊断；详细准确率、召回率、误报率和平衡准确率见指标文件。多数类基线准确率为 504/605≈0.8331，当前系统仍存在明显漏报。

- **课件对应待核验**：算法的课件标题、章节和页码尚未提供，配置保留 `pending_verification`，不能据此宣称课程全部要求已验收。
- **实时新图文推理待验收**：目前验证的是 B/C 真实离线证据回放与 D 融合。包中没有原图或大模型权重，上传新图片不会产生已验收的全链路实时识别结果。
- **最终测试成绩待 A 统一评测**：D 已交付冻结预测，没有使用 A 的私有测试标签计算或优化测试效果。

本次没有修改 A/B/C 模块，也没有合并到 main。第五周旧结果不代表本次冻结候选。
