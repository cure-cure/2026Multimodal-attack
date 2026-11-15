# D 第五周固定融合与交接说明

本周按最新版分工做首次真实联调。D 负责对齐证据、固定融合、批量预测、日志和演示。A 负责评测，B 和 C 各自维护模块及输出。计划原定接入第四周 v1，本次实际使用已取得并核验的 B `B-1.2.0-cider-platt-v1` 与 C `rule-baseline+tfidf-nb/v2`，输出仍是 D 首版固定融合系统。第六周才训练最终融合器、在验证集选择阈值。

## 真实运行结果

- [给 A 的预测与日志](../deliverables/D_week5_v1/)提供三份逐条预测、逐样本日志、运行汇总和输入来源记录，也可下载[完整交付包](../deliverables/D_week5_v1/D_week5_v1_for_A.zip)。运行使用 B 分支 `d588eb5` 随证据交付的 HoliSafe 分组清单，它来源于 A 的主数据，按原有 `validation` 选取 605 条。C 使用组内交接包里的 `pure_c_val_v2.jsonl`。B、C 和清单的样本编号完全一致。复跑所用的原始输入单独保存在本地的 `outputs/D_week5_inputs/`。
- 三种模式各生成 605 条判定，完整融合 497 条判正常、108 条判风险，拒判 0 条。OCR 有文字的 263 条使用 OCR／组合分数，无可用 OCR 的 342 条继续用图像和用户文字分数。这里是预测分布与覆盖情况，准确率、误报与漏报以 A 的评测为准。
- B 的 `visual_score` 表示图像扰动风险，按其校准报告生成，与整体图文风险真值的含义不同。D 沿用预设权重和 0.5 阈值；未用融合训练集拟合、未调整阈值，也未运行主测试集或 12 条校园补充测试集。
- 本次从 B、C 对真实图文预先计算的证据开始运行，计时仅覆盖证据读取和 D 的判定。演示展示两路证据及融合结果；新上传图片的实时检测需要可运行的 B、C 模块和模型，原图预览则需要主包图像文件。

## 需要的输入

| 来源 | 第一次真实运行所需 |
| --- | --- |
| A | 主数据的 `manifest.validation.jsonl`（`split` 为 `val` 或 `validation`），包含原样本编号及图片／请求组号。保留原包目录结构。缺组号时使用 A 的 v1.1 补丁，或用同版分组表运行现有 `adapt-manifest` 生成 D 副本。 |
| B | 同一验证集的 `ImageEvidence` JSONL，含 `sample_id`、模块版本、OCR 状态和可用图像证据。完整融合使用 B 按统一接口提供的 `[0,1]` `visual_score`。 |
| C | 同一验证集的 `TextEvidence` JSONL，或能加载的 C 模块及所需配置、权重。第四周 v1 可以只提供 `user_text_score`。如使用离线 OCR 或组合文字分数，须由 C 确认使用的是本次 B 的 OCR 缓存。 |

按 `sample_id` 对齐，文件顺序可以不同。B、C 的离线文件须对应同一划分，逐条失败也应保留带不可用状态的记录。`deliverables/predictions.rule_clf.jsonl` 是 D 输出格式，且仅含模拟编号，不能当作真实 C 证据输入。12 条校园自采正常样本是独立补充测试集，保留到配置冻结后单独使用。

## 本地批量运行

在仓库根目录运行。路径以组内实际交接为准。

```powershell
$env:PYTHONPATH = "$PWD\src"
python -B -m mm_jailbreak_guard week5-batch `
  --manifest <A主验证集清单> --image-evidence <B验证集证据JSONL> `
  --text-evidence <C验证集证据JSONL> --config configs/fusion_config.week5_fixed.json `
  --data-kind real --run-id D-week5-v1 --output-dir outputs/D_week5_v1
```

有可运行的 C 时，可用 `--text-provider mm_jailbreak_guard.rule_classifier_provider:create_provider` 替换 `--text-evidence`，以 C 确认的入口为准。程序每条只获取一次 B、C 证据，随后用同一份证据计算三个模式。配置沿用第四周示例的权重与阈值，作为预先确定、未校准的首次集成基线。真实入口限主验证集，不用测试集调参，也不会覆盖现有输出目录。

B 的原始 CIDER 统计量、比较方向和阈值用于 `image_only`。B 的部分导出器会有意将 `visual_score` 留空，D 不自行把原始统计量换算成融合分数。这种情况下仍可保留独立图像判定和文字判定，但 `full` 输出为空并记录原因所对应的状态，避免把缺少视觉分数的结果交作完整融合。OCR 失败时，如果有效图像分数和用户文字分数齐全，仍可以融合。

### 输出交给 A

- `predictions.image_only.jsonl` 对应图像对抗目标，`predictions.text_only.jsonl` 与 `predictions.full.jsonl` 对应整体风险目标。HoliSafe 主包的图像对抗真值未知，`image_only` 可展示或用于检查覆盖，不能在该主包上报图像对抗准确率。
- `run_log.jsonl` 记录逐条耗时、可用性、OCR 状态、视觉／文本分数是否实际参与、警告计数和模块版本。`run_summary.json` 汇总判定、拒判、输入文件指纹及固定配置。`partial_coverage` 表示部分样本或模式没有有效判定，须一并交代覆盖率。输出不附原始请求、OCR 文字或标签。
- A 用现有评测工具读取对应文件，例如指定 `--module D --target overall_risk --mode full --split val --run-id D-week5-v1 --data-kind real`。A 使用自己的真值计算指标，不根据本批预测重新选阈值。

## 分支演示和模拟验收

在仓库根目录运行以下命令，再打开 `http://127.0.0.1:8000`。输入验证集样本编号（页面默认填一条），依次切换「仅图像」「仅文字」「完整融合」。服务器按编号读取同一样本的原始文字及 B、C 的预计算证据，页面会明确标为回放。浏览器和批量使用同一套判定，缺少视觉或文本风险分数时，完整融合显示证据不足。

```powershell
$env:PYTHONPATH = "$PWD\src"
$validation = 'outputs/D_week5_inputs/validation'
python -B -m mm_jailbreak_guard web --config configs/fusion_config.week5_fixed.json `
  --replay-manifest "$validation/manifest.validation.jsonl" `
  --image-evidence "$validation/image_evidence.validation.jsonl" `
  --text-evidence "$validation/text_evidence.validation.jsonl" `
  --records runtime/D_week5_demo.jsonl
```

```powershell
python -B -m mm_jailbreak_guard week5-batch `
  --manifest examples/handoff/manifest.jsonl --image-evidence examples/handoff/image_evidence.jsonl `
  --text-evidence examples/handoff/text_evidence.jsonl --config configs/fusion_config.week5_fixed.json `
  --data-kind synthetic --run-id D-week5-smoke --output-dir outputs/D_week5_smoke
python -B -m unittest discover -s tests -v
```

模拟输出只用于检查三种输出、A 的读取格式、OCR 失败和证据不足。自主逻辑回归代码通过现有测试，课程课件对应核对仍单独保留。
