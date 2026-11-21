# Task B：图像侧证据、OCR 与 CIDER

本目录是课程任务 B 的公开可复现实现，只做检测与防御。它提供 CIDER 图文相似度变化、EasyOCR 中英文区域、高斯/频域辅助特征、图片哈希缓存、数据审计和 D `ImageEvidence` 适配。

本次按明确的仓库同步要求提交了 v2／第五周所需的派生逐样本结果；模型权重、原始受限图片和私有数据包仍不上传。更广泛的公开范围核对见 [B_PUBLICATION_SCOPE_REVIEW.md](reports/B_PUBLICATION_SCOPE_REVIEW.md)。聚合结果见 [双标签轴报告](reports/B_DUAL_AXIS_ANALYSIS.md)。

## 结论与标签边界

两个目标分开评估：

1. `image_adv_label`：CIDER 官方 clean/adversarial 数据，用于 B 的图片扰动专项评测。
2. `overall_risk_label`：HoliSafe 图文整体风险，仅用于探索 B 特征与广义安全的相关性。

| 目标 | 数据 | 结果 |
|---|---|---|
| 图片扰动排序 | CIDER validation | AUROC 0.9378，AP 0.9375 |
| 图片扰动排序 | CIDER test | AUROC 0.9679，AP 0.9720 |
| v2 validation 最优阈值 | validation | 阈值 0.0022773519，balanced accuracy 0.8963，F1 0.8931 |
| v2 冻结评估 | test | Accuracy/balanced accuracy 0.9055，Precision 0.9421，Recall 0.8641，F1 0.9014，FPR 0.0531 |
| 整体风险相关性 | HoliSafe 3,415 条有标签样本 | 各 B 信号 AUROC 约 0.48–0.54 |

限制：CIDER 官方 validation/test 共享 1 条请求、4 张图片，并形成 4 组完全相同的图文对。因此 test 既不是“完全请求留出”，也不是 image-holdout；这里只按“存在已量化重叠的官方划分迁移评测”报告，不证明对全新攻击图片的泛化。整体安全仍应由 B 图像证据与 C 文本证据交给 D 融合。

CIDER 的固定来源、实际权重版本和运行/冻结校验见 [`provenance/CIDER_SOURCE_ATTACHMENT.md`](provenance/CIDER_SOURCE_ATTACHMENT.md) 与 [`reports/B_CIDER_RUN_FREEZE_RECORD.json`](reports/B_CIDER_RUN_FREEZE_RECORD.json)。

## v2 与第五周稳定版交付

- 固定划分特征、OCR、证据、CIDER 预测、阈值报告和自主高斯／频域安全分类位于 `../artifacts/v2/`。
- TP/TN/FP/FN 演示、异常分析、实时复现结果和交付哈希位于 `../artifacts/week5/`。
- 完整结论见 [B_WEEKLY_COMPLETION_V2.md](reports/B_WEEKLY_COMPLETION_V2.md) 与 [B_WEEK5_IMAGE_CHAPTER.md](reports/B_WEEK5_IMAGE_CHAPTER.md)。
- 缓存无变化时直接复用；批量入口只运行缺失或明确要求重算的图像侧项，不改变 D 融合字段。

缓存交付校验与实时批量入口：

```bash
python team_b/scripts/build_week5_stable_delivery.py --verify
bash team_b/scripts/run_week5_batch.sh artifacts/week5/demo_manifest.jsonl outputs/week5_demo.jsonl
```

## 环境与上游代码

推荐 Python 3.10、CUDA 12.1、24GB GPU。先在仓库根目录取得固定版本 CIDER：

```bash
git clone https://github.com/PandragonXIII/CIDER.git CIDER
git -C CIDER checkout 3c08d7e2d52e27b47d1e428a030e8ffda5efae3e
cd team_b
bash scripts/setup_autodl.sh
source .venv/bin/activate
python scripts/download_weights_lite.py --output models \
  --llava-revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9
python run_b_fast.py check --config config.lite.json
```

## 运行

原始清单每行至少包含 `sample_id`、`image_path`、`user_text`：

```bash
python run_b_fast.py batch --config config.lite.json \
  --manifest /path/manifest.inputs.jsonl --output artifacts/cider.raw.jsonl
python b_pipeline.py features --dataset-zip /path/data.zip --output artifacts/features.jsonl
python b_pipeline.py ocr --dataset-zip /path/data.zip \
  --output artifacts/ocr.jsonl --cache-dir artifacts/ocr_cache
```

复现 CIDER 官方校准数据：

```bash
python scripts/prepare_cider_calibration.py --source-repo ../CIDER --output-parent data
python scripts/analyze_b_labels.py
python scripts/build_calibrated_evidence.py --include-splits validation \
  --output artifacts/image_evidence.validation.jsonl
python scripts/build_calibrated_evidence.py --include-splits test \
  --output artifacts/image_evidence.test.jsonl
```

`analyze_b_labels.py` 的双轴默认输入包含本地受限 HoliSafe 文件；公开仓库不提供这些输入，可通过命令行参数指定受控环境中的路径。

生成供 D 固定融合直接消费的 HoliSafe 全量图像证据：

```bash
python scripts/build_holisafe_fixed_fusion_evidence.py
```

输出位于 `../artifacts/image_evidence.fixed_fusion.all.jsonl`，实验参数和冻结测试指标见
`reports/B_VISUAL_SCORE_CALIBRATION.json`。脚本同时兼容独立 `team_b` 部署布局，
此时输出位于 `artifacts/`。CIDER 判定阈值只使用官方 `threshold_clean`；
`visual_score` 的 Platt 转换只拟合官方 `validation`，官方 `test` 仅作冻结评估；
程序不读取 HoliSafe 标签。该分数表示图像对抗风险概率，不是 HoliSafe 整体风险，
也不能单独替代多模态融合结论。


## 测试

```bash
python -m unittest discover -s team_b/tests -v
```

测试覆盖统计量方向、清单编号、空阈值、高斯卷积、特征范围、未知 CIDER 状态以及官方校准包的标签来源规则。
