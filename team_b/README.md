# Task B：图像侧证据、OCR 与 CIDER

本目录是课程任务 B 的公开可复现实现，只做检测与防御。它提供 CIDER 图文相似度变化、EasyOCR 中英文区域、高斯/频域辅助特征、图片哈希缓存、数据审计和 D `ImageEvidence` 适配。

公开仓库不包含 HoliSafe 图片、逐样本 OCR/文本、模型权重、缓存或完整原始结果。聚合结果见 [双标签轴报告](reports/B_DUAL_AXIS_ANALYSIS.md)。

## 结论与标签边界

两个目标分开评估：

1. `image_adv_label`：CIDER 官方 clean/adversarial 数据，用于 B 的图片扰动专项评测。
2. `overall_risk_label`：HoliSafe 图文整体风险，仅用于探索 B 特征与广义安全的相关性。

| 目标 | 数据 | 结果 |
|---|---|---|
| 图片扰动排序 | CIDER validation | AUROC 0.9378，AP 0.9375 |
| 图片扰动排序 | CIDER test | AUROC 0.9679，AP 0.9720 |
| 95% clean-pass 阈值 | validation | Recall 0.4268，FPR 0.0061 |
| 95% clean-pass 阈值 | test | Recall 0.4000，FPR 0.0000 |
| 整体风险相关性 | HoliSafe 3,415 条有标签样本 | 各 B 信号 AUROC 约 0.48–0.54 |

限制：CIDER 官方 validation/test 的 4 张对抗图片完全重复，只更换 HarmBench 请求；因此 test 是 request-holdout 迁移结果，不证明对全新攻击图片的泛化。整体安全仍应由 B 图像证据与 C 文本证据交给 D 融合。

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

## 测试

```bash
python -m unittest discover -s team_b/tests -v
```

测试覆盖统计量方向、清单编号、空阈值、高斯卷积、特征范围、未知 CIDER 状态以及官方校准包的标签来源规则。
