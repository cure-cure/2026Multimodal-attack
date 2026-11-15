# 2026Multimodal-attack

课程第十题「多模态大模型越狱攻击检测」：A 数据与评测、B 图像侧证据、C 文本侧检测、D 多证据融合与本地演示。仓库地址：https://github.com/cure-cure/2026Multimodal-attack 。

执行依据：课程任务 `2026多媒体大作业题目.docx` 与最新版 `题目十分工与DDL计划.md`。只做检测与防御。

## 成员从这里开始

- [B 图像侧模块与真实实验结果](team_b/README.md)：CIDER、OCR、高斯/频域特征、校准协议及双标签轴结果。
- [D第四周本地交付与复现](docs/WEEK4_D_DELIVERY.md)：自主训练、A数据适配、证据缓存和固定/训练融合对比。

- [第三周交接清单与待确认事项](docs/WEEK3_HANDOFF.md)：A、B、C分别需要读取、确认和提交什么。
- [统一接口 v2](docs/INTERFACE_SPEC.md)：样本清单、B/C 输出、CIDER 证据、模式与评测目标。
- [数据用途及隔离约定](docs/DATA_PROTOCOL.md)：基础训练、融合训练、验证、测试四类划分。
- [D 自主融合算法路线](docs/FUSION_ROADMAP.md)：固定融合与可训练算法候选、课件核对状态。
- [运行说明](PROJECT_README.md)：启动、批量预测和测试命令。
- [模拟交接包](examples/handoff/README.md)：含正常、风险、OCR失败和模块失败样例。
- [第四周统一评测说明](docs/WEEK4_EVALUATION.md)：预测格式、评测口径、覆盖率和测试集冻结要求。

当前 B 已公开图像侧实现及聚合实验报告：CIDER 扰动分数具有明显排序能力，但官方 validation/test 共享 1 条请求、4 张图片和 4 组完全相同图文，因此 test 既不是“完全请求留出”，也不是 image-holdout，只能作为存在已量化重叠的官方划分迁移评测。逐样本 OCR、校园衍生输出及其他 B 逐样本产物的公开范围仍待组长确认。D 已实现固定融合与手写逻辑回归训练；完整整体安全效果仍需 B/C 共同证据，模拟文件不用于报告成绩。

## 快速检查

Python 3.10+，无需第三方依赖。在仓库根目录运行：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m mm_jailbreak_guard validate --manifest examples/handoff/manifest.jsonl --image-evidence examples/handoff/image_evidence.jsonl --text-evidence examples/handoff/text_evidence.jsonl
python -m mm_jailbreak_guard compare --manifest examples/handoff/manifest.jsonl --image-evidence examples/handoff/image_evidence.jsonl --text-evidence examples/handoff/text_evidence.jsonl --config configs/fusion_config.example.json --output outputs/mock-comparison.jsonl
python -m unittest discover -s tests -v
python scripts/check_d_workflow.py
```

演示页面：`powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1`，访问 http://127.0.0.1:8000 。页面支持仅图像、仅文字、完整融合；仅图像针对图像对抗扰动，CIDER仍以用户文字为语义参照。
