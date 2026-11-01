# 2026Multimodal-attack

课程第十题「多模态大模型越狱攻击检测」的 D 模块：统一接口、多证据融合、批量预测与本地演示。仓库地址：https://github.com/cure-cure/2026Multimodal-attack 。

执行依据：课程任务 `2026多媒体大作业题目.docx` 与最新版 `题目十分工与DDL计划.md`。只做检测与防御。

## 成员从这里开始

- [第三周交接清单与待确认事项](docs/WEEK3_HANDOFF.md)：A、B、C分别需要读取、确认和提交什么。
- [统一接口 v2](docs/INTERFACE_SPEC.md)：样本清单、B/C 输出、CIDER 证据、模式与评测目标。
- [数据用途及隔离约定](docs/DATA_PROTOCOL.md)：基础训练、融合训练、验证、测试四类划分。
- [D 自主融合算法路线](docs/FUSION_ROADMAP.md)：固定融合与可训练算法候选、课件核对状态。
- [运行说明](PROJECT_README.md)：启动、批量预测和测试命令。
- [模拟交接包](examples/handoff/README.md)：含正常、风险、OCR失败和模块失败样例。

当前可运行的是模拟证据流程，未接入真实 B/C 模型，不具有正式检测指标。模拟文件不用于报告成绩。自主算法是否对应课件仍需提供课件核对；成员确认尚未完成。

## 快速检查

Python 3.10+，无需第三方依赖。在仓库根目录运行：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m mm_jailbreak_guard validate --manifest examples/handoff/manifest.jsonl --image-evidence examples/handoff/image_evidence.jsonl --text-evidence examples/handoff/text_evidence.jsonl
python -m mm_jailbreak_guard compare --manifest examples/handoff/manifest.jsonl --image-evidence examples/handoff/image_evidence.jsonl --text-evidence examples/handoff/text_evidence.jsonl --config configs/fusion_config.example.json --output outputs/mock-comparison.jsonl
python -m unittest discover -s tests -v
```

演示页面：`powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1`，访问 http://127.0.0.1:8000 。页面支持仅图像、仅文字、完整融合；仅图像针对图像对抗扰动，CIDER仍以用户文字为语义参照。
