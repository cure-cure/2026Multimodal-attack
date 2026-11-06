# D 模块运行说明

当前D交付包含第四周自主训练代码和系统框架，依据最新《题目十分工与DDL计划.md》。项目入口见[README](README.md)，详细训练与接入命令见[第四周交付](docs/WEEK4_D_DELIVERY.md)。

## 环境与示例

Python 3.10+，示例和测试无需第三方Python依赖。进入仓库根目录：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m mm_jailbreak_guard validate --manifest examples/handoff/manifest.jsonl --image-evidence examples/handoff/image_evidence.jsonl --text-evidence examples/handoff/text_evidence.jsonl
python -m mm_jailbreak_guard compare --manifest examples/handoff/manifest.jsonl --image-evidence examples/handoff/image_evidence.jsonl --text-evidence examples/handoff/text_evidence.jsonl --config configs/fusion_config.example.json --output outputs/mock-comparison.jsonl
python -m unittest discover -s tests -v
```

模拟包包含六个编号mock-001至mock-006，全部分数和标签均人工模拟，不是正式实验结果。validate检查必需字段、类型、编号、分组隔离及B/C文件；不检查图片存在性、真实标签正确性或图片近重复。

## 页面演示

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1
```

访问 http://127.0.0.1:8000 。脚本自动查找python或py；若未加入PATH：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1 -PythonPath "C:\path\to\python.exe"
```

页面可切换仅图像、仅文字、完整融合。使用mock-001至mock-006，离线证据按编号回放，上传图片只验证上传流程，不会对新上传内容重新识别或打分。真实单条检测须接入B/C在线提供方。

## 正式接入

B/C可以交付JSONL，也可实现[接口v2](docs/INTERFACE_SPEC.md)中的Python协议及无参数工厂：

```powershell
python -m mm_jailbreak_guard web --config configs/fusion_config.example.json --image-provider team_b.adapter:create_provider --text-provider team_c.adapter:create_provider
```

示例配置是尚未验证的固定基线。使用正式配置时替换config路径，并记录B/C版本。批量固定融合示例：

```powershell
python -m mm_jailbreak_guard batch --manifest data/manifest.test.jsonl --image-evidence outputs/B/image_evidence.test.jsonl --text-evidence outputs/C/text_evidence.test.jsonl --config configs/fusion_config.example.json --mode full --output outputs/D/predictions.full.test.jsonl
```

测试集必须由A按协议使用冻结版本运行，上述路径是待交付位置。batch支持image_only/text_only/full；user_text_only/user_text_ocr是额外文本对照。compare默认输出前三种。仅图像评测图像对抗扰动标签，另外两种评测整体风险标签，见[数据协议](docs/DATA_PROTOCOL.md)。

## 融合与进度

已实现固定加权融合、手写逻辑回归、启发式关联特征、异常处理及批量/页面框架。`train-fusion`在fusion_train训练、validation/val选阈值；`adapt-manifest`读取A分组CSV生成新清单；`collect-evidence`缓存B/C输出；compare的--baseline-config支持同证据固定融合对照。`fit`仍为独立网格搜索实验工具，不代替train-fusion。课件对应仍待核验，算法见[融合路线](docs/FUSION_ROADMAP.md)。

运行 `python scripts/check_d_workflow.py` 可生成outputs/D_mock/learned.json及模拟预测。页面脚本支持 `-ConfigPath outputs/D_mock/learned.json`。真实B/C接入、真实稳定分数训练、评测指标及真实演示尚未完成。A维护数据及真值并统一评测；D不实现B/C算法、不重新划分数据。输出和日志分别写入outputs/、runtime/，均被忽略。
