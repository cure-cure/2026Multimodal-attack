# 多证据融合越狱检测系统

本仓库是第十题“多模态大模型越狱攻击检测”的 D 模块交付。范围严格对应小组分工：统一接口、B/C 证据接入、基础融合、验证集选参、批量预测、三方案对比和在线演示。D 不替代 A 整理数据，不实现 B 的图像算法，也不实现 C 的文本分类算法。

当前版本已完成 D 在其他成员交付前能够独立完成的代码。B、C 尚未交付时，可用仓库内的脱敏示例证据验证流程；正式联调时替换为 B、C 的实现或结果文件，无需修改融合主流程。

本项目只做检测与防御，不生成越狱攻击内容。

## 已实现内容

- 统一输入、图像证据、文本证据、融合结果的数据结构和 JSON 约定。
- B 图像模块与 C 文本模块的可插拔接口，支持 Python 提供方或 JSONL 中间结果。
- D 自主实现的跨模态关联特征与加权后融合算法。
- 仅使用验证集的权重、阈值网格搜索；拒绝把测试集用于调参。
- “仅用户文字”“用户文字加 OCR 文字”“完整融合”三种模式。
- 批量预测输出，可按 `sample_id` 交给 A 统一评测。
- 图片上传、文字输入、风险分数、风险来源、模块状态和结果记录页面。
- 上游缺失、重复编号、分数越界和单模块异常时的显式错误处理。

## 目录

```text
configs/                         融合配置
docs/                            接口、算法和协作说明
examples/                        脱敏验证样例
src/mm_jailbreak_guard/          D 的实现
tests/                           自动化测试
```

## 快速运行

### 环境要求

- Python 3.10 或更高版本。
- 示例和测试不依赖第三方 Python 库。
- PowerShell 启动脚本会依次查找 `python` 和 Windows `py` 启动器；均不可用时会给出安装提示。

进入仓库根目录后执行：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m mm_jailbreak_guard web `
  --config configs/fusion_config.example.json `
  --image-evidence examples/image_evidence.validation.jsonl `
  --text-evidence examples/text_evidence.validation.jsonl
```

浏览器访问 `http://127.0.0.1:8000`。在示例模式中，样本编号需使用 `val-001` 到 `val-008`；上传图片用于演示上传链路，证据分数来自按编号匹配的脱敏示例结果。

也可以运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1
```

如果 Python 已安装但没有加入 `PATH`，可以显式指定解释器：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1 `
  -PythonPath "C:\path\to\python.exe"
```

## 先在验证集拟合融合参数

正式使用时，由 A 提供固定的验证集清单，由 B、C 分别输出证据文件。以下命令会同时拟合三种对比方案：

```powershell
python -m mm_jailbreak_guard fit `
  --manifest data/manifest.validation.jsonl `
  --image-evidence outputs/B/image_evidence.validation.jsonl `
  --text-evidence outputs/C/text_evidence.validation.jsonl `
  --output configs/fusion_config.json
```

程序会检查清单中的 `split`。一旦发现 `test`，会中止调参，避免违反“最终测试结果不能用于反复调整融合参数”的分工边界。

## 批量预测与三方案对比

```powershell
python -m mm_jailbreak_guard batch `
  --manifest data/manifest.test.jsonl `
  --image-evidence outputs/B/image_evidence.test.jsonl `
  --text-evidence outputs/C/text_evidence.test.jsonl `
  --config configs/fusion_config.json `
  --mode full `
  --output outputs/D/predictions.full.test.jsonl

python -m mm_jailbreak_guard compare `
  --manifest data/manifest.test.jsonl `
  --image-evidence outputs/B/image_evidence.test.jsonl `
  --text-evidence outputs/C/text_evidence.test.jsonl `
  --config configs/fusion_config.json `
  --output outputs/D/comparison.test.jsonl
```

`compare` 对每个样本输出三个方案的预测，不在 D 内计算最终测试指标；A 可以直接读取该文件完成统一评测。

仓库不提交 `outputs/` 中的运行结果。B、C、A 正式交付后，再由各负责人按约定路径生成对应文件。

## 项目进度与汇报

第十组按下半区安排在第 2、5、8、12 周进行正式汇报，依次由 D、B、C、A 主讲。每周任务、交付物、联调依赖和最终 8 分钟展示分配见 [每周进度与汇报安排](docs/WEEKLY_PLAN.md)。

## 接入 B、C

首选方式是让 B、C 实现 `ImageEvidenceProvider` 和 `TextEvidenceProvider` 协议，再通过 `模块路径:工厂函数` 启动：

```powershell
python -m mm_jailbreak_guard web `
  --config configs/fusion_config.json `
  --image-provider team_b.adapter:create_provider `
  --text-provider team_c.adapter:create_provider
```

如果 B、C 先交付离线结果，直接使用 `--image-evidence` 和 `--text-evidence` 即可，不需要改融合代码。字段定义见 [统一模块接口](docs/INTERFACE_SPEC.md)，融合依据见 [融合算法说明](docs/FUSION_ALGORITHM.md)，协作边界见 [D 实现与协作说明](docs/D_IMPLEMENTATION_NOTE.md)。

## 测试

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```

测试覆盖分数范围、不可用模块、重复编号、测试集调参保护、缺失证据降级、融合权重约束、上传文件签名和上游异常隔离。

## 上传到 GitHub

将本文件所在的 `multimodal-jailbreak-detection` 目录作为仓库根目录。不要把课程题目原文、汇报 PPT、讲稿、原始数据或本机临时目录一同上传。

如果使用命令行创建新仓库，可在本目录执行：

```powershell
git init
git add .
git commit -m "Implement D fusion and integration module"
git branch -M main
git remote add origin <你的 GitHub 仓库地址>
git push -u origin main
```

如果使用 GitHub Desktop，选择本目录创建或添加本地仓库即可。

## 当前边界

- 示例证据只用于联调，不是 B、C 的正式算法输出，也不能用于课程结论。
- D 不整理原始数据，不修改 A 的固定划分。
- D 不实现 B 的 OCR 或视觉特征算法，也不实现 C 的文本分类器。
- 视觉异常是辅助证据，默认和训练约束均不允许它单独主导判定。
- 测试集只能用于一次性最终预测和由 A 进行的统一评测。
- 当前基础融合算法为自主实现；正式课程报告中的“课件算法对应关系”须在取得课件后补充章节或页码依据。
- A/B/C/D 是仓库内的角色代号；课程正式分工材料应另行写明真实成员与角色对应关系。
