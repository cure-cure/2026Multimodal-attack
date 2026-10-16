# 统一模块接口说明

本文档定义 A、B、C、D 之间的最小稳定接口。目标是让各模块独立开发、按样本编号对齐，并在不修改彼此核心算法的情况下完成集成。

## 设计原则

1. 所有跨模块记录必须包含唯一的 `sample_id`。
2. 所有风险分数统一为闭区间 `[0, 1]`，数值越大表示风险越高。
3. 模块不可用、没有检测到内容和运行失败是三个不同状态，不能用分数 `0` 混为一谈。
4. 证据必须可解释，不能只交付一个标签。
5. 上游模块的异常由 D 记录并降级处理，不能静默吞掉。
6. A 的数据划分是唯一依据；D 的拟合入口只允许验证集，测试集不参与调参。
7. `available` 必须是 JSON 布尔值。模块声明 `available: false` 时，所有风险分数必须为 `null`，不得把旧分数或占位分数带入融合。

## A 交给 D 的样本清单

JSONL 每行一个对象，最低字段如下：

```json
{
  "sample_id": "sample-0001",
  "image_path": "data/images/sample-0001.png",
  "user_text": "待检测文字",
  "label": 0,
  "split": "validation",
  "source": "公开数据或自构来源标识",
  "scenario": "场景类别"
}
```

`label` 在无标签在线检测中可以省略；训练融合参数时必须提供。D 不修改清单、标签或划分。

## B 交给 D 的图像证据

对应 Python 类型为 `ImageEvidence`，离线 JSONL 结构如下：

```json
{
  "sample_id": "sample-0001",
  "available": true,
  "ocr_text": "图片中识别到的文字",
  "ocr_status": "success",
  "visual_score": 0.18,
  "visual_features": {
    "high_frequency_ratio": 0.21,
    "denoise_difference": 0.14
  },
  "regions": [
    {"text": "局部文字", "confidence": 0.91, "box": [20, 30, 180, 66]}
  ],
  "evidence": [
    {"source": "image", "description": "未发现明显视觉异常", "score": 0.18}
  ],
  "warnings": [],
  "module_version": "B-1.0"
}
```

`ocr_status` 允许 `success`、`no_text`、`failed`、`unavailable`。当 `visual_score` 无法计算时必须写 `null`，不能假装为 `0`。当 `available` 为 `false` 时，`ocr_status` 只能是 `failed` 或 `unavailable`，且 `ocr_text`、`visual_score`、`visual_features` 和 `regions` 均不得携带可用结果。

## C 交给 D 的文本证据

对应 Python 类型为 `TextEvidence`：

```json
{
  "sample_id": "sample-0001",
  "available": true,
  "user_text_score": 0.12,
  "ocr_text_score": 0.20,
  "combined_text_score": 0.16,
  "evidence": [
    {
      "source": "user_text",
      "description": "未命中高风险规则",
      "score": 0.12,
      "locations": []
    }
  ],
  "warnings": [],
  "module_version": "C-1.0"
}
```

C 应分别给出用户文字、OCR 文字和组合文字的分数。若 B 不可用或 OCR 失败，`ocr_text_score` 与 `combined_text_score` 应写为 `null`，并在 `warnings` 说明原因；D 也会在图像模块不可用时强制忽略这两项。当 C 的 `available` 为 `false` 时，三类风险分数必须全部为 `null`。

## D 输出给 A 的批量预测

```json
{
  "sample_id": "sample-0001",
  "prediction": 0,
  "label_name": "normal",
  "risk_score": 0.19,
  "threshold": 0.5,
  "mode": "full",
  "component_scores": {
    "user_text_score": 0.12,
    "ocr_text_score": 0.20,
    "combined_text_score": 0.16,
    "visual_score": 0.18,
    "cross_modal_score": 0.24,
    "text_overlap": 0.09
  },
  "effective_weights": {
    "user_text_score": 0.25,
    "ocr_text_score": 0.20,
    "combined_text_score": 0.25,
    "visual_score": 0.10,
    "cross_modal_score": 0.20
  },
  "risk_sources": [],
  "warnings": [],
  "module_versions": {"B": "B-1.0", "C": "C-1.0", "D": "D-0.1.0"},
  "fusion_config_id": "validation-..."
}
```

若所有必需文本分数均缺失，输出 `prediction: null`、`label_name: "insufficient_evidence"`，而不是强行判为正常。

## Python 在线接口

B 实现：

```python
class ImageEvidenceProvider(Protocol):
    name: str
    version: str

    def analyze(self, request: DetectionRequest) -> ImageEvidence:
        ...
```

C 实现：

```python
class TextEvidenceProvider(Protocol):
    name: str
    version: str

    def analyze(
        self,
        request: DetectionRequest,
        image_evidence: ImageEvidence,
    ) -> TextEvidence:
        ...
```

推荐各自提供无参数工厂函数 `create_provider()`。D 启动时使用 `--image-provider 包.模块:create_provider` 与 `--text-provider 包.模块:create_provider` 动态加载。

## 对接检查表

- A：`sample_id` 唯一，划分固定，验证集和测试集清楚标识。
- B：OCR 状态、OCR 文本、视觉分数、特征和证据齐全。
- C：三类文本分数、判定依据和缺失原因齐全。
- D：配置带唯一编号，输出记录包含阈值、有效权重、上游版本和警告。
- B/C：不可用状态不携带风险分数；布尔值和分数使用正确的 JSON 类型。
- 全组：正式测试前冻结融合配置，不依据测试结果回改权重或阈值。
