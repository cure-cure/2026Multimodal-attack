# 统一模块接口说明

接口版本：v2。以工作区最新版《题目十分工与DDL计划.md》为准。本文为D交付的对接提案，成员确认见 [第三周交接表](WEEK3_HANDOFF.md)。

CIDER依据：[正式论文及元数据](https://aclanthology.org/2024.findings-emnlp.803/)、[作者代码](https://github.com/PandragonXIII/CIDER)。本文扩展字段是本组工程接口，不是作者代码原生输出格式，B负责适配并记录复现差异。

本文档定义 A、B、C、D 之间的最小稳定接口。目标是让各模块独立开发、按样本编号对齐，并在不修改彼此核心算法的情况下完成集成。

## 设计原则

1. 所有跨模块记录必须包含唯一的 `sample_id`。
2. 所有风险分数统一为闭区间 `[0, 1]`，数值越大表示风险越高。
3. 模块不可用、没有检测到内容和运行失败是三个不同状态，不能用分数 `0` 混为一谈。
4. 证据必须可解释，不能只交付一个标签。
5. 上游模块的异常由 D 记录并降级处理，不能静默吞掉。
6. A 的数据划分是唯一依据；融合训练用fusion_train，阈值选择用validation，测试集不参与调参，见[数据用途](DATA_PROTOCOL.md)。旧fit只保留为原型调试工具。
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

正式清单还需 `image_group_id`、`request_group_id`，`split`固定为base_train/fusion_train/validation/test。保留harmful_label、jailbreak_label、image_adversarial_label，类型0/1/null；整体label映射由A确定，不能用预测替代真值。相对图片路径以清单所在文件夹为基准。

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

### CIDER扩展（B→D；同一份文件可供B→C提取OCR）

ImageEvidence新增可选 `cider` 对象；旧文件可省略，运行时提示未提供CIDER。不可用模块必须为null或省略。正常CIDER交付应提供：

```json
{"similarities":[0.30,0.40,0.60],"statistic":0.30,"statistic_name":"mock_max_minus_original","threshold":0.20,"comparison":"ge","config_id":"B-mock-config","reproduction":"mock"}
```

上例为模拟，绝非论文公式或真实实验。`similarities`按原图、去噪阶段1…顺序保存原始相似度（有限数，可为负，不强制[0,1]）。`statistic`和`threshold`也是原始有限数；`statistic_name`说明统计量定义，`comparison`用ge表示大于等于阈值为风险、le表示小于等于为风险。B在config_id对应配置中记录精确公式、编码器、去噪方法/阶段、阈值选择集、论文复现差异及视觉分数转换方法。D不猜测或硬编码论文公式。

`reproduction`为original（经B核验原配置）、adapted（适配）、mock（模拟）。`visual_score`为B转换后的[0,1]图像风险分数，供融合；不能把原始余弦相似度直接塞进去。`visual_features`仍只放归一化特征，原始相似度放cider，避免负数被校验拒绝。转换参数只能用允许的训练/验证数据确定。

available表示至少一项图像/OCR结果可供使用，不表示OCR一定成功。OCR失败但CIDER成功时available=true、ocr_status=failed，保留图像证据；图像检测失败但OCR成功时visual_score=null、cider=null，可继续文本分析。B→C的OCR缓存可直接使用本JSONL，C按sample_id读取ocr_text/status/regions，不必另造格式。

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

OCR状态不是success时，D不使用OCR与组合分数。第四周C不等待OCR，仅提供user_text_score即可。success与no_text不同于failed/unavailable；C不能将用户文字复制为OCR冒充结果。

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

### 模式和评测目标

| mode | 使用证据 | prediction_target与真值 |
| --- | --- | --- |
| image_only | B的CIDER判定；无CIDER时为visual_score基线，附警告 | image_adversarial；对照image_adversarial_label |
| text_only | C三类文本分数；不使用图像分数和D协同分数 | overall_risk；对照A整体label |
| full | B图像、C文本与D协同特征 | overall_risk；对照A整体label |

仅图像指仅启用B检测分支，仍需用户文字作CIDER语义参照，不等于完全不输入文字；仅文字可读B缓存的OCR，但不使用视觉分数。`compare`默认输出上述三模式。旧user_text_only和user_text_ocr保留为batch额外对照。

仅图像有cider时按B提供的原始statistic/threshold/comparison判断，真实规则见批量输出 `image_decision`，归一化threshold仅用于分数基线，不替代B原始阈值。仅图像缺少全部图像证据时返回证据不足；完整融合缺少全部文本分数时仍拒绝只凭图像异常判图文整体风险。A分别评测两种prediction_target，不把它们放到同一列直接比较“谁更准确”。

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
