# D 当前基础融合算法

负责人D。自主代码位于 `src/mm_jailbreak_guard/fusion.py`：text_overlap、build_feature_vector、FusionEngine.fuse。未调用现成分类器。课件对应关系仍待核对，本文件不声明已经满足课程算法核验。

## 输入与关联特征

B提供visual_score，C提供user_text_score、ocr_text_score、combined_text_score，均为[0,1]且越高越风险。缺失用null。只有OCR状态为success时使用OCR和组合分数。

对用户文字和OCR文字进行小写及空白归一，取字符二元组集合，计算Jaccard重合度。启发式公式：

```text
overlap = |G_user ∩ G_ocr| / |G_user ∪ G_ocr|
synergy = sqrt(combined_text_score * max(user_text_score, ocr_text_score))
concealment = max(0, ocr_text_score - user_text_score) * (1 - overlap)
cross_modal_score = clip(0.7 * synergy * (0.75 + 0.25 * overlap) + 0.3 * concealment, 0, 1)
```

空集合时overlap取0。缺少部分文本分数时按代码中的回退计算；缺少必要分数则cross_modal_score为null。该启发式不是语义模型，其有效性须用真实数据验证。text_overlap保留在含OCR模式的解释输出中，不作为独立加权项。

## 判定与模式

固定融合按可用正权重重新归一：`score = Σ(w_i*x_i)/Σ(w_i)`，分数达到阈值判风险。当前示例权重及0.5阈值未经过真实验证。

- image_only：有CIDER时按B提供的原始统计量、比较方向和阈值判断；原始值保留在decision_rule/image_decision。不使用C分数，目标为图像对抗扰动。无CIDER时仅作为visual_score基线并警告。
- text_only：只用C三类文本分数，不用视觉或D协同分数。
- full：图像、三类文本及D协同特征融合，目标为整体风险。缺少全部文本证据时拒绝只凭视觉异常作整体判断。

缺少当前模式全部有效分数时输出prediction=null、insufficient_evidence。缺失分量会记录警告。完整配置的视觉名义权重上限为0.2；缺失后重新归一的有效权重可能提高，必须结合effective_weights解释，不把有噪声直接当越狱。

## 训练边界与评测

保留的fit是验证集网格搜索基线工具，不是最新版要求的可训练融合实现。新路线在fusion_train拟合参数、validation选阈值，具体候选和阶段见[FUSION_ROADMAP.md](FUSION_ROADMAP.md)。

compare使用同一批证据输出三模式，但仅图像的真值是image_adversarial_label，其余为整体label，不能混用准确率。D交逐条结果，A统一计算正式指标。模拟数据只作软件验证，不报告检测成绩。
