# 模拟交接包

本目录六条样本、标签、CIDER统计和分数均为人工模拟，图片路径为null，不是公开或自采真实数据，不可报告检测成绩。`reproduction: mock` 不代表CIDER复现。阈值及原始统计量只是接口占位，真实定义由B交付配置说明。

| sample_id | split | 演示目的 |
| --- | --- | --- |
| mock-001 | base_train | 正常数据格式 |
| mock-002 | fusion_train | 模拟风险、CIDER证据 |
| mock-003 | fusion_train | OCR失败，图像检测仍可用 |
| mock-004 | validation | 正常、图中无文字 |
| mock-005 | validation | B失败，C用户文字仍可用 |
| mock-006 | test | B/C均失败，证据不足 |

运行仓库README中的validate和compare即可检查。启动页面时把两个 evidence 参数改为本目录文件，使用 mock-001 至 mock-006。固定融合不需要label；清单的label只演示A字段，不应进入推理。该极小包不满足正式训练样本要求。
