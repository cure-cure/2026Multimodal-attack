# D 第五周真实固定融合交付给 A

运行 ID 为 D-week5-v1。使用 HoliSafe 主数据原有验证划分，共 605 条。
三种模式逐条判定均已生成，其中完整融合判正常 497 条，判风险 108 条，拒判 0 条。
这些是系统输出统计。准确率、误报和漏报使用 A 保管的真值另算。

- predictions.full.jsonl：主评测文件，mode=full，target=overall_risk。
- predictions.text_only.jsonl：仅使用文本分数的对照，mode=text_only，target=overall_risk。
- predictions.image_only.jsonl：B 的 CIDER 图像对抗扰动判定。主数据缺少对应图像对抗真值，供展示和覆盖检查。
- run_log.jsonl、run_summary.json：运行状态、覆盖、固定配置、版本和输入哈希。耗时不含 B/C 模型推理。
- input_provenance.json：B 提交、A 主清单来源、C 压缩包和验证文件指纹。
- demo_verification.json：真实证据回放与批量判定的核对记录。

评测运行示例（在仓库根目录，解压此包到 outputs/D_week5_v1，替换 A 的实际评测目录）：

    python -B evaluate_predictions.py --work-dir "A的评测目录" --predictions "outputs/D_week5_v1/predictions.full.jsonl" --module D --target overall_risk --mode full --split val --run-id D-week5-v1 --data-kind real --out "outputs/A_evaluation_D_week5_v1"

只使用验证样本进行了首次集成。未训练最终融合器，未修改阈值，未运行主测试集或校园补充集。
