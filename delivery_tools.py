"""交付条件及D接口导出；不修改标签、不生成预测、不下载数据。"""
import json
from pathlib import Path


def release_issues(inputs, labels, splits):
    sp = {r['sample_id']: r['split'] for r in splits}
    verified = {r['sample_id']: r for r in labels if r['label_status'] == 'verified'}
    issues = []
    if not inputs:
        issues.append('没有正式图文样本')
    if len(verified) != len(inputs):
        issues.append('存在未核实标签，应从正式包隔离')
    for part in ('base_train', 'fusion_train', 'val', 'test'):
        values = {r['overall_risk_label'] for sid, r in verified.items() if sp.get(sid) == part}
        if values != {'0', '1'}:
            issues.append(f'{part}缺少已核实的整体风险正类或负类')
    return issues


def export_d_manifests(directory, inputs, labels, splits):
    """图片路径相对于清单目录。batch清单无真值，D只取得融合训练和验证标签。"""
    directory = Path(directory)
    sp = {r['sample_id']: r['split'] for r in splits}
    labs = {r['sample_id']: r for r in labels}
    manifests = {'manifest.inputs.jsonl': [], 'manifest.fusion_train.jsonl': [],
                 'manifest.validation.jsonl': [], 'manifest.test.jsonl': []}
    for row in inputs:
        sid = row['sample_id']
        part = sp[sid]
        record = {'sample_id': sid, 'image_path': row['image_path'],
                  'user_text': row['user_text'], 'split': 'validation' if part == 'val' else part,
                  'source': row['source_id']}
        manifests['manifest.inputs.jsonl'].append(record)
        if part == 'test':
            manifests['manifest.test.jsonl'].append(dict(record))
        if part in ('fusion_train', 'val'):
            label = labs[sid]
            if label['label_status'] == 'verified' and label['overall_risk_label'] in ('0', '1'):
                labelled = dict(record, label=int(label['overall_risk_label']))
                filename = 'manifest.validation.jsonl' if part == 'val' else 'manifest.fusion_train.jsonl'
                manifests[filename].append(labelled)
    for filename, rows in manifests.items():
        (directory / filename).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (directory / 'D_INTERFACE_NOTES.md').write_text(
        '# D接口使用说明\n\n'
        'manifest.inputs.jsonl供batch使用，不含真值。图片相对于清单位置。\n'
        'manifest.test.jsonl不含label；不得补入测试真值。\n'
        '仅val映射为validation；fusion_train保持原名。\n'
        'D当前fit仅接受validation，与组长计划冲突。D须将融合拟合入口改为fusion_train，'
        '验证集另用于阈值选择；适配前不要重命名数据划分来绕过检查。\n'
        'B的image_score与D的visual_score须约定语义及校准，不能默认直接复制CIDER原始相似度差。\n'
        'OCR状态统一为success/no_text/failed/unavailable，缺失分数使用null。\n', encoding='utf-8')
