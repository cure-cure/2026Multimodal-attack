# -*- coding: utf-8 -*-
"""A组第三周数据工程：仅整理数据，不训练模型、不产生攻击样本。Python 3.10+."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import random
import re
import shutil
import sys
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
ENTRY_COLS = [
    'image_file', 'user_text', 'source_id', 'image_group_id', 'request_group_id',
    'image_adv_label', 'harmful_content_label', 'jailbreak_intent_label',
    'prompt_injection_label', 'overall_risk_label', 'label_status',
    'adv_evidence', 'notes'
]
SOURCE_COLS = ['source_id', 'source_name', 'source_url', 'license', 'access_status', 'notes']
LABEL_COLS = ['image_adv_label', 'harmful_content_label',
              'jailbreak_intent_label', 'prompt_injection_label', 'overall_risk_label']
INPUT_COLS = ['sample_id', 'image_path', 'user_text', 'image_group_id',
              'request_group_id', 'source_id']
SPLITS = ('base_train', 'fusion_train', 'val', 'test')
SPLIT_RATIO = {'base_train': 0.50, 'fusion_train': 0.20, 'val': 0.15, 'test': 0.15}


def csv_read(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_write(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalized_text(s: str) -> str:
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', s)).strip().casefold()


def ensure_init(root: Path):
    (root / 'work').mkdir(parents=True, exist_ok=True)
    (root / 'output').mkdir(parents=True, exist_ok=True)
    (root / 'images').mkdir(parents=True, exist_ok=True)
    if not (root / 'entries.csv').exists():
        csv_write(root / 'entries.csv', [], ENTRY_COLS)
    if not (root / 'sources.csv').exists():
        csv_write(root / 'sources.csv', [], SOURCE_COLS)
    print(f'已初始化：{root}')
    print('请填写 sources.csv 和 entries.csv；原始图片可以放在项目 raw_images/ 中。')


def scan_cider(root: Path, repo: Path):
    repo = repo.expanduser().resolve()
    if not (repo / 'README.md').exists():
        raise ValueError('找不到仓库README.md，请传入CIDER根目录，例如 D:\\CIDER')
    lines = [f'CIDER目录：{repo}', '仅检查文件清单，不推断真实攻击标签。', '']
    img_dir = repo / 'data' / 'img'
    txt_dir = repo / 'data' / 'text'
    imgs = sorted(p for p in img_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXT) if img_dir.exists() else []
    csvs = sorted(txt_dir.rglob('*.csv')) if txt_dir.exists() else []
    lines.append(f'data/img 图片数量：{len(imgs)}')
    lines.append(f'data/text CSV数量：{len(csvs)}')
    inventory=[]
    for image in imgs:
        filename=image.name.lower()
        hint='clean_prefix' if filename.startswith('clean_') else ('prompt_constrained_prefix' if filename.startswith('prompt_constrained_') else 'none')
        inventory.append({'image_file':str(image),'relative_path':str(image.relative_to(repo)),
                          'filename_hint_UNVERIFIED':hint,'sha256':sha_file(image)})
    csv_write(root/'output'/'cider_image_inventory.csv', inventory,
              ['image_file','relative_path','filename_hint_UNVERIFIED','sha256'])
    lines.append('图片示例（仅文件名提示，不能视为真值）：')
    lines.extend('  ' + str(p.relative_to(repo)) for p in imgs[:35])
    lines.append('\nCSV信息：')
    for p in csvs[:30]:
        try:
            raw = p.open('r', encoding='utf-8-sig', errors='replace').readline().strip()
        except OSError as e:
            raw = f'无法读取: {e}'
        lines.append(f'  {p.relative_to(repo)} | 第一行: {raw[:230]}')
    lines.extend(['', '温馨提示：CIDER README描述的combine模式会生成笛卡尔积，',
                  'injection模式要求图文一一配对。A需确认配对来源，不要臆造对应关系。',
                  '缺少图片、CSV或模型权重均须如实记录，扫描不会下载资源。'])
    report = root / 'output' / 'cider_scan.txt'
    report.parent.mkdir(exist_ok=True, parents=True)
    report.write_text('\n'.join(lines), encoding='utf-8-sig')
    print('\n'.join(lines[:8]))
    print(f'完整结果已保存: {report}')
    print('完整图片清单：output/cider_image_inventory.csv（提示前缀不是真实标签）')


def safe_path(root: Path, input_path: str) -> Path:
    s = input_path.strip().strip('"')
    if not s:
        raise ValueError('image_file为空')
    p = Path(s).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def add_pairs(root: Path, args):
    ensure_init(root)
    path = Path(args.csv).expanduser().resolve()
    base = Path(args.image_root).expanduser().resolve()
    rows = csv_read(path)
    if not rows:
        raise ValueError('输入CSV没有数据行')
    for col in (args.image_col, args.text_col):
        if col not in rows[0]:
            raise ValueError(f'找不到CSV列 {col}；现有列: {list(rows[0].keys())}')
    old = csv_read(root / 'entries.csv')
    existing = {(r['image_file'], r['user_text'], r['source_id']) for r in old}
    added, skipped = 0, 0
    for r in rows:
        image_name = (r.get(args.image_col) or '').strip()
        text = (r.get(args.text_col) or '').strip()
        img = base / image_name
        if not text or not img.is_file():
            skipped += 1
            continue
        new = dict.fromkeys(ENTRY_COLS, '')
        new.update(image_file=str(img.resolve()), user_text=text,
                   source_id=args.source_id, label_status='pending',
                   notes='CSV批量导入；请人工核对图文对应及标签')
        key = (new['image_file'], new['user_text'], new['source_id'])
        if key in existing:
            skipped += 1
            continue
        existing.add(key)
        old.append(new)
        added += 1
    csv_write(root / 'entries.csv', old, ENTRY_COLS)
    print(f'批量导入完成：新增{added}条；跳过{skipped}条（缺失图片、空文本或重复）')
    print('重要：新增记录均为 pending，不能直接视为已标注。请先核对、填写标签。')


def import_pairing_plan(root: Path, args):
    """按资源ID增量导入已确认配对；dry-run不写入任何文件。"""
    plan_path = Path(args.plan).expanduser().resolve()
    image_path = Path(args.images).expanduser().resolve()
    text_path = Path(args.texts).expanduser().resolve()
    for path in (plan_path, image_path, text_path):
        if not path.is_file():
            raise ValueError(f'找不到输入文件：{path}')
    plans = csv_read(plan_path)
    def index(path, key):
        rows = csv_read(path)
        result = {}
        for row in rows:
            identifier = (row.get(key) or '').strip()
            if not identifier or identifier in result:
                raise ValueError(f'{path.name}含空白或重复{key}')
            result[identifier] = row
        return result
    images = index(image_path, 'image_resource_id')
    texts = index(text_path, 'text_resource_id')
    sources = index(root / 'sources.csv', 'source_id')
    required = {'image_resource_id', 'text_resource_id', 'pairing_type',
                'pairing_evidence_or_rule', 'label_status'}
    if plans and not required.issubset(plans[0]):
        raise ValueError(f'配对计划缺少字段：{sorted(required - set(plans[0]))}')
    old = csv_read(root / 'entries.csv')
    if old and set(old[0]) != set(ENTRY_COLS):
        raise ValueError('entries.csv字段与当前版本不一致；停止以防丢失自定义字段')
    def key_for(row):
        return (sha_file(safe_path(root, row['image_file'])),
                row['user_text'].strip(), row['source_id'].strip())
    existing = {key_for(r): r for r in old}
    additions, skipped, errors = [], [], []
    for row_number, plan in enumerate(plans, 2):
        image_id = (plan.get('image_resource_id') or '').strip()
        text_id = (plan.get('text_resource_id') or '').strip()
        status = (plan.get('label_status') or '').strip()
        image = images.get(image_id)
        text = texts.get(text_id)
        problems = []
        if image is None: problems.append(f'图片资源不存在：{image_id}')
        if text is None: problems.append(f'文本资源不存在：{text_id}')
        if not (plan.get('pairing_type') or '').strip(): problems.append('pairing_type为空')
        if not (plan.get('pairing_evidence_or_rule') or '').strip(): problems.append('缺少配对依据')
        if status not in ('verified', 'review', 'pending'): problems.append('label_status非法')
        if (plan.get('pairing_status') or '') != 'verified':
            problems.append('pairing_status必须为verified（确认配对不等于确认全部标签）')
        if image is not None:
            source_image = Path(image.get('image_file', '')).expanduser()
            if not source_image.is_file(): problems.append(f'图片文件不存在：{source_image}')
        if problems:
            errors.append(f'第{row_number}行：' + '；'.join(problems))
            continue
        source_id = (plan.get('source_id') or args.source_id).strip()
        if source_id not in sources:
            errors.append(f'第{row_number}行：source_id未在sources.csv登记')
            continue
        user_text = (plan.get('user_text_override') or text.get('behavior_text') or '').strip()
        if text.get('behavior_type_col2') in ('contextual', 'copyright'):
            if not (plan.get('user_text_override') or '').strip():
                errors.append(f'第{row_number}行：特殊文本须在user_text_override中确认完整上下文')
                continue
        new = dict.fromkeys(ENTRY_COLS, '')
        new.update(
            image_file=str(Path(image['image_file']).expanduser().resolve()),
            user_text=user_text,
            source_id=source_id,
            image_group_id=(plan.get('image_group_id_MANUAL') or image.get('visual_family_group_id_MANUAL') or '').strip(),
            request_group_id=(plan.get('request_group_id_MANUAL') or '').strip(),
            image_adv_label=(plan.get('image_adv_label') or '').strip(),
            harmful_content_label=(plan.get('harmful_content_label') or '').strip(),
            jailbreak_intent_label=(plan.get('jailbreak_intent_label') or '').strip(),
            prompt_injection_label=(plan.get('prompt_injection_label') or '').strip(),
            overall_risk_label=(plan.get('overall_risk_label') or '').strip(),
            label_status=status,
            adv_evidence=(plan.get('adv_evidence') or image.get('label_evidence') or '').strip(),
            notes='资源配对计划导入；类型=' + (plan.get('pairing_type') or '').strip() +
                  '；依据=' + (plan.get('pairing_evidence_or_rule') or '').strip() +
                  f'；资源={image_id}/{text_id}；标签依据=' + (plan.get('label_evidence') or '').strip() +
                  ('；' + (plan.get('notes') or '').strip() if (plan.get('notes') or '').strip() else '')
        )
        if not new['user_text']:
            errors.append(f'第{row_number}行：文本内容为空')
            continue
        if not new['image_group_id'] or not new['request_group_id']:
            errors.append(f'第{row_number}行：图片家族和请求组必须审核后填写')
            continue
        if status == 'verified' and (not new['overall_risk_label'] or not (plan.get('label_evidence') or '').strip()):
            errors.append(f'第{row_number}行：verified须有整体标签及label_evidence')
            continue
        if new['image_adv_label'] == '1' and not new['adv_evidence']:
            errors.append(f'第{row_number}行：缺少对抗图像证据')
            continue
        key = key_for(new)
        if key in existing:
            fields = LABEL_COLS + ['label_status', 'image_group_id', 'request_group_id', 'adv_evidence', 'notes']
            if any(existing[key].get(k, '') != new[k] for k in fields):
                errors.append(f'第{row_number}行：已有样本的标签、分组或证据冲突，未覆盖')
                continue
            skipped.append(f'第{row_number}行：已有相同图片+文字+来源')
            continue
        for label in LABEL_COLS:
            if new[label] not in ('', '0', '1'):
                errors.append(f'第{row_number}行：{label}只能为0、1或留空')
                break
        else:
            existing[key] = new
            additions.append(new)
    if errors:
        raise ValueError('配对计划未导入任何记录：\n' + '\n'.join(errors))
    print(f'配对计划检查完成：可新增{len(additions)}条，跳过{len(skipped)}条。')
    if args.dry_run:
        print('dry-run：未修改entries.csv。')
        return
    if additions:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        backup = root / 'backups' / f'pairing_import_{timestamp}'
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / 'entries.csv', backup / 'entries.csv')
        csv_write(root / 'entries.csv', old + additions, ENTRY_COLS)
    log = root / 'work' / 'pairing_import_log.txt'
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text('\n'.join([f'新增：{len(additions)}', f'跳过：{len(skipped)}'] + skipped) + '\n', encoding='utf-8-sig')
    print(f'已导入{len(additions)}条记录；备份和日志已保存。')



def add_sample(root: Path, args):
    ensure_init(root)
    image = safe_path(root, args.image)
    if not image.is_file():
        raise ValueError(f'图片不存在：{image}')
    if not args.text.strip() or not args.source_id.strip():
        raise ValueError('文字和source-id不能为空')
    old = csv_read(root / 'entries.csv')
    row = dict.fromkeys(ENTRY_COLS, '')
    row.update(image_file=str(image), user_text=args.text.strip(),
               source_id=args.source_id.strip(),
               image_group_id=args.image_group, request_group_id=args.request_group,
               image_adv_label=args.image_adv, harmful_content_label=args.harmful,
               jailbreak_intent_label=args.jailbreak,
               prompt_injection_label=args.prompt_injection,
               overall_risk_label=args.overall, label_status=args.status,
               adv_evidence=args.adv_evidence, notes=args.notes)
    if any(x['image_file']==row['image_file'] and x['user_text']==row['user_text']
           and x['source_id']==row['source_id'] for x in old):
        raise ValueError('entries.csv已有完全相同的图片+文字+来源记录，请勿重复录入')
    old.append(row)
    csv_write(root / 'entries.csv', old, ENTRY_COLS)
    print(f'已新增第{len(old)}条草稿。请检查entries.csv，随后运行build。')


def build(root: Path):
    ensure_init(root)
    raw = csv_read(root / 'entries.csv')
    if not raw:
        raise ValueError('entries.csv仍为空。请先登记真实图文，再运行build。')
    inputs, labels, problems = [], [], []
    seen_id = set()
    for idx, row in enumerate(raw, 2):
        path_text, text, src = [str(row.get(k, '') or '').strip() for k in ('image_file', 'user_text', 'source_id')]
        if not path_text and not text and not src:
            continue
        if not path_text or not text or not src:
            problems.append(f'第{idx}行：image_file / user_text / source_id必须填写')
            continue
        try:
            path = safe_path(root, path_text)
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.suffix.lower() not in IMAGE_EXT:
                raise ValueError('文件扩展名不是受支持的图片')
            with Image.open(path) as image:
                image.verify()
            digest = sha_file(path)
        except Exception as exc:
            problems.append(f'第{idx}行：图片无效：{path_text}（{exc}）')
            continue
        fp = sha_bytes((digest + '\0' + text + '\0' + src).encode('utf-8'))
        sid = 'S' + fp[:12].upper()
        if sid in seen_id:
            problems.append(f'第{idx}行：重复样本（同图片、文字和来源）：{sid}')
            continue
        seen_id.add(sid)
        dest = root / 'images' / f'{digest[:20]}{path.suffix.lower()}'
        if not dest.exists():
            shutil.copy2(path, dest)
        image_group = (row.get('image_group_id') or '').strip() or 'IG_' + digest[:12]
        request_group = (row.get('request_group_id') or '').strip() or 'RG_' + sha_bytes(normalized_text(text).encode())[:12]
        inputs.append({'sample_id': sid, 'image_path': 'images/' + dest.name,
                       'user_text': text, 'image_group_id': image_group,
                       'request_group_id': request_group, 'source_id': src})
        lab = {'sample_id': sid}
        for k in LABEL_COLS:
            lab[k] = str(row.get(k, '') or '').strip()
        lab['label_status'] = str(row.get('label_status', '') or '').strip() or 'pending'
        lab['adv_evidence'] = str(row.get('adv_evidence', '') or '').strip()
        lab['notes'] = str(row.get('notes', '') or '').strip()
        labels.append(lab)
    if problems:
        log = root / 'work' / 'build_errors.txt'
        log.write_text('\n'.join(problems), encoding='utf-8-sig')
        raise ValueError(f'build发现{len(problems)}处问题。请查看 {log} 并修正entries.csv。前次work文件未覆盖。')
    if not inputs:
        raise ValueError('没有有效图文记录')
    inputs.sort(key=lambda r: r['sample_id'])
    labels.sort(key=lambda r: r['sample_id'])
    csv_write(root / 'work' / 'inputs.csv', inputs, INPUT_COLS)
    csv_write(root / 'work' / 'master_labels.csv', labels,
              ['sample_id'] + LABEL_COLS + ['label_status', 'adv_evidence', 'notes'])
    print(f'生成数据：{len(inputs)}条样本，{len({x["image_path"] for x in inputs})}张不同文件；sample_id稳定（由内容哈希生成）。')
    print('修改原始图片或文字会改变该条sample_id；冻结后不要随意改动。')


class DSU:
    def __init__(self, ids):
        self.parent = {v:v for v in ids}
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a,b):
        pa, pb = self.find(a), self.find(b)
        if pa != pb:
            self.parent[pb] = pa


def connected_components(inputs):
    dsu = DSU([r['sample_id'] for r in inputs])
    for field in ('image_group_id', 'request_group_id'):
        seen = {}
        for r in inputs:
            key = r[field]
            if key in seen:
                dsu.union(seen[key], r['sample_id'])
            else:
                seen[key] = r['sample_id']
    groups = defaultdict(list)
    for r in inputs:
        groups[dsu.find(r['sample_id'])].append(r['sample_id'])
    return list(groups.values())


def split_dataset(root, seed=42, reset=False):
    inputs = csv_read(root / 'work' / 'inputs.csv')
    if len(inputs) < 4:
        raise ValueError('至少4条独立候选数据才能分为4类用途；请继续收集并分组。')
    components = connected_components(inputs)
    if len(components) < 4:
        raise ValueError(f'当前只有{len(components)}个独立关联组，无法合理划成4份；检查是否复用了同一图片/请求。')
    target = {k:len(inputs)*v for k,v in SPLIT_RATIO.items()}
    label_map = {r['sample_id']: r.get('overall_risk_label', '')
                 for r in csv_read(root / 'work' / 'master_labels.csv')}
    label_totals = Counter(v for v in label_map.values() if v in ('0', '1'))
    strata_map = {}
    strata_file = root / 'work' / 'split_strata.csv'
    if strata_file.exists():
        for row in csv_read(strata_file):
            if row.get('sample_id') in label_map:
                strata_map[row['sample_id']] = (row.get('source_type', ''), row.get('category', ''))
    strata_totals = Counter(v for v in strata_map.values() if all(v))
    split_file = root / 'work' / 'splits.csv'
    old = {} if reset or not split_file.exists() else {
        r['sample_id']:r['split'] for r in csv_read(split_file)
    }
    if split_file.exists() and not reset:
        old_rows = csv_read(split_file)
        if len(old) != len(old_rows):
            raise ValueError('已有划分存在重复sample_id，停止写入')
        if set(old) - {r['sample_id'] for r in inputs}:
            raise ValueError('已有划分含当前输入中不存在的ID；需核对输入更正影响，不能静默删除')
    allocated = {}
    current = Counter()
    current_labels = {k: Counter() for k in SPLITS}
    current_strata = {k: Counter() for k in SPLITS}
    todo = []
    for comp in components:
        forced = {old[sid] for sid in comp if sid in old}
        if len(forced) > 1:
            raise ValueError('新增图片/请求组把已划分到不同集合的样本关联起来。请人工修正分组；不能自动跨集迁移。')
        if forced:
            choice = next(iter(forced))
            if choice not in SPLITS:
                raise ValueError(f'原划分中存在未知split: {choice}')
            for sid in comp:
                allocated[sid] = choice
                if label_map.get(sid) in ('0', '1'):
                    current_labels[choice][label_map[sid]] += 1
                if sid in strata_map:
                    current_strata[choice][strata_map[sid]] += 1
            current[choice] += len(comp)
        else:
            todo.append(comp)
    rng = random.Random(seed)
    rng.shuffle(todo)
    todo.sort(key=len, reverse=True)
    for comp in todo:
        # 若有整体风险真值，按组分配并尽量保持每个split的类别比例；
        # 没有真值时退化为按总量分配。关联组始终不可拆分。
        comp_labels = Counter(label_map.get(sid, '') for sid in comp)
        comp_strata = Counter(strata_map.get(sid) for sid in comp if sid in strata_map)
        def cost(k):
            size_before = (current[k] - target[k]) ** 2 / max(target[k], 1)
            size_after = (current[k] + len(comp) - target[k]) ** 2 / max(target[k], 1)
            size_cost = 0.12 * (size_after - size_before)
            label_cost = 0.0
            for label in ('0', '1'):
                if not label_totals[label]:
                    continue
                expected = label_totals[label] * SPLIT_RATIO[k]
                before = (current_labels[k][label] - expected) ** 2 / max(expected, 1)
                after = current_labels[k][label] + comp_labels[label]
                label_cost += (after - expected) ** 2 / max(expected, 1) - before
            strata_cost = 0.0
            for key, total in strata_totals.items():
                expected = total * SPLIT_RATIO[k]
                before = (current_strata[k][key] - expected) ** 2 / max(expected, 1)
                after = current_strata[k][key] + comp_strata[key]
                strata_cost += (after - expected) ** 2 / max(expected, 1) - before
            return size_cost + label_cost + strata_cost
        choice = min(SPLITS, key=lambda k: (cost(k), SPLITS.index(k)))
        for sid in comp:
            allocated[sid] = choice
            if label_map.get(sid) in ('0', '1'):
                current_labels[choice][label_map[sid]] += 1
            if sid in strata_map:
                current_strata[choice][strata_map[sid]] += 1
        current[choice] += len(comp)
    # 索引仍缺乏的类别将作为警告，由A判断数据是否足够
    csv_write(split_file,
              [{'sample_id':r['sample_id'], 'split':allocated[r['sample_id']]} for r in inputs],
              ['sample_id','split'])
    print('分组划分完成：', dict(current))
    print('默认保留已有划分；只有确认允许重划分才传 --reset。请保存固定划分。')
    if any(current[k]==0 for k in SPLITS):
        print('警告：至少有一个划分为空。请检查样本组分布后再正式交付。')


def check(root, verbose=True):
    inputs_file = root/'work'/'inputs.csv'
    labels_file = root/'work'/'master_labels.csv'
    splits_file = root/'work'/'splits.csv'
    for file in (inputs_file,labels_file,splits_file):
        if not file.exists():
            raise ValueError(f'文件不存在：{file.name}。先运行build和split。')
    ins = csv_read(inputs_file)
    labs = csv_read(labels_file)
    splits = csv_read(splits_file)
    sources = csv_read(root/'sources.csv')
    srcs = {r.get('source_id') for r in sources}
    errors, warnings, missing = [], [], []
    if not ins:
        errors.append('正式输入为空，不能打包')
    if len(srcs) != len(sources) or '' in srcs:
        errors.append('sources.csv存在空白或重复source_id')
    fingerprints = defaultdict(list)
    id_lists = [[r['sample_id'] for r in ls] for ls in (ins, labs, splits)]
    for typ,ids in zip(('inputs','master_labels','splits'), id_lists):
        if len(ids)!=len(set(ids)):
            errors.append(f'{typ}存在重复sample_id')
    if set(id_lists[0]) != set(id_lists[1]) or set(id_lists[0])!=set(id_lists[2]):
        errors.append('三个CSV的sample_id集合不一致，请重新build/split')
    sp = {r['sample_id']:r['split'] for r in splits}
    if set(sp.values()) - set(SPLITS):
        errors.append('splits.csv包含非法split')
    lab_map = {r['sample_id']:r for r in labs}
    for r in ins:
        sid=r['sample_id']
        relative = Path(r['image_path'])
        image_path = (root / relative).resolve()
        if relative.is_absolute() or not image_path.is_relative_to((root / 'images').resolve()):
            errors.append(f'{sid}：图片路径必须位于数据包images目录内')
            continue
        if not image_path.is_file():
            errors.append(f'{sid}：图片缺失 {r["image_path"]}')
            missing.append({'sample_id':sid,'issue':'image_missing','detail':r['image_path']})
        else:
            try:
                with Image.open(image_path) as im:
                    im.load()
                    rgb = im.convert('RGB')
                    pixel_hash = sha_bytes(str(rgb.size).encode() + rgb.tobytes())
                fingerprints[('bytes', sha_file(image_path))].append(r)
                fingerprints[('pixels', pixel_hash)].append(r)
            except Exception as exc:
                errors.append(f'{sid}：图片无法解码 {exc}')
                missing.append({'sample_id':sid,'issue':'image_broken','detail':str(exc)})
        if not r['user_text'].strip():
            errors.append(f'{sid}：用户文本为空')
        if r['source_id'] not in srcs:
            errors.append(f'{sid}：source_id={r["source_id"]} 未在sources.csv登记')
        l=lab_map.get(sid)
        if l is None: continue
        for col in LABEL_COLS:
            if l[col] not in ('', '0', '1'):
                errors.append(f'{sid}：{col}只能为0、1或留空')
        if l['label_status'] not in ('verified','review','pending'):
            errors.append(f'{sid}：label_status只能为verified/review/pending')
        if l['label_status'] != 'verified':
            warnings.append(f'{sid}：标签状态{l["label_status"]}，不得计入正式真值统计')
            missing.append({'sample_id':sid,'issue':'label_unverified','detail':l['label_status']})
        if l['label_status']=='verified' and l['overall_risk_label']=='':
            errors.append(f'{sid}：已核实的整体风险标签不能为空')
        if l['image_adv_label']=='1' and not l['adv_evidence']:
            errors.append(f'{sid}：对抗图像标签为1时必须提供adv_evidence证明')
        if l['overall_risk_label']=='0' and any(l[k]=='1' for k in LABEL_COLS if k!='image_adv_label' and k!='overall_risk_label'):
            warnings.append(f'{sid}：overall=0但文本风险轴为1，需检查标签映射规则')
    for src in sources:
        if not src.get('source_url','').strip() and not src.get('notes','').strip():
            warnings.append(f'来源{src.get("source_id")}：缺少URL及说明，无法追溯')
        if not src.get('license','').strip():
            warnings.append(f'来源{src.get("source_id")}：未记录使用许可')
    for comp in connected_components(ins):
        assignments={sp.get(sid) for sid in comp}
        if len(assignments)>1:
            errors.append('同一图像组/请求组跨多个split：'+','.join(comp[:4]))
    # 检查内容而非路径，识别改名副本和无损重新编码的副本。
    for (kind, digest), items in fingerprints.items():
        groups={r['image_group_id'] for r in items}
        affected={sp.get(r['sample_id']) for r in items}
        if len(affected)>1:
            errors.append(f'相同图片内容({kind}:{digest[:12]})跨数据划分')
        if len(groups)>1:
            warnings.append(f'相同图片内容({kind}:{digest[:12]})分属不同image_group_id，请统一')
    requests=defaultdict(list)
    for r in ins:
        requests[normalized_text(r['user_text'])].append(r)
    for txt, items in requests.items():
        groups={r['request_group_id'] for r in items}
        if len(groups)>1:
            if len({sp.get(r['sample_id']) for r in items})>1:
                errors.append('相同用户请求跨划分，即使人工填写的request_group_id不同也属于泄漏：'+txt[:35])
            else:
                warnings.append('相同用户请求使用了不同request_group_id：'+txt[:35])
    counts=Counter(sp.values())
    for s in SPLITS:
        if counts[s]==0:
            warnings.append(f'{s}中没有样本，需补充或人工调整分组')
    report=[f'生成时间：{datetime.now().isoformat(timespec="seconds")}',
            f'总样本：{len(ins)}；不同图片：{len({r["image_path"] for r in ins})}；独立关联组：{len(connected_components(ins))}',
            f'划分统计：{json.dumps(dict(counts), ensure_ascii=False)}',
            f'verified标签数：{sum(x["label_status"]=="verified" for x in labs)}',
            f'错误数：{len(errors)}；提醒数：{len(warnings)}',
            '\n[错误（必须修复）]']+[f'- {x}' for x in errors]+['\n[提醒（人工复核）]']+[f'- {x}' for x in warnings]
    out=root/'work'/'qc_report.txt'
    out.write_text('\n'.join(report)+'\n',encoding='utf-8-sig')
    csv_write(root/'work'/'missing_log.csv',missing,['sample_id','issue','detail'])
    stats=[{'split':k,'samples':counts[k],
            'verified':sum(sp.get(l['sample_id'])==k and l['label_status']=='verified' for l in labs),
            'overall_positive_verified':sum(sp.get(l['sample_id'])==k and l['label_status']=='verified' and l['overall_risk_label']=='1' for l in labs)} for k in SPLITS]
    csv_write(root/'work'/'stats.csv',stats,['split','samples','verified','overall_positive_verified'])
    if verbose:
        print('\n'.join(report[:5]))
        print(f'详情：{out}')
        if errors:
            print('错误示例：',errors[:4])
        if warnings:
            print('提醒示例：',warnings[:4])
    return errors,warnings


def package(root, draft=False):
    errors, warnings=check(root)
    if errors:
        raise ValueError(f'存在{len(errors)}个严重错误，请先修复后打包。')
    ins=csv_read(root/'work'/'inputs.csv')
    spl=csv_read(root/'work'/'splits.csv')
    labs=csv_read(root/'work'/'master_labels.csv')
    sp={r['sample_id']:r['split'] for r in spl}
    valid={r['sample_id']:r for r in labs if r['label_status']=='verified'}
    from delivery_tools import release_issues, export_d_manifests
    issues = release_issues(ins, labs, spl)
    if issues and not draft:
        raise ValueError('正式交付条件未满足：' + '；'.join(issues) + '。接口试跑可用package --draft，不能作为正式实验数据。')
    name = 'team_dataset_DRAFT' if draft else 'team_dataset_v1'
    package_dir=root/'output'/name
    zip_path=root/'output'/f'{name}.zip'
    if package_dir.exists() or zip_path.exists():
        raise ValueError('输出包已存在，停止覆盖；请先另存旧版本')
    (package_dir/'images').mkdir(parents=True)
    for img in sorted({r['image_path'] for r in ins}):
        shutil.copy2(root/img,package_dir/img)
    csv_write(package_dir/'inputs.csv',ins,INPUT_COLS)
    csv_write(package_dir/'splits.csv',spl,['sample_id','split'])
    csv_write(package_dir/'sources.csv',csv_read(root/'sources.csv'),SOURCE_COLS)
    b=[{'sample_id':sid,'image_adv_label':l['image_adv_label']} for sid,l in valid.items()
       if sp[sid] in ('base_train','val') and l['image_adv_label']!='']
    c=[{'sample_id':sid,**{k:l[k] for k in ('harmful_content_label','jailbreak_intent_label','prompt_injection_label','overall_risk_label')}}
       for sid,l in valid.items() if sp[sid] in ('base_train','val')]
    d=[{'sample_id':sid,'overall_risk_label':l['overall_risk_label']} for sid,l in valid.items()
       if sp[sid] in ('fusion_train','val') and l['overall_risk_label']!='']
    csv_write(package_dir/'labels_for_B.csv',b,['sample_id','image_adv_label'])
    csv_write(package_dir/'labels_for_C.csv',c,['sample_id','harmful_content_label','jailbreak_intent_label','prompt_injection_label','overall_risk_label'])
    csv_write(package_dir/'labels_for_D.csv',d,['sample_id','overall_risk_label'])
    csv_write(package_dir/'B_prediction_template.csv',[],['sample_id','image_score','image_pred','ocr_text','ocr_status'])
    csv_write(package_dir/'C_prediction_template.csv',[],['sample_id','user_text_score','ocr_text_score','combined_text_score','text_pred','ocr_status'])
    csv_write(package_dir/'D_prediction_template.csv',[],['sample_id','fusion_score','fusion_pred','fusion_mode'])
    readme='''# team_dataset_v1（A第三周交付）

- inputs.csv 每行一组图文，sample_id唯一。image_path相对于本目录。
- splits.csv：base_train（基础模型训练），fusion_train（融合器训练），val（选择超参数/阈值），test（最终冻结评测）。
- B用labels_for_B.csv，C用labels_for_C.csv，D用labels_for_D.csv。仅提供相应训练/验证真值；当前HoliSafe没有经过核验的image_adv_label，因此labels_for_B.csv可能只有表头，不能把空表当作负例。
- 最终测试集真值由A独立保存；严禁使用test标签调整模型/阈值。
- 预测文件按各自template.csv填写，必须以sample_id连接，禁止仅依赖行序。
- OCR无法识别和OCR未运行/失败不是一回事，需在ocr_status中明确记录。
- image_score用于对抗图像分支；fusion_score用于整体风险，不可混用真值。当前包没有图像对抗监督真值。
- 请先核验本数据包实际样本数和每一类可用真值数量；空的训练标签不代表负例。
- 图像组/请求组由来源ID、内容哈希和固定近似启发式生成；近似关系不是人工语义审核。原始图片/文字来自sources.csv所记载来源。
- CIDER原仓库图文格式与本课程统一sample_id接口不完全相同；B负责实际适配。
'''
    readme += '''
## 本版本的数据与标签边界

- 本包包含4,024条去重后的HoliSafe图文输入和1,787张图片，来自HoliSafe数据卡固定revision；课程划分为base_train=2,002、fusion_train=808、val=605、test=609。
- HoliSafe原始样本是官方test基准。本包按图片/请求组重新生成的划分属于课程派生协议，不是作者官方划分，也不能报告为官方协议结果。
- `harmful_content_label`映射HoliSafe官方`type`第二位（query safety）；`overall_risk_label`映射第三位（final input safety）。这些是来源标签，不是本组模型预测或目标模型越狱成功标签。
- `image_adv_label`、`jailbreak_intent_label`及`prompt_injection_label`在本数据源中未知，空值不是负例。该数据可评估图文安全风险，不能单独支持越狱意图、提示注入或图像对抗扰动的准确率结论。
- 5个原图路径在18条源记录中有相互冲突的`image_safe`/`image_safety_label`值；原始标签留在A侧映射，不作为本包的图像分类或对抗真值。
- 近似图像及近似请求按固定启发式并入同一划分组；这不是人工语义复核。细则、候选边和非测试集来源ID见`D_INTERFACE_NOTES.md`及`provenance_train_validation.csv`。
- HoliSafe包含敏感图像。仅限获准的学术安全研究成员使用；不得公开上传或转发数据图片。图片仍受各自原始来源许可约束，使用前需遵守数据卡和原始来源条款。
- JailBreakV-28K目前未纳入本包：本地完整CSV的28,000行中仅360行有可读配图。
'''
    (package_dir/'README.md').write_text(readme,encoding='utf-8')
    if draft:
        (package_dir/'DRAFT_NOT_FOR_EVALUATION.txt').write_text('\n'.join(issues) or '调试包，不是正式评测数据', encoding='utf-8')
    export_d_manifests(package_dir, ins, labs, spl)
    source_records = root/'work'/'holisafe_source_records.csv'
    if source_records.is_file():
        test_ids = {r['sample_id'] for r in spl if r['split'] == 'test'}
        provenance = [r for r in csv_read(source_records) if r['sample_id'] not in test_ids]
        csv_write(package_dir/'provenance_train_validation.csv', provenance,
                  list(provenance[0]) if provenance else ['sample_id'])
        rules = root/'work'/'grouping_rules.json'
        if rules.is_file(): shutil.copy2(rules, package_dir/'GROUPING_RULES.json')
    restrictions = (
        '# Restricted source data handling\n\n'
        'This package contains sensitive HoliSafe research data. Restrict access to authorized project members; '
        'do not publish, mirror, or forward the images. Use only for academic AI-safety evaluation and follow '
        'the gated dataset terms. Generated text and generated images are CC BY-NC 4.0; images sourced from '
        'third-party datasets keep their original licenses. Check those source licenses before any redistribution. '
        'Cite the HoliSafe paper listed in the official dataset README: https://arxiv.org/abs/2506.04704.\n'
    )
    (package_dir/'SOURCE_DATA_RESTRICTIONS.md').write_text(restrictions, encoding='utf-8')
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package_dir.rglob('*')):
            if p.is_file():
                z.write(p,p.relative_to(package_dir.parent))
    # private label export is intentionally outside the shared ZIP
    private=root/'output'/'A_private_DO_NOT_SHARE'
    private.mkdir(exist_ok=True)
    csv_write(private/'master_labels.csv',labs,['sample_id']+LABEL_COLS+['label_status','adv_evidence','notes'])
    csv_write(private/'splits.csv',spl,['sample_id','split'])
    if source_records.is_file(): shutil.copy2(source_records, private/'holisafe_source_records.csv')
    if (root/'work'/'grouping_rules.json').is_file(): shutil.copy2(root/'work'/'grouping_rules.json', private/'grouping_rules.json')
    shutil.copy2(root/'work'/'qc_report.txt',private/'qc_report.txt')
    print(f'团队交付包: {zip_path}')
    print(f'仅A保管的真值: {private}')
    print(f'B可训练标签{len(b)}条，C可训练标签{len(c)}条，D可训练标签{len(d)}条')
    print('切勿把A_private_DO_NOT_SHARE文件夹一起发给组员。')


def selftest():
    with tempfile.TemporaryDirectory(prefix='a_week3_selftest_') as td:
        root=Path(td)
        ensure_init(root)
        csv_write(root/'sources.csv',[{'source_id':'SRC_DEMO','source_name':'临时测试图片',
              'source_url':'','license':'自建程序测试，不可作为实验数据',
              'access_status':'available','notes':'仅单元自检，不会写入你真实项目'}], SOURCE_COLS)
        entries=[]
        for i in range(16):
            img=root/f'im{i:02d}.png'
            Image.new('RGB',(40,40),color=((i*11)%255,(i*19)%255,(i*7)%255)).save(img)
            row=dict.fromkeys(ENTRY_COLS,'')
            row.update(image_file=str(img),user_text=f'程序自动测试样本编号{i}的描述请求',
                       source_id='SRC_DEMO',image_adv_label='0',
                       harmful_content_label='0',jailbreak_intent_label='0',
                       prompt_injection_label='0',overall_risk_label='0',
                       label_status='verified')
            entries.append(row)
        csv_write(root/'entries.csv',entries,ENTRY_COLS)
        build(root)
        split_dataset(root)
        errors,_=check(root,False)
        if errors: raise AssertionError(errors)
        package(root, draft=True)
        # Ensure test labels are absent from all shared files
        shared=root/'output'/'team_dataset_DRAFT'
        assert len(csv_read(shared/'inputs.csv'))==16
        assert len(csv_read(shared/'labels_for_D.csv'))<16
        assert not (shared/'master_labels.csv').exists()
        assert (root/'output'/'team_dataset_DRAFT.zip').is_file()
    print('自检成功：16条临时示例完成build/split/check/package；测试数据已自动清理。')


def main():
    parser=argparse.ArgumentParser(description='A第三周图文数据整理工具（项目根目录为week3.py所在文件夹）')
    sub=parser.add_subparsers(dest='cmd',required=True)
    sub.add_parser('init',help='创建空CSV与工作文件夹，不覆盖已填数据')
    p=sub.add_parser('scan-cider',help='核查已下载CIDER仓库结构，不自动标注')
    p.add_argument('--repo',required=True,help='CIDER源码仓库根目录')
    p=sub.add_parser('import-pairs',help='从图片+文字CSV批量录入待审核图文')
    for arg in ('csv','image-root','image-col','text-col','source-id'):
        p.add_argument('--'+arg,required=True)
    p=sub.add_parser('import-pairing-plan', help='按资源ID导入人工审核配对计划；默认不覆盖已有记录')
    p.add_argument('--plan', required=True)
    p.add_argument('--images', default='output/cider_ready/images_unique_review.csv')
    p.add_argument('--texts', default='output/cider_ready/texts_full_review.csv')
    p.add_argument('--source-id', default='')
    p.add_argument('--dry-run', action='store_true')
    p=sub.add_parser('add-sample',help='在命令行添加一条图文，减少手改CSV的操作')
    p.add_argument('--image',required=True)
    p.add_argument('--text',required=True)
    p.add_argument('--source-id',required=True)
    p.add_argument('--image-group',default='')
    p.add_argument('--request-group',default='')
    for name in ('image-adv','harmful','jailbreak','prompt-injection','overall'):
        p.add_argument('--'+name,default='',choices=('', '0', '1'))
    p.add_argument('--status',default='pending',choices=('pending','review','verified'))
    p.add_argument('--adv-evidence',default='')
    p.add_argument('--notes',default='')
    sub.add_parser('build',help='验证图片并生成稳定编号、数据表和私有标签')
    p=sub.add_parser('split',help='按图片/请求关联组划分4种用途')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--reset',action='store_true',help='危险：主动重划分（冻结后不要使用）')
    sub.add_parser('check',help='检查路径、标签、来源、划分和缺失项')
    p=sub.add_parser('package',help='为B/C/D导出不包含test真值的ZIP包')
    p.add_argument('--draft', action='store_true', help='生成单独命名的接口调试包，不能用于正式评测')
    sub.add_parser('selftest',help='临时生成测试图片验证整条管线，不污染真实数据')
    args=parser.parse_args()
    try:
        if args.cmd=='init': ensure_init(ROOT)
        elif args.cmd=='scan-cider': scan_cider(ROOT,Path(args.repo))
        elif args.cmd=='import-pairs': add_pairs(ROOT,args)
        elif args.cmd=='import-pairing-plan': import_pairing_plan(ROOT,args)
        elif args.cmd=='add-sample': add_sample(ROOT,args)
        elif args.cmd=='build': build(ROOT)
        elif args.cmd=='split': split_dataset(ROOT,args.seed,args.reset)
        elif args.cmd=='check':
            errors, _ = check(ROOT)
            if errors: sys.exit(1)
        elif args.cmd=='package': package(ROOT, draft=args.draft)
        elif args.cmd=='selftest': selftest()
    except (ValueError,OSError,AssertionError) as e:
        print(f'操作未完成：{e}',file=sys.stderr)
        sys.exit(1)


if __name__=='__main__':
    main()
