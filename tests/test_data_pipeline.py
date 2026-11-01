import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image
import week3 as w


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)
        w.ensure_init(self.root)
        w.csv_write(self.root/'sources.csv', [dict(source_id='UNIT', source_name='Temporary fixtures',
                    source_url='', license='test only', access_status='available', notes='not real samples')], w.SOURCE_COLS)
        rows = []
        for i in range(16):
            path = self.root/f'{i}.png'
            Image.new('RGB', (8, 8), (i*10, 20, 30)).save(path)
            row = dict.fromkeys(w.ENTRY_COLS, '')
            row.update(image_file=str(path), user_text=f'unit-test input {i}', source_id='UNIT',
                       image_group_id=f'I{i}', request_group_id=f'R{i}', label_status='verified',
                       overall_risk_label=str(i % 2), harmful_content_label=str(i % 2),
                       image_adv_label='0', notes='test fixture only')
            rows.append(row)
        self.rows = rows

    def build_split(self):
        w.csv_write(self.root/'entries.csv', self.rows, w.ENTRY_COLS)
        w.build(self.root)
        inputs = w.csv_read(self.root/'work/inputs.csv')
        # 两类样本在每个集合各两个，测试打包边界，不测试分层算法本身。
        by_text = {r['user_text']: r['sample_id'] for r in inputs}
        split_rows = [dict(sample_id=by_text[r['user_text']], split=w.SPLITS[i//4]) for i, r in enumerate(self.rows)]
        w.csv_write(self.root/'work/splits.csv', split_rows, ['sample_id', 'split'])
        return inputs

    def plan_args(self):
        image = dict(image_resource_id='I', image_file=self.rows[0]['image_file'], label_evidence='',
                     visual_family_group_id_MANUAL='I0')
        text = dict(text_resource_id='T', behavior_text='unit-test input 0', behavior_type_col2='standard')
        plan = dict(image_resource_id='I', text_resource_id='T', pairing_type='official_pair',
                    pairing_evidence_or_rule='unit fixture', pairing_status='verified', label_status='pending',
                    image_group_id_MANUAL='I0', request_group_id_MANUAL='R0', source_id='UNIT')
        for name, row in [('images.csv', image), ('texts.csv', text), ('plan.csv', plan)]:
            w.csv_write(self.root/name, [row], list(row))
        return argparse.Namespace(plan=str(self.root/'plan.csv'), images=str(self.root/'images.csv'),
                                  texts=str(self.root/'texts.csv'), source_id='UNIT', dry_run=True)

    def test_plan_dry_run_idempotence_and_conflict(self):
        args = self.plan_args()
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        w.import_pairing_plan(self.root, args)
        self.assertEqual(before, {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        args.dry_run = False
        w.import_pairing_plan(self.root, args)
        first = (self.root/'entries.csv').read_bytes()
        w.import_pairing_plan(self.root, args)
        self.assertEqual(first, (self.root/'entries.csv').read_bytes())
        plan = w.csv_read(self.root/'plan.csv')[0]
        plan['request_group_id_MANUAL'] = 'conflict'
        w.csv_write(self.root/'plan.csv', [plan], list(plan))
        with self.assertRaisesRegex(ValueError, '冲突'):
            w.import_pairing_plan(self.root, args)
        self.assertEqual(first, (self.root/'entries.csv').read_bytes())

    def test_plan_requires_registered_source_and_confirmed_pair(self):
        args = self.plan_args()
        plan = w.csv_read(self.root/'plan.csv')[0]
        plan['pairing_status'] = 'pending'
        w.csv_write(self.root/'plan.csv', [plan], list(plan))
        with self.assertRaisesRegex(ValueError, 'pairing_status'):
            w.import_pairing_plan(self.root, args)
        plan.update(pairing_status='verified', source_id='UNKNOWN')
        w.csv_write(self.root/'plan.csv', [plan], list(plan))
        with self.assertRaisesRegex(ValueError, '未在sources'):
            w.import_pairing_plan(self.root, args)

    def test_context_and_duplicate_resource_guard(self):
        args = self.plan_args()
        row = w.csv_read(self.root/'texts.csv')[0]
        row['behavior_type_col2'] = 'contextual'
        w.csv_write(self.root/'texts.csv', [row], list(row))
        with self.assertRaisesRegex(ValueError, '完整上下文'):
            w.import_pairing_plan(self.root, args)
        w.csv_write(self.root/'texts.csv', [row, row], list(row))
        with self.assertRaisesRegex(ValueError, '重复'):
            w.import_pairing_plan(self.root, args)

    def test_release_and_all_shared_labels_exclude_test(self):
        inputs = self.build_split()
        splits = w.csv_read(self.root/'work/splits.csv')
        w.csv_write(self.root/'work/holisafe_source_records.csv',
                    [dict(sample_id=r['sample_id'], original_id=r['sample_id'],
                          source_type=next(x['overall_risk_label'] for x in w.csv_read(self.root/'work/master_labels.csv')
                                           if x['sample_id']==r['sample_id'])) for r in splits],
                    ['sample_id','original_id','source_type'])
        w.package(self.root)
        labels = w.csv_read(self.root/'work/master_labels.csv')
        test_ids = {r['sample_id'] for r in w.csv_read(self.root/'work/splits.csv') if r['split']=='test'}
        with zipfile.ZipFile(self.root/'output/team_dataset_v1.zip') as archive:
            self.assertFalse(any('master_labels' in n or 'A_private' in n for n in archive.namelist()))
            self.assertIn('team_dataset_v1/SOURCE_DATA_RESTRICTIONS.md', archive.namelist())
            with archive.open('team_dataset_v1/provenance_train_validation.csv') as stream:
                provenance_text = stream.read().decode('utf-8-sig')
            self.assertFalse(any(sid in provenance_text for sid in test_ids))
            for name in archive.namelist():
                if '/labels_for_' in name:
                    content = archive.read(name).decode('utf-8-sig')
                    self.assertFalse(any(sid in content for sid in test_ids))
                if name.endswith('.jsonl'):
                    records = [json.loads(line) for line in archive.read(name).decode('utf-8').splitlines()]
                    for row in records:
                        if row['sample_id'] in test_ids:
                            self.assertNotIn('label', row)
                        self.assertFalse(Path(row['image_path']).is_absolute())
                        self.assertIn('team_dataset_v1/' + row['image_path'], archive.namelist())
            fusion = [json.loads(line) for line in archive.read('team_dataset_v1/manifest.fusion_train.jsonl').decode().splitlines()]
            self.assertTrue(fusion)
            self.assertTrue(all(r['split']=='fusion_train' and type(r['label']) is int for r in fusion))
        self.assertEqual(len(inputs), len(labels))
        with self.assertRaisesRegex(ValueError, '已存在'):
            w.package(self.root)

    def test_single_class_release_rejected(self):
        for row in self.rows:
            row.update(overall_risk_label='0', harmful_content_label='0')
        self.build_split()
        with self.assertRaisesRegex(ValueError, '正类或负类'):
            w.package(self.root)
        self.assertFalse((self.root/'output/team_dataset_v1.zip').exists())

    def test_renamed_and_reencoded_images_leak(self):
        inputs = self.build_split()
        target_texts = ('unit-test input 0', 'unit-test input 4')
        first, second = [next(r for r in inputs if r['user_text']==text) for text in target_texts]
        alias = self.root/'images/alias.bmp'
        with Image.open(self.root/first['image_path']) as image:
            image.save(alias)
        second['image_path'] = 'images/alias.bmp'
        w.csv_write(self.root/'work/inputs.csv', inputs, w.INPUT_COLS)
        errors, _ = w.check(self.root, verbose=False)
        self.assertTrue(any('pixels' in error and '跨' in error for error in errors))

    def test_stale_split_not_silently_removed(self):
        self.build_split()
        path = self.root/'work/splits.csv'
        rows = w.csv_read(path)
        rows.append(dict(sample_id='old_deleted_id', split='test'))
        w.csv_write(path, rows, ['sample_id', 'split'])
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, '不存在的ID'):
            w.split_dataset(self.root)
        self.assertEqual(before, path.read_bytes())

    def test_split_uses_source_strata_and_keeps_each_group_together(self):
        types = ('SSS','SSU','SUU','USU','UUU')
        entries, strata = [], []
        for i in range(40):
            path = self.root/f'stratum_{i}.png'
            Image.new('RGB', (8, 8), (i*5, 30, 70)).save(path)
            source_type = types[i % len(types)]
            category = f'category_{i % 4}'
            row = dict.fromkeys(w.ENTRY_COLS, '')
            row.update(image_file=str(path), user_text=f'stratum input {i}', source_id='UNIT',
                       image_group_id=f'IG{i//2}', request_group_id=f'RG{i}',
                       overall_risk_label='0' if source_type.endswith('S') else '1',
                       label_status='verified')
            entries.append(row)
            sid = 'S' + w.sha_bytes((w.sha_file(path)+'\0'+row['user_text']+'\0UNIT').encode())[:12].upper()
            strata.append(dict(sample_id=sid, source_type=source_type, category=category))
        w.csv_write(self.root/'entries.csv', entries, w.ENTRY_COLS)
        w.csv_write(self.root/'work/split_strata.csv', strata, ['sample_id','source_type','category'])
        w.build(self.root)
        w.split_dataset(self.root, seed=8, reset=True)
        inputs = w.csv_read(self.root/'work/inputs.csv')
        split = {r['sample_id']:r['split'] for r in w.csv_read(self.root/'work/splits.csv')}
        strata_by_id = {r['sample_id']:r for r in w.csv_read(self.root/'work/split_strata.csv')}
        for name in w.SPLITS:
            ids = [sid for sid, part in split.items() if part == name]
            self.assertEqual({strata_by_id[sid]['source_type'] for sid in ids}, set(types))
            self.assertEqual({strata_by_id[sid]['category'] for sid in ids},
                             {f'category_{i}' for i in range(4)})
        groups = w.connected_components(inputs)
        self.assertTrue(all(len({split[sid] for sid in group}) == 1 for group in groups))


if __name__ == '__main__':
    unittest.main()
