import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from mm_jailbreak_guard.cli import main
from mm_jailbreak_guard.contracts import DetectionRequest, ImageEvidence, TextEvidence
from mm_jailbreak_guard.fusion import FusionBundle, TuningExample
from mm_jailbreak_guard.io import load_jsonl, write_json, write_jsonl
from mm_jailbreak_guard.learned_fusion import (LearnedFusionEngine, train_bundle,
                                               train_logistic, select_threshold)
from mm_jailbreak_guard.training_io import adapt_manifest, validate_training_manifests

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/fusion_config.example.json'


def examples(prefix='t'):
    return [TuningExample(prefix+str(i), label, {
        'user_text_score': value, 'ocr_text_score': None, 'combined_text_score': None,
        'visual_score': None, 'cross_modal_score': None})
        for i, (label, value) in enumerate(((0, .05), (0, .15), (1, .85), (1, .95)))]


class LearnedFusionTests(unittest.TestCase):
    def test_first_gradient_step_matches_analytic_solution(self):
        rows = [TuningExample('a', 0, {'x': 0.0}), TuningExample('b', 1, {'x': 1.0})]
        model = train_logistic(rows, ['x'], epochs=1, learning_rate=.2, l2=0)
        self.assertAlmostEqual(model.weights[0], .05)
        self.assertAlmostEqual(model.bias, 0.0)
        self.assertAlmostEqual(model.weights[1], 0.0)

    def test_learns_signal_and_reduces_loss_deterministically(self):
        model = train_logistic(examples(), ['user_text_score'], epochs=250)
        self.assertGreater(model.score({'user_text_score': .9}), model.score({'user_text_score': .1}))
        self.assertGreater(model.score({'user_text_score': .9}), .5)
        self.assertLess(model.score({'user_text_score': .1}), .5)
        self.assertLess(model.loss_history[-1], model.loss_history[0])
        self.assertEqual(asdict(model), asdict(train_logistic(examples(), ['user_text_score'], epochs=250)))

    def test_validation_changes_no_learned_parameter_or_imputation(self):
        model = train_logistic(examples(), ['user_text_score', 'visual_score'])
        before = copy.deepcopy(asdict(model))
        validation = examples('v')
        for row in validation:
            row.features['visual_score'] = .99
        with self.assertRaisesRegex(ValueError, '训练从未覆盖'):
            select_threshold(model, validation)
        for row in validation:
            row.features['visual_score'] = None
        select_threshold(model, validation)
        self.assertEqual(before, asdict(model))
        self.assertEqual(model.means[1], 0.0)

    def test_bundle_roundtrip_missing_evidence_and_version_guard(self):
        payload = train_bundle(examples(), examples('v'), FusionBundle.load(CONFIG),
                               provenance={'module_versions': {'C': ['C1']}}, epochs=100)
        engine = LearnedFusionEngine(json.loads(json.dumps(payload)))
        request = DetectionRequest('x', '普通占位')
        image = ImageEvidence.unavailable('x', '没有图像模块')
        result = engine.fuse(request, image, TextEvidence('x', True, .9, module_version='C1'))
        self.assertEqual(result.prediction, 1)
        self.assertEqual(result.effective_weights, {})
        self.assertIn('logit_contributions', result.decision_rule)
        no_text = engine.fuse(request, image, TextEvidence.unavailable('x', '缺失'))
        self.assertIsNone(no_text.prediction)
        changed = engine.fuse(request, image, TextEvidence('x', True, .9, module_version='C2'))
        self.assertIsNone(changed.prediction)
        self.assertIn('版本', ''.join(changed.warnings))
        unseen = engine.fuse(request, ImageEvidence('x', True, visual_score=.7),
                             TextEvidence('x', True, .9, module_version='C1'))
        self.assertIsNone(unseen.prediction)
        self.assertIn('未覆盖', ''.join(unseen.warnings))

    def test_invalid_training_settings_and_single_class_rejected(self):
        for options in ({'epochs': 0}, {'learning_rate': float('nan')}, {'l2': -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                train_logistic(examples(), ['user_text_score'], **options)
        with self.assertRaises(ValueError):
            train_logistic(examples()[:2], ['user_text_score'])

    def prepare(self, folder):
        rows = load_jsonl(ROOT / 'examples/handoff/manifest.jsonl')
        train = [r for r in rows if r['split'] == 'fusion_train']
        val = [r for r in rows if r['split'] == 'validation']
        a, b = folder / 'train.jsonl', folder / 'val.jsonl'
        write_jsonl(a, train)
        write_jsonl(b, val)
        return a, b

    def test_training_rejects_wrong_splits_unknown_labels_and_group_leak(self):
        for change in ('test', 'base_train', 'label', 'group', 'request'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                a, b = self.prepare(Path(tmp))
                train, val = load_jsonl(a), load_jsonl(b)
                if change in {'test', 'base_train'}:
                    train[0]['split'] = change
                elif change == 'label':
                    train[0]['label'] = None
                elif change == 'group':
                    val[0]['image_group_id'] = train[0]['image_group_id']
                else:
                    val[0]['user_text'] = train[0]['user_text']
                write_jsonl(a, train)
                write_jsonl(b, val)
                with self.assertRaises(ValueError):
                    validate_training_manifests(a, b)

    def test_A_export_adapter_preserves_original_and_image_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'original.jsonl'
            row = dict(sample_id='A1', image_path='images/a.png', user_text='', split='val', source='A')
            write_jsonl(source, [row])
            original = source.read_bytes()
            groups = root / 'groups.csv'
            with groups.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['sample_id', 'image_group_id', 'request_group_id'])
                writer.writeheader()
                writer.writerow(dict(sample_id='A1', image_group_id='i1', request_group_id='r1'))
            target = root / 'D' / 'adapted.jsonl'
            adapt_manifest(source, groups, target)
            adapted = load_jsonl(target)[0]
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(adapted['split'], 'validation')
            self.assertEqual((target.parent / adapted['image_path']).resolve(), (root / row['image_path']).resolve())
            with self.assertRaises(ValueError):
                adapt_manifest(source, groups, source)

    def test_cli_train_compare_batch_and_prediction_consistency(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(tmp)
            a, b = self.prepare(folder)
            config = folder / 'learned.json'
            evidence = ['--image-evidence', str(ROOT / 'examples/handoff/image_evidence.jsonl'),
                        '--text-evidence', str(ROOT / 'examples/handoff/text_evidence.jsonl')]
            cached_image, cached_text = folder/'B.jsonl', folder/'C.jsonl'
            self.assertEqual(main(['collect-evidence', '--manifest', str(ROOT/'examples/handoff/manifest.jsonl'),
                                   *evidence, '--config', str(CONFIG), '--image-output', str(cached_image),
                                   '--text-output', str(cached_text)]), 0)
            self.assertEqual(len(load_jsonl(cached_image)), 6)
            self.assertFalse(load_jsonl(cached_text)[-1]['available'])
            self.assertEqual(main(['train-fusion', '--train-manifest', str(a), '--validation-manifest', str(b),
                                   *evidence, '--baseline-config', str(CONFIG), '--epochs', '100', '--output', str(config)]), 0)
            payload = json.loads(config.read_text(encoding='utf-8'))
            self.assertFalse(payload['provenance']['test_set_used'])
            batch, compare = folder / 'batch.jsonl', folder / 'compare.jsonl'
            common = ['--manifest', str(b), *evidence, '--config', str(config)]
            self.assertEqual(main(['batch', *common, '--output', str(batch)]), 0)
            self.assertEqual(main(['compare', *common, '--baseline-config', str(CONFIG), '--output', str(compare)]), 0)
            for batch_row, comparison in zip(load_jsonl(batch), load_jsonl(compare)):
                self.assertEqual(batch_row['risk_score'], comparison['schemes']['full']['risk_score'])
                self.assertEqual(set(comparison['fixed_baseline_schemes']), {'image_only', 'text_only', 'full'})
