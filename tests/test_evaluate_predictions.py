import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import evaluate_predictions as e


def dataset(truths, states=None):
    states = states or ['verified'] * len(truths)
    inputs, labels, splits = {}, {}, {}
    for i, (truth, status) in enumerate(zip(truths, states)):
        sid = f'fixture-{i}'
        inputs[sid] = dict(sample_id=sid, image_group_id=f'I{i}', request_group_id=f'R{i}')
        labels[sid] = dict(sample_id=sid, overall_risk_label=truth, image_adv_label='', label_status=status)
        splits[sid] = dict(sample_id=sid, split='val')
    return dict(inputs=inputs, labels=labels, splits=splits)


class MetricsTests(unittest.TestCase):
    def test_hand_calculated_confusion(self):
        result = e.binary_metrics([(1, 1), (1, 1), (0, 0), (0, 1), (1, 0), (1, 0)])
        self.assertEqual([result[k] for k in ('tp', 'tn', 'fp', 'fn')], [2, 1, 1, 2])
        for key, expected in dict(accuracy=.5, precision=2/3, recall=.5, f1=4/7,
                                  fpr=.5, fnr=.5, balanced_accuracy=.5).items():
            self.assertAlmostEqual(result[key], expected)

    def test_perfect_all_positive_all_negative_empty(self):
        for predicted, expected in (([0, 1], 1), ([1, 1], .5), ([0, 0], .5)):
            self.assertEqual(e.binary_metrics(zip([0, 1], predicted))['accuracy'], expected)
        self.assertIsNone(e.binary_metrics([])['accuracy'])
        safe_only = e.binary_metrics([(0, 0)])
        self.assertIsNone(safe_only['recall'])
        self.assertIsNone(safe_only['precision'])
        self.assertIsNone(safe_only['f1'])
        self.assertIsNone(safe_only['balanced_accuracy'])
        self.assertEqual(e.binary_metrics([(1, 0)])['f1'], 0)

    def test_unknown_unverified_missing_and_abstained_are_separate(self):
        data = dataset(['1', '0', '1', '0', '', '1'], ['verified'] * 5 + ['pending'])
        predictions = e.index_rows([dict(sample_id='fixture-0', prediction=1),
                                    dict(sample_id='fixture-1', prediction=0),
                                    dict(sample_id='fixture-3', prediction=None),
                                    dict(sample_id='fixture-4', prediction=1),
                                    dict(sample_id='fixture-5', prediction=1)], 'pred')
        summary, details = e.evaluate(**data, predictions=predictions, target='overall_risk', split='val')
        self.assertEqual(summary['metrics']['n_scored'], 2)
        self.assertEqual(summary['metrics']['accuracy'], 1)
        self.assertEqual(summary['prediction_coverage'], .5)
        self.assertEqual(summary['correct_fraction_of_all_eligible'], .5)
        self.assertEqual([summary[k] for k in ('n_unknown', 'n_unverified', 'n_eligible_missing', 'n_eligible_abstained')], [1, 1, 1, 1])
        self.assertEqual(summary['status'], 'partial_coverage')
        self.assertEqual(sum(bool(row['outcome']) for row in details), 2)

    def test_all_unknown_has_no_score(self):
        summary, _ = e.evaluate(**dataset(['', '']), predictions={}, target='overall_risk', split='val')
        self.assertEqual(summary['status'], 'no_valid_labels')
        self.assertIsNone(summary['metrics']['accuracy'])
        self.assertIsNone(summary['prediction_coverage'])

    def test_ids_and_group_isolation(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            e.index_rows([{'sample_id': 'x'}, {'sample_id': 'x'}], 'predictions')
        with self.assertRaisesRegex(ValueError, 'outside dataset'):
            e.evaluate(**dataset(['0']), predictions={'unexpected': {'prediction': 1}}, target='overall_risk', split='val')
        data = dataset(['0', '1'])
        data['inputs']['fixture-1']['image_group_id'] = 'I0'
        data['splits']['fixture-1']['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'cross-split'):
            e.evaluate(**data, predictions={}, target='overall_risk', split='val')

    def test_d_formats_aliases_and_null_not_rethresholded(self):
        rows = [{'sample_id': 'x', 'fusion_mode': 'full', 'fusion_pred': '', 'fusion_score': '.99'}]
        selected, _ = e.select_predictions(rows, 'D', 'overall_risk', 'full')
        self.assertIsNone(selected['x']['prediction'])
        compare = [{'sample_id': 'x', 'schemes': {'full': {'prediction': 1, 'prediction_target': 'overall_risk'},
                                                'image_only': {'prediction': 0, 'prediction_target': 'image_adversarial'}}}]
        selected, _ = e.select_predictions(compare, 'D', 'overall_risk', 'full')
        self.assertEqual(selected['x']['prediction'], 1)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            e.select_predictions(compare, 'D', 'overall_risk', 'image_only')
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            e.select_predictions([dict(sample_id='x', mode='full', prediction=1, fusion_pred=0)], 'D', 'overall_risk', 'full')

    def test_invalid_values_ocr_failure_and_unavailable(self):
        for value in ('nan', 'inf', -0.1, 1.1, True):
            with self.assertRaises(ValueError):
                e.score(value, 'score')
        for value in (True, 0.0, 2, 'safe'):
            with self.assertRaises(ValueError):
                e.binary(value, 'prediction')
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            e.select_predictions([dict(sample_id='x', image_pred=0, available=False)], 'B', 'image_adversarial')
        row = dict(sample_id='x', user_text_score=.7, ocr_text_score=None, ocr_status='failed')
        result, _ = e.select_predictions([row], 'C', 'harmful_content', score_field='user_text_score', threshold=.5)
        self.assertEqual(result['x']['prediction'], 1)
        result, _ = e.select_predictions([row], 'C', 'harmful_content', score_field='ocr_text_score', threshold=.5)
        self.assertIsNone(result['x']['prediction'])
        with self.assertRaisesRegex(ValueError, 'OCR'):
            e.select_predictions([{**row, 'ocr_text_score': .9}], 'C', 'harmful_content', score_field='ocr_text_score', threshold=.5)

    def test_score_only_requires_explicit_threshold_and_target(self):
        row = dict(sample_id='x', user_text_score=.5)
        with self.assertRaisesRegex(ValueError, 'threshold'):
            e.select_predictions([row], 'C', 'harmful_content', score_field='user_text_score')
        result, _ = e.select_predictions([row], 'C', 'harmful_content', score_field='user_text_score', threshold=.5)
        self.assertEqual(result['x']['prediction'], 1)
        with self.assertRaisesRegex(ValueError, 'image_adversarial'):
            e.select_predictions([], 'B', 'overall_risk')

    def test_freeze_must_match_selected_task_and_threshold(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'freeze.json'
            options = dict(run_id='fixed-C-v1', module='C', target='harmful_content', mode=None,
                           score_field='user_text_score', threshold=.5)
            config = dict(**options, frozen=True, model_version='fixture-version', frozen_at='2026-11-07', test_used_for_selection=False)
            path.write_text(json.dumps(config), encoding='utf-8')
            self.assertEqual(e.check_frozen_config(path, argparse.Namespace(**options)), config)
            with self.assertRaisesRegex(ValueError, 'threshold'):
                e.check_frozen_config(path, argparse.Namespace(**{**options, 'threshold': .9}))

    def test_cli_synthetic_outputs_and_test_freeze_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = dataset(['1', '0'])
            for key, filename in (('inputs', 'inputs.csv'), ('labels', 'master_labels.csv'), ('splits', 'splits.csv')):
                rows = list(data[key].values())
                e.write_csv(root / filename, list(rows[0]), rows)
            preds = root / 'predictions.csv'
            e.write_csv(preds, ['sample_id', 'text_pred'], [dict(sample_id='fixture-1', text_pred=1), dict(sample_id='fixture-0', text_pred=0)])
            args = ['--work-dir', str(root), '--predictions', str(preds), '--module', 'C', '--target', 'overall_risk',
                    '--run-id', 'synthetic-test-only', '--data-kind', 'synthetic', '--out', str(root / 'report')]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(e.main(args), 0)
            report = json.loads((root / 'report/summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['use'], 'software_verification_only')
            self.assertEqual(report['metrics']['fp'], 1)
            self.assertEqual(report['metrics']['fn'], 1)
            self.assertEqual(len(e.read_rows(root / 'report/failure_cases_PRIVATE.csv')), 2)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                e.main(args + ['--split', 'test'])
            self.assertEqual(failure.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
