"""A's binary metrics, implemented with the standard library.

Joins by sample_id, keeps unknown truth and missing predictions out of the
confusion matrix, and reports their counts. Never fits a threshold or model.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


TARGETS = {
    'overall_risk': 'overall_risk_label',
    'harmful_content': 'harmful_content_label',
    'image_adversarial': 'image_adv_label',
    'jailbreak_intent': 'jailbreak_intent_label',
    'prompt_injection': 'prompt_injection_label',
}
SPLITS = ('base_train', 'fusion_train', 'val', 'test')
MODES = ('full', 'text_only', 'image_only', 'user_text_only', 'user_text_ocr')
PREDICTION_FIELDS = {'B': ('image_pred',), 'C': ('text_pred',), 'D': ('prediction', 'fusion_pred')}
SCORE_FIELDS = ('image_score', 'visual_score', 'user_text_score', 'ocr_text_score',
                'combined_text_score', 'risk_score', 'fusion_score')


def read_rows(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        if path.suffix.lower() == '.csv':
            reader = csv.DictReader(handle)
            if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f'{path}: missing or repeated CSV headers')
            rows = list(reader)
            if any(None in row or None in row.values() for row in rows):
                raise ValueError(f'{path}: malformed CSV row')
            return rows
        if path.suffix.lower() != '.jsonl':
            raise ValueError('Only CSV and JSONL prediction files are supported')
        rows = [json.loads(line) for line in handle if line.strip()]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError(f'{path}: every JSONL line must be an object')
        return rows


def index_rows(rows, name):
    indexed = {}
    for row in rows:
        sid = row.get('sample_id')
        if not isinstance(sid, str) or not sid.strip() or sid != sid.strip():
            raise ValueError(f'{name}: sample_id must be a nonempty, unpadded string')
        if sid in indexed:
            raise ValueError(f'{name}: duplicate sample_id {sid}')
        indexed[sid] = row
    return indexed


def binary(value, field):
    if value is None or value == '' or value == 'null':
        return None
    if type(value) is int and value in (0, 1):
        return value
    if type(value) is str and value in ('0', '1'):
        return int(value)
    raise ValueError(f'{field}: expected 0, 1, or missing; got {value!r}')


def score(value, field):
    if value is None or value == '' or value == 'null':
        return None
    if isinstance(value, bool):
        raise ValueError(f'{field}: boolean is not a score')
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f'{field}: score must be finite and in [0,1]')
    return number


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def binary_metrics(pairs):
    """Positive class is 1; undefined ratios stay None, including empty inputs."""
    counts = Counter((truth, prediction) for truth, prediction in pairs)
    if set(counts) - {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError('metrics only accept binary truth/prediction pairs')
    tn, fp, fn, tp = (counts[(0, 0)], counts[(0, 1)], counts[(1, 0)], counts[(1, 1)])
    recall, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    return {
        'n_scored': tp + tn + fp + fn,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': divide(tp + tn, tp + tn + fp + fn),
        'precision': divide(tp, tp + fp), 'recall': recall,
        'f1': divide(2 * tp, 2 * tp + fp + fn),
        'fpr': divide(fp, fp + tn), 'fnr': divide(fn, fn + tp),
        'specificity': specificity,
        'balanced_accuracy': (recall + specificity) / 2 if recall is not None and specificity is not None else None,
    }


def alias(row, fields):
    found = [field for field in fields if field in row]
    if not found:
        raise ValueError(f"{row.get('sample_id')}: missing field; expected one of {fields}")
    values = [binary(row[field], field) for field in found]
    if len(set(values)) != 1:
        raise ValueError(f"{row.get('sample_id')}: conflicting prediction aliases")
    return values[0]


def select_predictions(rows, module, target, mode=None, prediction_field=None, score_field=None, threshold=None):
    if module == 'D' and mode not in MODES:
        raise ValueError('D evaluation requires an explicit supported --mode')
    if module == 'B' and target != 'image_adversarial':
        raise ValueError('B image predictions must be compared with image_adversarial truth')
    if module == 'D' and target != ('image_adversarial' if mode == 'image_only' else 'overall_risk'):
        raise ValueError('D mode and evaluation target disagree')
    if module == 'C' and target == 'image_adversarial':
        raise ValueError('C text predictions cannot score image-adversarial detection')
    if score_field and (threshold is None or score(threshold, 'threshold') is None):
        raise ValueError('score-only evaluation requires an explicit fixed --threshold')
    if threshold is not None and not score_field:
        raise ValueError('--threshold is only used with --score-field')
    if module == 'D' and score_field:
        raise ValueError('Use D submitted decisions; do not re-threshold D/CIDER results in A evaluation')
    selected, skipped = [], 0
    for original in rows:
        row = dict(original)
        sid = row.get('sample_id')
        if not isinstance(sid, str) or not sid.strip() or sid != sid.strip():
            raise ValueError('Every prediction, including unselected modes, needs a valid sample_id')
        if module == 'D':
            if 'schemes' in row:
                schemes = row['schemes']
                if not isinstance(schemes, dict):
                    raise ValueError('D schemes must be an object')
                if mode not in schemes:
                    skipped += 1
                    continue
                if not isinstance(schemes[mode], dict):
                    raise ValueError('D selected scheme must be an object')
                row = {**schemes[mode], 'sample_id': row.get('sample_id'), 'mode': mode}
            row_mode = row.get('mode', row.get('fusion_mode'))
            if row_mode not in MODES:
                raise ValueError('D prediction lacks a supported mode/fusion_mode')
            if 'mode' in row and 'fusion_mode' in row and row['mode'] != row['fusion_mode']:
                raise ValueError('conflicting mode aliases')
            if row_mode != mode:
                skipped += 1
                continue
        declared_target = row.get('prediction_target')
        if declared_target and declared_target != target:
            raise ValueError(f"{row.get('sample_id')}: prediction_target conflicts with --target")
        for field in SCORE_FIELDS:
            if field in row:
                score(row[field], field)
        if 'ocr_status' in row and row['ocr_status'] not in ('success', 'no_text', 'failed', 'unavailable', ''):
            raise ValueError('invalid ocr_status')
        available = row.get('available', True)
        if available in ('true', 'false'):
            available = available == 'true'
        if type(available) is not bool:
            raise ValueError('available must be a JSON boolean (or CSV true/false)')
        if score_field:
            if score_field not in row:
                raise ValueError(f'missing score field {score_field}')
            value = score(row[score_field], score_field)
            prediction = int(value >= threshold) if value is not None else None
        else:
            prediction = alias(row, (prediction_field,) if prediction_field else PREDICTION_FIELDS[module])
        if not available and (prediction is not None or any(score(row.get(field), field) is not None for field in SCORE_FIELDS)):
            raise ValueError('unavailable module carries a usable prediction or score')
        if module == 'C' and score_field in ('ocr_text_score', 'combined_text_score') and row.get('ocr_status') in ('failed', 'unavailable', 'no_text') and prediction is not None:
            raise ValueError('OCR/combined prediction cannot use failed, unavailable or absent OCR text')
        selected.append({'sample_id': row.get('sample_id'), 'prediction': prediction})
    return index_rows(selected, 'selected predictions'), skipped


def evaluate(inputs, labels, splits, predictions, target, split):
    if target not in TARGETS or split not in SPLITS:
        raise ValueError('unsupported target or split')
    if not inputs or set(inputs) != set(labels) or set(inputs) != set(splits):
        raise ValueError('inputs/labels/splits must have the same nonempty ID set')
    owners = {}
    for sid, row in inputs.items():
        part = splits[sid].get('split')
        if part not in SPLITS:
            raise ValueError(f'{sid}: invalid split')
        for field in ('image_group_id', 'request_group_id'):
            group = row.get(field)
            if not group:
                raise ValueError(f'{sid}: missing {field}')
            key = (field, group)
            if key in owners and owners[key] != part:
                raise ValueError(f'cross-split group leakage: {field}')
            owners[key] = part
    extra = set(predictions) - set(inputs)
    if extra:
        raise ValueError(f'prediction IDs outside dataset: {len(extra)}, e.g. {sorted(extra)[:3]}')
    details, pairs = [], []
    for sid in sorted(inputs):
        if splits[sid]['split'] != split:
            continue
        label = labels[sid]
        if TARGETS[target] not in label:
            raise ValueError(f'{sid}: missing required truth column {TARGETS[target]}')
        truth = binary(label.get(TARGETS[target]), TARGETS[target])
        if label.get('label_status') not in ('verified', 'review', 'pending'):
            raise ValueError(f'{sid}: invalid label_status')
        prediction = binary(predictions.get(sid, {}).get('prediction'), 'prediction')
        truth_state = 'unverified' if label['label_status'] != 'verified' else 'unknown' if truth is None else 'eligible'
        prediction_state = 'missing_row' if sid not in predictions else 'abstained' if prediction is None else 'available'
        outcome = ''
        if truth_state == 'eligible' and prediction_state == 'available':
            pairs.append((truth, prediction))
            outcome = {(1, 1): 'TP', (1, 0): 'FN', (0, 1): 'FP', (0, 0): 'TN'}[(truth, prediction)]
        details.append(dict(sample_id=sid, split=split, target=target, truth=truth,
                            prediction=prediction, truth_state=truth_state,
                            prediction_state=prediction_state, outcome=outcome))
    if not details:
        raise ValueError(f'no samples in selected split {split}')
    eligible = [row for row in details if row['truth_state'] == 'eligible']
    positives = sum(row['truth'] == 1 for row in eligible)
    metrics = binary_metrics(pairs)
    summary = dict(target=target, label_field=TARGETS[target], split=split,
                   n_split=len(details), n_eligible=len(eligible),
                   n_unknown=sum(row['truth_state'] == 'unknown' for row in details),
                   n_unverified=sum(row['truth_state'] == 'unverified' for row in details),
                   eligible_positive=positives, eligible_negative=len(eligible) - positives,
                   n_missing_rows=sum(row['prediction_state'] == 'missing_row' for row in details),
                   n_abstained=sum(row['prediction_state'] == 'abstained' for row in details),
                   n_eligible_missing=sum(row['prediction_state'] == 'missing_row' for row in eligible),
                   n_eligible_abstained=sum(row['prediction_state'] == 'abstained' for row in eligible),
                   prediction_rows_other_splits=len(set(predictions) - {row['sample_id'] for row in details}),
                   label_coverage=divide(len(eligible), len(details)),
                   prediction_coverage=divide(len(pairs), len(eligible)),
                   positive_prediction_coverage=divide(sum(y == 1 for y, _ in pairs), positives),
                   negative_prediction_coverage=divide(sum(y == 0 for y, _ in pairs), len(eligible) - positives),
                   correct_fraction_of_all_eligible=divide(metrics['tp'] + metrics['tn'], len(eligible)),
                   majority_baseline_accuracy=divide(max(positives, len(eligible) - positives), len(eligible)),
                   status='no_valid_labels' if not eligible else 'no_predictions' if not pairs else 'partial_coverage' if len(pairs) < len(eligible) else 'complete',
                   metrics=metrics)
    return summary, details


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def check_frozen_config(path, args):
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    expected = dict(run_id=args.run_id, module=args.module, target=args.target,
                    mode=args.mode, score_field=args.score_field, threshold=args.threshold)
    if not isinstance(config, dict) or config.get('frozen') is not True:
        raise ValueError('frozen config must be a JSON object with frozen: true')
    if not config.get('model_version') or not config.get('frozen_at') or config.get('test_used_for_selection') is not False:
        raise ValueError('frozen config requires model_version, frozen_at and test_used_for_selection: false')
    for key, value in expected.items():
        if key not in config or config[key] != value:
            raise ValueError(f'frozen config {key} differs from evaluation arguments')
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=Path(__file__).resolve().parent / 'work')
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--module', choices=('B', 'C', 'D'), required=True)
    parser.add_argument('--target', choices=TARGETS, required=True)
    parser.add_argument('--split', choices=(*SPLITS, 'validation'), default='val')
    parser.add_argument('--mode', choices=MODES)
    fields = parser.add_mutually_exclusive_group()
    fields.add_argument('--prediction-field')
    fields.add_argument('--score-field', choices=SCORE_FIELDS)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--run-id', required=True, help='module/config version, fixed before test evaluation')
    parser.add_argument('--frozen-config', type=Path, help='required for test: records fixed model/version/threshold')
    parser.add_argument('--data-kind', choices=('real', 'synthetic'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        part = 'val' if args.split == 'validation' else args.split
        if not args.run_id.strip():
            raise ValueError('--run-id cannot be blank')
        if part == 'test' and not args.frozen_config:
            raise ValueError('test evaluation requires --frozen-config; no threshold selection on test')
        if args.frozen_config:
            check_frozen_config(args.frozen_config, args)
        if args.out.exists():
            raise ValueError('output directory already exists; choose a new run directory')
        paths = {name: args.work_dir / filename for name, filename in
                 (('inputs', 'inputs.csv'), ('labels', 'master_labels.csv'), ('splits', 'splits.csv'))}
        tables = {name: index_rows(read_rows(path), name) for name, path in paths.items()}
        raw = read_rows(args.predictions)
        predictions, skipped = select_predictions(raw, args.module, args.target, args.mode,
                                                  args.prediction_field, args.score_field, args.threshold)
        summary, details = evaluate(**tables, predictions=predictions, target=args.target, split=part)
        paths['predictions'] = args.predictions
        if args.frozen_config:
            paths['frozen_config'] = args.frozen_config
        summary.update(run_id=args.run_id, module=args.module, mode=args.mode,
                       data_kind=args.data_kind, use='software_verification_only' if args.data_kind == 'synthetic' else 'submitted_prediction_evaluation',
                       split_role='training_diagnostic_only' if part in ('base_train', 'fusion_train') else 'validation' if part == 'val' else 'frozen_test',
                       score_field=args.score_field, prediction_field=args.prediction_field,
                       fixed_threshold=args.threshold, rows_other_modes=skipped,
                       input_sha256={key: file_hash(path) for key, path in paths.items()},
                       evaluator_sha256=file_hash(__file__))
        args.out.mkdir(parents=True)
        (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        write_csv(args.out / 'sample_results_PRIVATE.csv', list(details[0]), details)
        failures = [row for row in details if row['outcome'] in ('FP', 'FN')]
        write_csv(args.out / 'failure_cases_PRIVATE.csv', list(details[0]), failures)
        matrix = summary['metrics']
        write_csv(args.out / 'confusion_matrix.csv', ['truth', 'predicted_0', 'predicted_1'], [
            dict(truth=0, predicted_0=matrix['tn'], predicted_1=matrix['fp']),
            dict(truth=1, predicted_0=matrix['fn'], predicted_1=matrix['tp'])])
        coverage = {key: summary[key] for key in ('data_kind', 'status', 'n_split', 'n_eligible', 'n_unknown', 'n_unverified',
                    'n_eligible_missing', 'n_eligible_abstained', 'label_coverage', 'prediction_coverage',
                    'positive_prediction_coverage', 'negative_prediction_coverage', 'majority_baseline_accuracy')}
        write_csv(args.out / 'metrics.csv', ['run_id', 'target', 'split', *coverage, *matrix],
                  [dict(run_id=args.run_id, target=args.target, split=part, **coverage, **matrix)])
        (args.out / 'README.md').write_text(
            f"# 评测记录\n\n类型：{args.data_kind}；状态：{summary['status']}。\n\n"
            'synthetic 仅验证软件，不能作为真实成绩。指标基于有已核实标签且有预测的样本；'
            '必须同时报告 summary.json 中的标签覆盖率、预测覆盖率、缺失和拒判数量。'
            '没有有效标签或预测时不报告成绩，零分母指标为 null。\n\n'
            'PRIVATE 表含真值及失败样本编号，A 本地保管；测试集不能用于调参。'
            '冻结配置文件的哈希只记录评测声明，不证明成员训练时确实遵守隔离。\n', encoding='utf-8')
        print(json.dumps({'output': str(args.out), 'status': summary['status'], 'data_kind': args.data_kind,
                          'n_scored': matrix['n_scored']}, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
