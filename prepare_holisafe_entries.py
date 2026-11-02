"""Import the downloaded HoliSafe metadata into the A-side canonical tables.

This is a deterministic source adapter. It does not infer jailbreak success or
image adversarial labels, and refuses to overwrite a non-empty project table.
"""
import json
from pathlib import Path
import shutil

import week3 as w


def prepare(root=w.ROOT):
    source_root = root/'public_sources/holisafe'
    metadata = source_root/'holisafe_bench.json'
    if not metadata.is_file():
        raise ValueError('HoliSafe metadata is missing')
    existing = w.csv_read(root/'entries.csv')
    if existing:
        raise ValueError('entries.csv is not empty; refusing to overwrite existing work')
    rows = json.loads(metadata.read_text(encoding='utf-8'))
    image_audit = {r['path']: r for r in w.csv_read(root/'output/public_audit/images.csv')
                   if r['source'] == 'holisafe'}
    entries, source_rows, strata_rows = [], [], []
    seen = {}
    for row in rows:
        original = (source_root/'images'/row['image']).resolve()
        key = original.relative_to(root).as_posix()
        audit = image_audit.get(key)
        if audit is None or audit['error']:
            raise ValueError(f'cannot import invalid image: {key}')
        digest = audit['sha256']
        dest = root/'images'/f'{digest[:20]}{original.suffix.lower()}'
        if not dest.exists():
            shutil.copy2(original, dest)
        image_group = 'HOLI_PIXEL_' + audit['pixel_sha256'][:16]
        request_group = 'HOLI_QUERY_' + w.sha_bytes(
            w.normalized_text(row['query']).encode('utf-8'))[:16]
        sample_key = (digest, row['query'].strip(), 'SRC_HOLISAFE')
        sample_id = 'S' + w.sha_bytes((digest + '\0' + row['query'].strip() +
                                        '\0SRC_HOLISAFE').encode('utf-8'))[:12].upper()
        duplicate_of = seen.get(sample_key)
        if duplicate_of is not None:
            source_rows.append({
                'sample_id': duplicate_of[1], 'sample_id_source': str(row['id']), 'image': row['image'],
                'type': row['type'], 'category': row['category'],
                'subcategory': row['subcategory'], 'image_safe': str(row['image_safe']).lower(),
                'image_safety_label': str(row['image_safety_label']),
                'image_sha256': digest, 'pixel_sha256': audit['pixel_sha256'],
                'image_group_id': image_group, 'request_group_id': request_group,
                'source_id': 'SRC_HOLISAFE', 'dedup_status': 'duplicate_exact_content_text',
                'canonical_source_id': duplicate_of[0],
            })
            continue
        seen[sample_key] = (str(row['id']), sample_id)
        out = dict.fromkeys(w.ENTRY_COLS, '')
        out.update(
            image_file=str(original), user_text=row['query'].strip(),
            source_id='SRC_HOLISAFE', image_group_id=image_group,
            request_group_id=request_group,
            # HoliSafe's third type character is the official final-input
            # safety label. It is not a model prediction or attack-success tag.
            harmful_content_label='1' if row['type'][1] == 'U' else '0',
            overall_risk_label='1' if row['type'][2] == 'U' else '0',
            label_status='verified',
            notes=('source_official; HoliSafe type=%s; image_safe=%s; '
                   'image_safety_label=%s; image_adv_label_unknown; '
                   'jailbreak_intent_unknown; prompt_injection_unknown') %
                  (row['type'], str(row['image_safe']).lower(), row['image_safety_label']))
        entries.append(out)
        strata_rows.append({'sample_id': sample_id, 'source_type': row['type'],
                            'category': row['category']})
        source_rows.append({
            'sample_id': sample_id, 'sample_id_source': str(row['id']), 'image': row['image'],
            'type': row['type'], 'category': row['category'],
            'subcategory': row['subcategory'], 'image_safe': str(row['image_safe']).lower(),
            'image_safety_label': str(row['image_safety_label']),
            'image_sha256': digest, 'pixel_sha256': audit['pixel_sha256'],
            'image_group_id': image_group, 'request_group_id': request_group,
            'source_id': 'SRC_HOLISAFE', 'dedup_status': 'canonical',
            'canonical_source_id': str(row['id']),
        })
    w.csv_write(root/'entries.csv', entries, w.ENTRY_COLS)
    w.csv_write(root/'work/holisafe_source_records.csv', source_rows,
                list(source_rows[0]))
    w.csv_write(root/'work/split_strata.csv', strata_rows,
                ['sample_id','source_type','category'])
    print(f'prepared {len(entries)} unique inputs from {len(source_rows)} source records and {len({r["image_sha256"] for r in source_rows})} image-byte hashes')


if __name__ == '__main__':
    prepare()
