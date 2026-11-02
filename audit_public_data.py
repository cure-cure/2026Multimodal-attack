"""Audit already-downloaded public data offline; never download or alter originals."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image
import week3 as w


def image_record(path, root):
    row = dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size,
               sha256=w.sha_file(path), pixel_sha256='', width='', height='',
               dhash='', error='')
    try:
        with Image.open(path) as im:
            im.load()
            rgb = im.convert('RGB')
            row.update(width=rgb.width, height=rgb.height,
                       pixel_sha256=w.sha_bytes(str(rgb.size).encode() + rgb.tobytes()))
            gray = rgb.convert('L')
            pixels = list(gray.resize((9, 8)).get_flattened_data())
            bits = [pixels[y*9+x] > pixels[y*9+x+1] for y in range(8) for x in range(8)]
            row['dhash'] = f'{sum(int(b) << i for i, b in enumerate(bits)):016x}'
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        row['error'] = type(exc).__name__ + ': ' + str(exc)
    return row


def audit(root, out):
    out.mkdir(parents=True, exist_ok=True)
    images, missing, versions = [], [], []
    reports = {}
    for name in ('holisafe', 'jailbreakv'):
        base = root/'public_sources'/name
        local_images = sorted(p for p in base.rglob('*')
                              if p.suffix.lower() in w.IMAGE_EXT and '.cache' not in p.parts)
        records = [dict(source=name, **image_record(p, root)) for p in local_images]
        images.extend(records)
        image_map = {r['path']: r for r in records}
        metadata = list((base/'.cache/huggingface/download').rglob('*.metadata'))
        commits = Counter()
        hash_errors = []
        for meta in metadata:
            fields = meta.read_text().splitlines()
            commits[fields[0]] += 1
            relative = meta.relative_to(base/'.cache/huggingface/download').as_posix()[:-9]
            file = base/relative
            if not file.is_file():
                hash_errors.append(relative + ': metadata_without_file')
                continue
            digest = w.sha_file(file)
            etag = fields[1]
            if len(etag) == 64:
                valid = digest == etag
            else:
                content = file.read_bytes()
                valid = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest() == etag
            if not valid:
                hash_errors.append(relative + ': etag_mismatch')
            versions.append(dict(source=name, file=file.relative_to(root).as_posix(),
                                 commit=fields[0], sha256=digest, metadata_match=valid))
        if name == 'holisafe':
            rows = json.loads((base/'holisafe_bench.json').read_text(encoding='utf-8'))
            image_base, field = base/'images', 'image'
            bad_labels = [r['id'] for r in rows if r['type'] not in {'SSS','SSU','SUU','USU','UUU'}
                          or r['image_safe'] != (r['type'][0] == 'S')
                          or (r['image_safety_label'] == 0) != r['image_safe']
                          or not 0 <= r['image_safety_label'] <= 19]
            extra = dict(label_conflicts=bad_labels, types=dict(Counter(r['type'] for r in rows)),
                         categories=dict(Counter(r['category'] for r in rows)))
            labels_by_image = defaultdict(set)
            pairs = Counter()
            for row in rows:
                labels_by_image[row['image']].add((row['image_safe'], row['image_safety_label']))
                image_info = image_map.get((image_base/row['image']).relative_to(root).as_posix())
                if image_info:
                    pairs[(image_info['sha256'], w.normalized_text(row['query']))] += 1
            conflicting_images = {image for image, labels in labels_by_image.items() if len(labels) > 1}
            extra.update(image_safety_conflicting_paths=len(conflicting_images),
                         source_rows_on_conflicting_images=sum(
                             row['image'] in conflicting_images for row in rows),
                         duplicate_exact_image_query_records=sum(count-1 for count in pairs.values() if count > 1))
            text_field = 'query'
        else:
            image_base, field = base/'JailBreakV_28K', 'image_path'
            rows = w.csv_read(image_base/'JailBreakV_28K.csv')
            mini = w.csv_read(image_base/'mini_JailBreakV_28K.csv')
            by_id = {r['id']: r for r in rows}
            full_pairs = {(r['image_path'], r['jailbreak_query']) for r in rows}
            extra = dict(mini_rows=len(mini),
                         mini_not_exact_subset=sum((r['image_path'], r['jailbreak_query']) not in full_pairs for r in mini),
                         mini_missing_images=sum(not (image_base/r[field]).is_file() for r in mini),
                         redteam_rows=len(w.csv_read(image_base/'RedTeam_2K.csv')),
                         formats=dict(Counter(r['format'] for r in rows)))
            text_field = 'jailbreak_query'
        usable = 0
        paths = set()
        for row in rows:
            path = (image_base/row[field]).resolve()
            if not path.is_relative_to(image_base.resolve()):
                raise ValueError('Source image path escapes its directory')
            relative = path.relative_to(root).as_posix()
            paths.add(relative)
            info = image_map.get(relative)
            reason = 'image_missing' if info is None else ('image_decode_failed' if info['error'] else '')
            if not row[text_field].strip():
                reason = 'empty_text'
            if reason:
                missing.append(dict(source=name, original_id=row['id'], image_path=relative, issue=reason))
            else:
                usable += 1
        reports[name] = dict(rows=len(rows), unique_ids=len({r['id'] for r in rows}),
                             referenced_images=len(paths), downloaded_images=len(records),
                             usable_pairs=usable, missing_or_invalid_pairs=len(rows)-usable,
                             missing_distinct_images=len(paths-set(image_map)),
                             unreferenced_images=len(set(image_map)-paths),
                             decoded_images=sum(not r['error'] for r in records),
                             unique_image_bytes=len({r['sha256'] for r in records}),
                             unique_image_pixels=len({r['pixel_sha256'] for r in records if not r['error']}),
                             commits=dict(commits), metadata_hash_errors=hash_errors,
                             incomplete_files=[str(p.relative_to(root)) for p in base.rglob('*.incomplete')],
                             **extra)
    w.csv_write(out/'images.csv', images, list(images[0]))
    w.csv_write(out/'missing_pairs.csv', missing, ['source','original_id','image_path','issue'])
    w.csv_write(out/'versions.csv', versions, ['source','file','commit','sha256','metadata_match'])
    inputs_file, labels_file, splits_file = (root/'work'/name for name in
                                             ('inputs.csv','master_labels.csv','splits.csv'))
    strata_file = root/'work'/'split_strata.csv'
    if all(p.is_file() for p in (inputs_file, labels_file, splits_file, strata_file)):
        inputs = w.csv_read(inputs_file)
        labels = {r['sample_id']: r for r in w.csv_read(labels_file)}
        splits = {r['sample_id']: r['split'] for r in w.csv_read(splits_file)}
        strata = {r['sample_id']: r for r in w.csv_read(strata_file)}
        components = w.connected_components(inputs)
        groups_by_split = Counter(next(iter({splits[sid] for sid in group}))
                                  for group in components if len({splits[sid] for sid in group}) == 1)
        types = sorted({r['source_type'] for r in strata.values()})
        categories = sorted({r['category'] for r in strata.values()})
        stats = []
        for part in w.SPLITS:
            ids = [sid for sid, split in splits.items() if split == part]
            by_type = Counter(strata[sid]['source_type'] for sid in ids if sid in strata)
            by_category = Counter(strata[sid]['category'] for sid in ids if sid in strata)
            row = dict(split=part, samples=len(ids), groups=groups_by_split[part],
                       overall_safe=sum(labels[sid]['overall_risk_label'] == '0' for sid in ids),
                       overall_unsafe=sum(labels[sid]['overall_risk_label'] == '1' for sid in ids))
            row.update({'type_'+key: by_type[key] for key in types})
            row.update({'category_'+key: by_category[key] for key in categories})
            stats.append(row)
        w.csv_write(out/'split_statistics.csv', stats, list(stats[0]))
    (out/'summary.json').write_text(json.dumps(reports, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='output/public_audit')
    args = parser.parse_args()
    audit(w.ROOT, w.ROOT/args.out)
