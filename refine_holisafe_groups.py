"""Conservatively co-allocate similar inputs, without changing any risk labels."""
from datetime import datetime
from itertools import combinations
import json
import shutil

import week3 as w


def group_keys(items, edges):
    dsu = w.DSU(items)
    for a, b in edges:
        dsu.union(a, b)
    groups = {}
    for key in sorted(items):
        groups.setdefault(dsu.find(key), []).append(key)
    return {key: min(group) for group in groups.values() for key in group}


def refine(root=w.ROOT):
    entries = w.csv_read(root/'entries.csv')
    provenance = w.csv_read(root/'work/holisafe_source_records.csv')
    if not entries or any(r['source_id'] != 'SRC_HOLISAFE' for r in entries):
        raise ValueError('This adapter only supports the current HoliSafe-only corpus')
    images = {r['pixel_sha256']: r for r in w.csv_read(root/'output/public_audit/images.csv')
              if r['source'] == 'holisafe' and not r['error']}
    image_edges = []
    for (ka, a), (kb, b) in combinations(sorted(images.items()), 2):
        distance = (int(a['dhash'], 16) ^ int(b['dhash'], 16)).bit_count()
        if distance <= 4:
            image_edges.append(dict(a=ka, b=kb, distance=distance))
    image_groups = group_keys(images, [(r['a'], r['b']) for r in image_edges])
    queries = {w.sha_bytes(w.normalized_text(r['user_text']).encode()):
               set(w.normalized_text(r['user_text']).split()) for r in entries}
    text_edges = []
    for (ka, a), (kb, b) in combinations(sorted(queries.items()), 2):
        if min(len(a), len(b)) < 6 or min(len(a), len(b)) / max(len(a), len(b)) < .85:
            continue
        similarity = len(a & b) / len(a | b)
        if similarity >= .85:
            text_edges.append(dict(a=ka, b=kb, jaccard=similarity))
    text_groups = group_keys(queries, [(r['a'], r['b']) for r in text_edges])
    pixel_by_group = {r['image_group_id']: r['pixel_sha256'] for r in provenance}
    for row in entries:
        pixel = pixel_by_group[row['image_group_id']]
        query = w.sha_bytes(w.normalized_text(row['user_text']).encode())
        row['image_group_id'] = 'HOLI_VISUAL_' + image_groups[pixel][:16]
        row['request_group_id'] = 'HOLI_REQUEST_' + text_groups[query][:16]
    by_sid = {}
    for row in entries:
        digest = w.sha_file(w.safe_path(root, row['image_file']))
        sid = 'S' + w.sha_bytes((digest+'\0'+row['user_text']+'\0SRC_HOLISAFE').encode())[:12].upper()
        by_sid[sid] = row
    for row in provenance:
        for key in ('image_group_id', 'request_group_id'):
            row[key] = by_sid[row['sample_id']][key]
    backup = root/'backups'/('before_group_refinement_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    for relative in ('entries.csv','work/holisafe_source_records.csv','work/splits.csv'):
        source = root/relative
        if source.exists():
            target = backup/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    w.csv_write(root/'entries.csv', entries, w.ENTRY_COLS)
    w.csv_write(root/'work/holisafe_source_records.csv', provenance, list(provenance[0]))
    w.csv_write(root/'work/near_image_groups.csv', image_edges, ['a','b','distance'])
    w.csv_write(root/'work/near_query_groups.csv', text_edges, ['a','b','jaccard'])
    rules = dict(image_rule='RGB pixel hash exact; dHash-64 Hamming <=4 co-allocation',
                 query_rule='NFKC casefold whitespace exact; token-set Jaccard >=0.85, >=6 distinct tokens',
                 image_edges=len(image_edges), query_edges=len(text_edges),
                 note='Conservative co-allocation, not proof of semantic equivalence or manual review; no labels changed.')
    (root/'work/grouping_rules.json').write_text(json.dumps(rules, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(rules))


if __name__ == '__main__':
    refine()
