"""Read-only full-label audit for remaining, unexposed source groups."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sqlite3
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]


def main():
    started = time.time()
    planner = ROOT / 'experiments/p2_planning_sft_upright_3b_gpu7_20261105/data/index.sqlite'
    with sqlite3.connect(planner.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        planner_groups = {r[0] for r in conn.execute('SELECT DISTINCT group_id FROM samples')}
    source = ROOT / 'experiments/p2_ltx_event_current_sources_v2_20261106/sources.sqlite'
    with sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        train_groups = {r[0] for r in conn.execute("SELECT DISTINCT json_extract(value,'$.source.group_id') FROM samples")}
    tuning = json.loads((ROOT / 'experiments/p2_loss_search28_evaluation_20261110/tuning200.json').read_text())
    tuning_groups = {r['group_id'] for r in tuning}
    previous = json.loads((WORK / 'CANDIDATE_AUDIT.json').read_text())
    generated_ids = set()
    folders = previous['scanned_directories']
    for folder in folders:
        path = Path(folder)
        if not path.is_dir():
            continue
        for item in path.iterdir():
            if item.is_dir() and any((item / f).is_file() for f in
                                    ('metadata.json', 'complete.json', 'generated.mp4', 'payload.pt')):
                generated_ids.add(item.name)
    manifest = ROOT / 'experiments/listen_j_labels_manifest_20261105/labels.jsonl'
    before = manifest.stat()
    digest = hashlib.sha256()
    splits = Counter()
    group_counts = Counter()
    output_groups = set()
    output_ids_found = set()
    with manifest.open('rb') as stream:
        for i, raw in enumerate(stream, 1):
            digest.update(raw)
            row = json.loads(raw)
            splits[row['split']] += 1
            group_counts[row['split'], row['group_id']] += 1
            if row['sample_id'] in generated_ids:
                output_groups.add(row['group_id'])
                output_ids_found.add(row['sample_id'])
            if i % 100000 == 0:
                print(json.dumps(dict(rows=i, seconds=time.time()-started)), flush=True)
    after = manifest.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    excluded = planner_groups | train_groups | tuning_groups
    prehistory = {g: n for (split, g), n in group_counts.items() if split == 'test' and g not in excluded}
    available = {g:n for g,n in prehistory.items() if g not in output_groups}
    report = dict(success=True, at=time.time(), rows=sum(splits.values()), splits=dict(splits),
                  manifest=str(manifest), manifest_sha256=digest.hexdigest(),
                  planner_groups=len(planner_groups), generation_train_groups=len(train_groups),
                  tuning_groups=len(tuning_groups), historical_output_folders=len(folders),
                  historical_output_ids=len(generated_ids), output_ids_found_in_current_labels=len(output_ids_found),
                  test_groups_outside_training_and_tuning=prehistory,
                  test_groups_without_known_historical_outputs=available,
                  scope='Known local output layouts recorded in CANDIDATE_AUDIT.json; deleted or external outputs are not covered',
                  no_scores_used=True, no_gpu=True, final_test_ready=False,
                  elapsed_seconds=time.time()-started)
    dest = WORK / 'FULL_POOL_AUDIT.json'
    temporary = dest.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(dest)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
