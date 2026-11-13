"""Remove leading Assistant labels, rejudge affected test rows, and retain all decisions."""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time

import filter_full_response_sft as base

PREVIOUS = base.ROOT / 'experiments/listen_j_test_response_filter_20261102'
DEFAULT = base.ROOT / 'experiments/listen_j_test_response_prefixfix_20261102'


def decisions(work):
    result = {}
    for meta in sorted((work / 'batches/test').glob('*.meta.json')):
        info = base.read(meta)
        path = work / info['result_file']
        assert base.sha(path) == info['result_sha256']
        for line in path.open():
            row = json.loads(line)
            assert row['sample_id'] not in result
            result[row['sample_id']] = row
    return result


def clean_row(row):
    text = row['target']['text']
    match = re.match(r'^\s*Assistant\s*:\s*', text, re.I)
    if not match:
        return None
    assert text[match.end():].strip()
    result = copy.deepcopy(row)
    result['target'].update(format='original_caption_strip_chat_prefix_v1',
        original_text=text, removed_prefix=match.group(), text=text[match.end():])
    return result


def prepare(work):
    old_cfg = base.read(PREVIOUS / 'config.json')
    complete = base.read(PREVIOUS / 'COMPLETE.json')
    assert complete['summary_sha256'] == base.sha(PREVIOUS / 'filtered/summary.json')
    source = Path(old_cfg['sources']['test']['path'])
    assert base.sha(source) == old_cfg['sources']['test']['sha256']
    old = decisions(PREVIOUS)
    work.mkdir(exist_ok=False)
    candidates = work / 'candidates'
    candidates.mkdir()
    edits = []
    with (candidates / 'test.jsonl').open('w') as out:
        for raw in source.open('rb'):
            row = json.loads(raw)
            cleaned = clean_row(row)
            if cleaned is None:
                continue
            before = old[row['sample_id']]
            assert before['decision'] == 'reject'
            out.write(json.dumps(cleaned, ensure_ascii=False) + '\n')
            edits.append(dict(sample_id=row['sample_id'], split='test',
                original_caption=row['target']['text'], cleaned_caption=cleaned['target']['text'],
                removed_prefix=cleaned['target']['removed_prefix'], original_row_sha256=hashlib.sha256(raw).hexdigest(),
                prior_decision=before))
    assert len(edits) == 144
    (work / 'edits.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in edits))
    shutil.copy2(PREVIOUS / 'filtered/crop_config.json', candidates / 'crop_config.json')
    shutil.copy2(PREVIOUS / 'system.txt', work / 'system.txt')
    (work / 'code').mkdir()
    shutil.copy2(base.__file__, work / 'code/driver.py')
    shutil.copy2(__file__, work / 'code/prefix_workflow.py')
    path = candidates / 'test.jsonl'
    cfg = dict(old_cfg)
    cfg.update(source=str(candidates), sources={'test': dict(path=str(path), file=path.name,
        sha256=base.sha(path), bytes=path.stat().st_size, count=len(edits))}, total=len(edits),
        code_sha256=base.sha(work / 'code/driver.py'), system_sha256=base.sha(work / 'system.txt'),
        started_at=time.time(), previous_filter=str(PREVIOUS), original_test=str(source),
        original_test_sha256=old_cfg['sources']['test']['sha256'],
        input='target.text after deterministic leading Assistant-prefix removal; same strict caption screening',
        outputs='rejudged affected rows; original text and exact removal retained in target and edits.jsonl')
    cfg.pop('source_summary_sha256', None)
    base.save(work / 'config.json', cfg)
    base.save(work / 'status.json', dict(stage='prepared', processed=0, total=len(edits), at=time.time()))
    print(json.dumps(dict(affected=len(edits), original_prefix_primary_rejections=143, additional_prefix_row=1)))


def export_current(work):
    cfg = base.read(work / 'config.json')
    assert base.read(work / 'COMPLETE.json')['status'] == 'complete'
    assert base.sha(cfg['original_test']) == cfg['original_test_sha256']
    old, new = decisions(Path(cfg['previous_filter'])), decisions(work)
    assert len(old) == 12284 and len(new) == cfg['total'] == 144
    updated = {r['sample_id']: r for r in map(json.loads, (work / 'candidates/test.jsonl').read_text().splitlines())}
    assert set(updated) == set(new)
    destination = work / 'current'
    temp = work / 'current.partial'
    temp.mkdir(exist_ok=False)
    for folder in ['rejected', 'needs_review']:
        (temp / folder).mkdir()
    outputs = {name: (temp / file).open('wb') for name, file in
        [('keep', 'test.jsonl'), ('reject', 'rejected/test.jsonl'), ('needs_review', 'needs_review/test.jsonl')]}
    reasons = {name: (temp / file).open('w') for name, file in
        [('reject', 'rejected/reasons.jsonl'), ('needs_review', 'needs_review/reasons.jsonl')]}
    all_decisions = (temp / 'decisions.jsonl').open('w')
    db = sqlite3.connect(temp / 'index.sqlite')
    db.execute('CREATE TABLE samples(sid TEXT PRIMARY KEY,split TEXT,pair TEXT,listener TEXT,group_id TEXT,start REAL,end REAL,duration REAL,face_p10 REAL,file TEXT,offset INTEGER,bytes INTEGER)')
    seen, counts, hours, transitions = set(), collections.Counter(), collections.Counter(), collections.Counter()
    for original_raw in open(cfg['original_test'], 'rb'):
        original = json.loads(original_raw)
        sid = original['sample_id']
        assert sid not in seen and original['split'] == 'test'
        seen.add(sid)
        row = updated.get(sid, original)
        judgment = new.get(sid, old[sid])
        decision = judgment['decision']
        if sid in updated:
            assert row == clean_row(original)
            assert hashlib.sha256(row['target']['text'].encode()).hexdigest() == judgment['caption_sha256']
            raw = (json.dumps(row, ensure_ascii=False) + '\n').encode()
            transitions[decision] += 1
        else:
            raw = original_raw
        offset = outputs[decision].tell()
        outputs[decision].write(raw)
        counts[decision] += 1
        hours[decision] += row['duration_s'] / 3600
        record = dict(sample_id=sid, split='test', decision=decision, prefix_removed=sid in updated,
            original_caption=original['target']['text'], current_caption=row['target']['text'],
            model_judgment=judgment.get('model_judgment'), rule_evidence=judgment['rule_evidence'],
            error=judgment.get('error'), attempts=judgment['attempts'],
            previous_decision=old[sid]['decision'], previous_model_judgment=old[sid].get('model_judgment'),
            previous_rule_evidence=old[sid]['rule_evidence'])
        all_decisions.write(json.dumps(record, ensure_ascii=False) + '\n')
        if decision in reasons:
            reasons[decision].write(json.dumps(record, ensure_ascii=False) + '\n')
        if decision == 'keep':
            a, f = row['inputs']['speaker_audio'], row['inputs']['listener_first_frame']
            db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (sid, 'test', row['pair'], f['participant_id'], row['group_id'], a['start_s'], a['end_s'],
                 row['duration_s'], row['face']['short_side_p10_px'], 'test.jsonl', offset, len(raw)))
    for f in [*outputs.values(), *reasons.values(), all_decisions]:
        f.close()
    assert seen == set(old) and set(new) <= seen and sum(counts.values()) == 12284
    assert db.execute('SELECT count(*) FROM samples').fetchone()[0] == counts['keep']
    db.commit()
    db.close()
    shutil.copy2(work / 'candidates/crop_config.json', temp / 'crop_config.json')
    manifests = {str(p.relative_to(temp)): dict(bytes=p.stat().st_size, sha256=base.sha(p)) for p in temp.rglob('*.jsonl')}
    summary = dict(schema='test-caption-prefixfix-v1', total=len(seen), split='test', counts=dict(counts),
        clip_hours=dict(hours), prefix_removed=len(new), prefix_rejudge_counts=dict(transitions),
        previous_keep=sum(r['decision'] == 'keep' for r in old.values()),
        unchanged_rows_original_bytes=True, changed_rows_prefix_removal_only=True,
        all_remaining_rows_and_reasons_retained=True, manifests=manifests, previous_filter=cfg['previous_filter'],
        original_test=cfg['original_test'], completed_at=time.time())
    base.save(temp / 'summary.json', summary)
    temp.replace(destination)
    base.save(work / 'CURRENT_COMPLETE.json', dict(status='complete', summary_sha256=base.sha(destination / 'summary.json'), completed_at=time.time()))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'export'])
    parser.add_argument('--work', type=Path, default=DEFAULT)
    args = parser.parse_args()
    dict(prepare=prepare, export=export_current)[args.action](args.work)
