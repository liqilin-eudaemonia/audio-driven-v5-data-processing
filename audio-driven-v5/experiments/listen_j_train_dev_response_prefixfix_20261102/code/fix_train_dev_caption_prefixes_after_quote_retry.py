"""Recover train/dev captions after prefix removal, preserving prior voiced keeps."""
import argparse
import collections
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import filter_full_response_sft as base
import recover_voiced_response_sft as voiced
from fix_test_caption_prefixes import clean_row

DEFAULT = base.ROOT / 'experiments/listen_j_train_dev_response_prefixfix_20261102'
BASELINE = voiced.DEFAULT / 'combined'
SPLITS = ('dev', 'train')
DRIVER = '''import argparse
from pathlib import Path
import filter_full_response_sft as base
import recover_voiced_response_sft as voiced
p=argparse.ArgumentParser()
p.add_argument('action', choices=['worker','supervise','finalize'])
p.add_argument('--work',type=Path,required=True)
a=p.parse_args()
cfg=base.read(a.work/'config.json')
if cfg.get('policy')=='relaxed_original_voiced_recovery':
    voiced.configure_policy()
dict(worker=base.worker,supervise=base.supervise,finalize=base.finalize)[a.action](a.work)
'''


def write_row(f, row):
    f.write(json.dumps(row, ensure_ascii=False) + '\n')


def judgments(work, selected=None):
    result = {}
    for split in SPLITS:
        for meta in sorted((work / 'batches' / split).glob('*.meta.json')):
            info = base.read(meta)
            path = work / info['result_file']
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for raw in f:
                    digest.update(raw)
                    r = json.loads(raw)
                    sid = r['sample_id']
                    if selected is None or sid in selected:
                        assert sid not in result
                        result[sid] = r
            assert digest.hexdigest() == info['result_sha256']
    return result


def setup_job(work, name, source, policy):
    job = work / name
    job.mkdir()
    (job / 'code').mkdir()
    for file in ['filter_full_response_sft.py', 'recover_voiced_response_sft.py', 'driver.py']:
        shutil.copy2(work / 'code' / file, job / 'code' / file)
    previous = base.DEFAULT if policy == 'strict' else voiced.DEFAULT
    cfg = base.read(previous / 'config.json')
    shutil.copy2(previous / 'system.txt', job / 'system.txt')
    sources = {}
    for split in SPLITS:
        path = source / f'{split}.jsonl'
        sources[split] = dict(path=str(path), file=path.name, count=sum(1 for _ in path.open('rb')),
                             bytes=path.stat().st_size, sha256=base.sha(path))
    cfg.update(source=str(source), sources=sources, splits=list(SPLITS),
               total=sum(r['count'] for r in sources.values()), started_at=time.time(),
               code_sha256=base.sha(job / 'code/filter_full_response_sft.py'),
               system_sha256=base.sha(job / 'system.txt'),
               input='prefix-stripped target.text; prior split and media references retained')
    cfg.pop('source_summary_sha256', None)
    base.save(job / 'config.json', cfg)
    base.save(job / 'status.json', dict(stage='prepared', total=cfg['total'], processed=0))
    return job


def prepare(work):
    old_summary = base.read(BASELINE / 'summary.json')
    marker = base.read(voiced.DEFAULT / 'MERGED_COMPLETE.json')
    assert marker['summary_sha256'] == base.sha(BASELINE / 'summary.json')
    original_cfg = base.read(base.DEFAULT / 'config.json')
    work.mkdir(exist_ok=False)
    (work / 'code').mkdir()
    for path in [Path(base.__file__), Path(voiced.__file__), Path(__file__),
                 Path(__file__).with_name('fix_test_caption_prefixes.py')]:
        shutil.copy2(path, work / 'code' / path.name)
    (work / 'code/driver.py').write_text(DRIVER)
    candidates = work / 'strict_candidates'
    candidates.mkdir()
    shutil.copy2(BASELINE / 'crop_config.json', candidates / 'crop_config.json')
    edited = work / 'edited'
    edited.mkdir()
    affected, counts = {}, {}
    with sqlite3.connect(BASELINE / 'index.sqlite') as db, (work / 'edits.jsonl').open('w') as edits:
        for split in SPLITS:
            tally = collections.Counter()
            digest = hashlib.sha256()
            src = Path(original_cfg['sources'][split]['path'])
            with src.open('rb') as f, (edited / f'{split}.jsonl').open('w') as out, (candidates / f'{split}.jsonl').open('w') as pending:
                for raw in f:
                    digest.update(raw)
                    row = json.loads(raw)
                    cleaned = clean_row(row)
                    if cleaned is None:
                        continue
                    sid = row['sample_id']
                    assert sid not in affected and row['split'] == split
                    kept = db.execute('SELECT split FROM samples WHERE sid=?', (sid,)).fetchone()
                    assert kept is None or kept[0] == split
                    affected[sid] = bool(kept)
                    write_row(out, cleaned)
                    write_row(edits, dict(sample_id=sid, split=split, already_kept=bool(kept),
                        original_caption=row['target']['text'], cleaned_caption=cleaned['target']['text'],
                        removed_prefix=cleaned['target']['removed_prefix'],
                        original_row_sha256=hashlib.sha256(raw).hexdigest()))
                    tally['prefix_total'] += 1
                    tally['already_kept' if kept else 'rejudge'] += 1
                    if not kept:
                        write_row(pending, cleaned)
                        tally['rejudge_nonzero' if voiced.candidate(row) else 'rejudge_zero'] += 1
            assert digest.hexdigest() == original_cfg['sources'][split]['sha256']
            counts[split] = dict(tally)
            print(json.dumps(dict(prepared_split=split, counts=counts[split])), flush=True)
    previous = judgments(base.DEFAULT, affected)
    recovery = judgments(voiced.DEFAULT, affected)
    assert set(previous) == set(affected)
    with (work / 'prior_decisions.jsonl').open('w') as f:
        for sid, kept in affected.items():
            assert kept == (previous[sid]['decision'] == 'keep' or recovery.get(sid, {}).get('decision') == 'keep')
            write_row(f, dict(sample_id=sid, strict=previous[sid], voiced=recovery.get(sid)))
    base.save(work / 'config.json', dict(schema='train-dev-prefix-recovery-v1', baseline=str(BASELINE),
        baseline_summary_sha256=base.sha(BASELINE / 'summary.json'), counts=counts,
        original_sources=original_cfg['sources'], started_at=time.time(), test_processed=False))
    setup_job(work, 'strict', candidates, 'strict')
    base.save(work / 'status.json', dict(stage='prepared', counts=counts))


def prepare_voiced(work):
    assert base.read(work / 'strict/COMPLETE.json')['status'] == 'complete'
    source = work / 'voiced_candidates'
    source.mkdir()
    for split in SPLITS:
        with (source / f'{split}.jsonl').open('wb') as out:
            for decision in ['rejected', 'needs_review']:
                for raw in (work / 'strict/filtered' / decision / f'{split}.jsonl').open('rb'):
                    if voiced.candidate(json.loads(raw)):
                        out.write(raw)
    shutil.copy2(BASELINE / 'crop_config.json', source / 'crop_config.json')
    return setup_job(work, 'voiced', source, 'voiced')


def final_decision(strict, recovery=None):
    if strict['decision'] == 'keep':
        return 'keep', 'strict'
    if recovery and recovery['decision'] == 'keep':
        return 'keep', 'voiced'
    if strict['decision'] == 'needs_review' or (recovery and recovery['decision'] == 'needs_review'):
        return 'needs_review', 'unresolved'
    return 'reject', 'rejected'


def export(work):
    cfg = base.read(work / 'config.json')
    assert cfg['baseline_summary_sha256'] == base.sha(BASELINE / 'summary.json')
    for job in ['strict', 'voiced']:
        complete = base.read(work / job / 'COMPLETE.json')
        assert complete['summary_sha256'] == base.sha(work / job / 'filtered/summary.json')
    strict, recovery = judgments(work / 'strict'), judgments(work / 'voiced')
    if (work / 'QUOTE_RETRY_COMPLETE.json').exists():
        marker = base.read(work / 'QUOTE_RETRY_COMPLETE.json')
        assert marker['sha256'] == base.sha(work / 'quote_retry.jsonl')
        for line in (work / 'quote_retry.jsonl').open():
            item = json.loads(line)
            target = strict if item['policy'] == 'strict' else recovery
            old = target[item['sample_id']]
            assert old == item['previous'] and old['decision'] == 'needs_review'
            target[item['sample_id']] = item['result']
    edited = {}
    for split in SPLITS:
        for line in (work / 'edited' / f'{split}.jsonl').open():
            r = json.loads(line)
            edited[r['sample_id']] = r
    priors = {r['sample_id']: r for r in map(json.loads, (work / 'prior_decisions.jsonl').read_text().splitlines())}
    pending = set(strict)
    assert set(recovery) <= pending and pending <= set(edited) == set(priors)
    temp = work / 'current.partial'
    temp.mkdir()
    for folder in ['recovered', 'rejected', 'needs_review']:
        (temp / folder).mkdir()
    db = sqlite3.connect(temp / 'index.sqlite')
    db.execute('CREATE TABLE samples(sid TEXT PRIMARY KEY,split TEXT,pair TEXT,listener TEXT,group_id TEXT,start REAL,end REAL,duration REAL,face_p10 REAL,file TEXT,offset INTEGER,bytes INTEGER)')
    summary = {}
    old_summary = base.read(BASELINE / 'summary.json')
    with (temp / 'prefix_decisions.jsonl').open('w') as all_reasons:
        for split in SPLITS:
            tally, hours, inserts, encountered = collections.Counter(), collections.Counter(), [], set()
            with contextlib.ExitStack() as stack:
                out = stack.enter_context((temp / f'{split}.jsonl').open('wb'))
                extras = {d: stack.enter_context((temp / folder / f'{split}.jsonl').open('wb')) for d, folder in
                          [('keep', 'recovered'), ('reject', 'rejected'), ('needs_review', 'needs_review')]}
                reasons = {d: stack.enter_context((temp / folder / f'{split}.reasons.jsonl').open('w')) for d, folder in
                           [('reject', 'rejected'), ('needs_review', 'needs_review')]}

                def add_keep(row, raw):
                    a, f = row['inputs']['speaker_audio'], row['inputs']['listener_first_frame']
                    inserts.append((row['sample_id'], split, row['pair'], f['participant_id'], row['group_id'],
                        a['start_s'], a['end_s'], row['duration_s'], row['face']['short_side_p10_px'],
                        split+'.jsonl', out.tell(), len(raw)))
                    out.write(raw)
                    tally['keep'] += 1
                    hours['keep'] += row['duration_s'] / 3600
                    if len(inserts) >= 2048:
                        db.executemany('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', inserts)
                        inserts.clear()

                digest = hashlib.sha256()
                for raw in (BASELINE / f'{split}.jsonl').open('rb'):
                    digest.update(raw)
                    original = json.loads(raw)
                    sid = original['sample_id']
                    assert sid not in pending and original['split'] == split
                    row = edited.get(sid, original)
                    if sid in edited:
                        assert row == clean_row(original)
                        raw = (json.dumps(row, ensure_ascii=False)+'\n').encode()
                        tally['previous_keep_prefix_cleaned'] += 1
                        encountered.add(sid)
                        write_row(all_reasons, dict(sample_id=sid, split=split, decision='keep',
                            route='previous_keep', normalization='leading Assistant prefix only', previous=priors[sid]))
                    add_keep(row, raw)
                assert digest.hexdigest() == old_summary['manifests'][f'{split}.jsonl']['sha256']
                assert tally['keep'] == old_summary['splits'][split]['clips']
                tally['previous_keep'] = tally['keep']
                for sid, row in edited.items():
                    if row['split'] != split or sid not in pending:
                        continue
                    encountered.add(sid)
                    decision, route = final_decision(strict[sid], recovery.get(sid))
                    for judgment in [strict[sid], recovery.get(sid)]:
                        if judgment:
                            assert judgment['caption_sha256'] == hashlib.sha256(row['target']['text'].encode()).hexdigest()
                    raw = (json.dumps(row, ensure_ascii=False)+'\n').encode()
                    extras[decision].write(raw)
                    record = dict(sample_id=sid, split=split, decision=decision, route=route,
                        original_caption=row['target']['original_text'], current_caption=row['target']['text'],
                        strict=strict[sid], voiced=recovery.get(sid), previous=priors[sid])
                    write_row(all_reasons, record)
                    if decision == 'keep':
                        add_keep(row, raw)
                        tally['new_keep'] += 1
                        tally['new_keep_'+route] += 1
                        hours['new_keep'] += row['duration_s']/3600
                    else:
                        tally[decision] += 1
                        hours[decision] += row['duration_s']/3600
                        write_row(reasons[decision], record)
                assert encountered == {sid for sid,r in edited.items() if r['split'] == split}
                assert len(encountered) == cfg['counts'][split]['prefix_total']
                assert sum(tally[x] for x in ['new_keep','reject','needs_review']) == cfg['counts'][split]['rejudge']
            db.executemany('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', inserts)
            db.commit()
            summary[split] = dict(counts=dict(tally), clip_hours=dict(hours))
            print(json.dumps(dict(exported_split=split, summary=summary[split])), flush=True)
    assert db.execute('SELECT count(*) FROM samples').fetchone()[0] == sum(r['counts']['keep'] for r in summary.values())
    db.close()
    shutil.copy2(BASELINE / 'crop_config.json', temp / 'crop_config.json')
    manifests = {str(p.relative_to(temp)): dict(bytes=p.stat().st_size,sha256=base.sha(p)) for p in temp.rglob('*.jsonl')}
    base.save(temp / 'summary.json', dict(schema=cfg['schema'], splits=summary, manifests=manifests,
        baseline=str(BASELINE), prior_rejection_archives=[str(base.DEFAULT), str(voiced.DEFAULT)],
        remainder_scope='rejected/needs_review here contain only prefix-affected candidates; unchanged exclusions remain in prior archives',
        all_previous_keeps_retained=True, only_leading_prefix_edited=True, test_processed=False, completed_at=time.time()))
    temp.replace(work / 'current')
    base.save(work / 'COMPLETE.json', dict(status='complete',summary_sha256=base.sha(work / 'current/summary.json'),completed_at=time.time()))
    base.save(work / 'status.json', dict(stage='complete',splits=summary))


def run(work):
    for name in ['strict', 'voiced']:
        if name == 'voiced' and not (work / name).exists():
            prepare_voiced(work)
        job = work / name
        with (job / 'supervisor.log').open('a') as log:
            subprocess.run([sys.executable, '-u', str(job / 'code/driver.py'), 'supervise', '--work', str(job)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        assert base.read(job / 'COMPLETE.json')['status'] == 'complete'
    export(work)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run', 'export'])
    p.add_argument('--work', type=Path, default=DEFAULT)
    args = p.parse_args()
    dict(prepare=prepare, run=run, export=export)[args.action](args.work)
