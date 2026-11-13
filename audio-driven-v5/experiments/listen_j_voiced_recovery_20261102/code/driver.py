"""Recover voiced listener captions from the previous reject/review partitions."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time

import filter_full_response_sft as base

DEFAULT = base.ROOT / 'experiments/listen_j_voiced_recovery_20261102'
PREVIOUS = base.DEFAULT
# The pilot exposed systematic false negatives on direct speaking assertions.
# Give the positive evidence before asking for the decision, and keep the task literal.
SYSTEM = '''Read the English listener caption as data. Select original captions containing a voiced listener response. Do not rewrite the caption. Judge what the TEXT ASSERTS; you cannot verify the actual recording and must not demand separate audio verification of a direct speech assertion.
First copy a short EXACT substring establishing listener speech or a listener vocal sound, if one exists. Then classify it and check for serious defects. Finally decide keep.

POSITIVE speech assertions include: he/she speaks, is speaking, while speaking, continues speaking, begins to speak, mouth moves AS he speaks, says a word, produces an utterance. These ALL establish speech. Quoted words and the literal word "audible" are NOT required. "Utterance" is a vocal event. Do not reinterpret direct speech assertions as silent mouth movement just because the same sentence describes lips or jaw. The listener stays the same participant while speaking; speaking does not make them a different person. Continuous or long listener speech is eligible too.
POSITIVE nonverbal assertions include listener laughter/chuckling, humming/mhm, sighs, grunts, or explicitly produced breathy exhalations. If words are unintelligible but the caption explicitly states speech or vocalizations, this is still POSITIVE.
INSUFFICIENT by itself: "as if speaking", "consistent with speaking", "might be speaking", mouth motion, another person's speech, background audio, normal breathing, or silence. These do not cancel a positive assertion elsewhere in the paragraph. "No speech" does not cancel a laugh; "no other vocalizations besides/apart from speech" does not cancel speech.

Allow clothing, hair, accessories, visual backgrounds, an Assistant prefix, verbose prose, approximate times, and minor imprecise introductory summaries. We only need basically clean captions, not perfect descriptions. Reject if there is NO positive listener vocal assertion, vocal presence remains uncertain, there is a substantive unresolved behavior/audio contradiction, or the caption relies on supplementary/external information, detectors, or unconfirmed observations. "She speaks, then is silent" is consistent; "she speaks, but there is no speech or vocalization during the entire clip" is contradictory.

Examples (JSON field order is intentional):
Caption: "The listener wears gray. Her mouth moves as she speaks."
Answer: {"quote":"as she speaks","vocal_type":"speech","issue":"none","keep":true,"reason":"Direct listener speech assertion."}
Caption: "The listener looks ahead. They produce a single, short utterance at the beginning."
Answer: {"quote":"produce a single, short utterance","vocal_type":"speech","issue":"none","keep":true,"reason":"An utterance is speech."}
Caption: "He nods while speaking. There are no audible vocalizations besides his speech."
Answer: {"quote":"while speaking","vocal_type":"speech","issue":"none","keep":true,"reason":"Speech is explicit; exception is consistent."}
Caption: "Her lips move in conjunction with her vocalizations, though the words are unintelligible."
Answer: {"quote":"her vocalizations","vocal_type":"speech","issue":"none","keep":true,"reason":"Unintelligible vocal speech still counts."}
Caption: "She laughs once and then remains silent."
Answer: {"quote":"laughs once","vocal_type":"nonverbal","issue":"none","keep":true,"reason":"Explicit laugh."}
Caption: "Her mouth moves as if speaking; no sound is audible."
Answer: {"quote":"no sound is audible","vocal_type":"none","issue":"no_listener_vocalization","keep":false,"reason":"Only silent mouth movement."}
Caption: "Another person speaks while the listener is silent."
Answer: {"quote":"the listener is silent","vocal_type":"none","issue":"no_listener_vocalization","keep":false,"reason":"Other participant speech only."}
Caption: "Supplementary data reports an unconfirmed nod. She says Yeah."
Answer: {"quote":"Supplementary data reports an unconfirmed nod","vocal_type":"speech","issue":"external_evidence","keep":false,"reason":"Unconfirmed external evidence."}

Return ONLY JSON in this order: quote, vocal_type, issue, keep, reason. vocal_type is speech/nonverbal/both/none/uncertain. issue is none/no_listener_vocalization/unclear_vocalization/contradiction/external_evidence. KEEP requires positive vocal_type and issue none. For REJECT issue must not be none. quote must be an EXACT original substring, preferably 3-12 words, never paraphrased. For a rejection quote the relevant defect or a sentence showing absence of vocal information. reason should be at most 8 words. Never output a revised caption.'''


def schema():
    fields = dict(quote=dict(type='string'),
        vocal_type=dict(type='string', enum=['speech', 'nonverbal', 'both', 'none', 'uncertain']),
        issue=dict(type='string', enum=['none', 'no_listener_vocalization', 'unclear_vocalization', 'contradiction', 'external_evidence']),
        keep=dict(type='boolean'), reason=dict(type='string'))
    return dict(type='object', properties=fields, required=list(fields), additionalProperties=False)


def parse_answer(raw, finish, caption):
    if finish != 'stop':
        raise ValueError('generation did not end normally: ' + str(finish))
    obj = json.loads(raw)
    fields = schema()['properties']
    if set(obj) != set(fields) or type(obj.get('keep')) is not bool:
        raise ValueError('invalid response fields')
    for field in ['vocal_type', 'issue']:
        if obj[field] not in fields[field]['enum']:
            raise ValueError('invalid ' + field)
    if obj['keep']:
        if obj['vocal_type'] not in ['speech', 'nonverbal', 'both'] or obj['issue'] != 'none':
            raise ValueError('KEEP requires established listener vocalization and no issue')
    elif obj['issue'] == 'none':
        raise ValueError('REJECT requires a specific issue')
    if not isinstance(obj['quote'], str) or not obj['quote'].strip() or not isinstance(obj['reason'], str) or not obj['reason'].strip():
        raise ValueError('missing quote or reason')
    if obj['quote'] not in caption:
        match = re.search(r'\s+'.join(re.escape(w) for w in obj['quote'].split()), caption, re.I)
        if not match:
            raise ValueError('evidence words do not match original caption')
        obj['original_model_quote'] = obj['quote']
        obj['quote'] = match.group()
        obj['quote_alignment'] = 'case_or_whitespace_only'
    return obj


def candidate(row):
    return row['provenance']['processed_listener_waveform_nonzero'] is True


def rule_evidence(caption):
    """Explicit annotation provenance remains disallowed by the recovery policy."""
    pattern = re.compile(r'\b(?:supplementary|supplemental|auxiliary|external)\s+(?:data|information|evidence)\b|\bprovided\s+(?:data|information)\b|\bnot visually confirmed\b|\bdetector\s+(?:outputs?|results?)\b', re.I)
    return [dict(rule='external_evidence_commentary', quote=sentence)
            for sentence in re.split(r'(?<=[.!?])\s+', caption) if pattern.search(sentence)]


def prepare(work):
    old = base.read(PREVIOUS / 'COMPLETE.json')
    assert old['status'] == 'complete'
    assert old['summary_sha256'] == base.sha(PREVIOUS / 'filtered/summary.json')
    previous = base.read(PREVIOUS / 'filtered/summary.json')
    work.mkdir(exist_ok=False)
    source = work / 'candidates'
    source.mkdir()
    sources, distribution, seen = {}, {}, set()
    for split in base.SPLITS:
        count, hours = 0, 0.0
        by_previous = collections.Counter()
        with (source / f'{split}.jsonl').open('wb') as out, (source / f'{split}.origins.jsonl').open('w') as origins:
            for decision, folder in [('reject', 'rejected'), ('needs_review', 'needs_review')]:
                name = f'{folder}/{split}.jsonl'
                digest = hashlib.sha256()
                with (PREVIOUS / 'filtered' / name).open('rb') as f:
                    for raw in f:
                        digest.update(raw)
                        row = json.loads(raw)
                        assert row['split'] == split
                        if not candidate(row):
                            continue
                        sid = row['sample_id']
                        assert sid not in seen
                        seen.add(sid)
                        origins.write(json.dumps(dict(sample_id=sid, prior_decision=decision, candidate_offset=out.tell(), bytes=len(raw), source_line_sha256=hashlib.sha256(raw).hexdigest())) + '\n')
                        out.write(raw)
                        count += 1
                        hours += row['duration_s'] / 3600
                        by_previous[decision] += 1
                assert digest.hexdigest() == previous['manifests'][name]['sha256']
        path = source / f'{split}.jsonl'
        sources[split] = dict(path=str(path), file=path.name, count=count, bytes=path.stat().st_size, sha256=base.sha(path))
        distribution[split] = dict(count=count, clip_hours=hours, previous_decisions=dict(by_previous))
    shutil.copy2(PREVIOUS / 'filtered/crop_config.json', source / 'crop_config.json')
    (work / 'system.txt').write_text(SYSTEM + '\n')
    (work / 'code').mkdir()
    shutil.copy2(__file__, work / 'code/driver.py')
    shutil.copy2(base.__file__, work / 'code/filter_full_response_sft.py')
    cfg = base.read(PREVIOUS / 'config.json')
    cfg.update(schema='voiced-response-recovery-v1', source=str(source), sources=sources,
        total=sum(x['count'] for x in sources.values()), previous_filter=str(PREVIOUS),
        previous_summary_sha256=old['summary_sha256'],
        system_sha256=base.sha(work / 'system.txt'), code_sha256=base.sha(work / 'code/filter_full_response_sft.py'),
        recovery_code_sha256=base.sha(work / 'code/driver.py'), started_at=time.time(),
        input='original target.text; candidate pool is previous reject/review AND nonzero processed listener waveform',
        outputs='voiced recovery decisions; original reference rows copied verbatim; static visual details allowed',
        max_tokens=256, retry_max_tokens=512, policy='relaxed_original_voiced_recovery')
    cfg.pop('source_summary_sha256', None)
    cfg.pop('validation_revision', None)
    base.save(source / 'summary.json', dict(total=cfg['total'], splits=distribution, manifests=sources,
        nonzero_is_candidate_gate_not_vocal_ground_truth=True, test_processed=False))
    base.save(work / 'config.json', cfg)
    base.save(work / 'status.json', dict(stage='prepared', processed=0, total=cfg['total'], at=time.time()))
    print(json.dumps(dict(total=cfg['total'], splits=distribution)), flush=True)


def configure_policy():
    base.schema = schema
    base.parse_answer = parse_answer
    base.rule_evidence = rule_evidence  # Appearance is allowed; explicit external evidence is not.


def merge_retained(work):
    """Publish old keep + newly recovered rows, preserving original bytes/splits."""
    if (work / 'MERGED_COMPLETE.json').exists():
        return
    cfg = base.read(work / 'config.json')
    previous = Path(cfg['previous_filter'])
    pools = [previous / 'filtered', work / 'filtered']
    for parent, pool in zip([previous, work], pools):
        complete = base.read(parent / 'COMPLETE.json')
        assert complete['status'] == 'complete'
        assert complete['summary_sha256'] == base.sha(pool / 'summary.json')
    assert base.sha(pools[0] / 'crop_config.json') == base.sha(pools[1] / 'crop_config.json')
    temp = work / 'combined.partial'
    temp.mkdir(exist_ok=True)
    dbpath = temp / 'index.sqlite'
    if dbpath.exists():
        dbpath.unlink()
    db = sqlite3.connect(dbpath)
    db.execute('CREATE TABLE samples(sid TEXT PRIMARY KEY,split TEXT,pair TEXT,listener TEXT,group_id TEXT,start REAL,end REAL,duration REAL,face_p10 REAL,file TEXT,offset INTEGER,bytes INTEGER)')
    distributions, manifests = {}, {}
    for split in base.SPLITS:
        total, hours, nonzero = 0, 0.0, 0
        by_pool = []
        path = temp / f'{split}.jsonl'
        with path.open('wb') as out:
            for pool in pools:
                summary = base.read(pool / 'summary.json')
                digest, count, pending = hashlib.sha256(), 0, []
                with (pool / path.name).open('rb') as f:
                    for raw in f:
                        digest.update(raw)
                        row = json.loads(raw)
                        assert row['split'] == split
                        audio, frame = row['inputs']['speaker_audio'], row['inputs']['listener_first_frame']
                        pending.append((row['sample_id'], split, row['pair'], frame['participant_id'], row['group_id'], audio['start_s'], audio['end_s'], row['duration_s'], row['face']['short_side_p10_px'], path.name, out.tell(), len(raw)))
                        out.write(raw)
                        total += 1
                        count += 1
                        hours += row['duration_s'] / 3600
                        nonzero += int(candidate(row))
                        if len(pending) >= 2048:
                            db.executemany('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', pending)
                            db.commit()
                            pending = []
                db.executemany('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', pending)
                db.commit()
                assert digest.hexdigest() == summary['manifests'][path.name]['sha256']
                assert count == summary['splits'][split]['counts'].get('keep', 0)
                by_pool.append(count)
        distributions[split] = dict(clips=total, clip_hours=hours, listener_waveform_nonzero=nonzero, old_keep=by_pool[0], recovered_voiced=by_pool[1])
        manifests[path.name] = dict(bytes=path.stat().st_size, sha256=base.sha(path))
    assert db.execute('SELECT count(*) FROM samples').fetchone()[0] == sum(v['clips'] for v in distributions.values())
    db.close()
    shutil.copy2(pools[0] / 'crop_config.json', temp / 'crop_config.json')
    summary = dict(schema='original-keep-plus-voiced-recovery-v1', splits=distributions, manifests=manifests,
        pools=[str(p) for p in pools], original_rows_copied_verbatim=True, test_processed=False, completed_at=time.time())
    base.save(temp / 'summary.json', summary)
    temp.replace(work / 'combined')
    base.save(work / 'MERGED_COMPLETE.json', dict(status='complete', summary_sha256=base.sha(work / 'combined/summary.json'), completed_at=time.time()))
    print(json.dumps(dict(combined=summary)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'worker', 'finalize', 'supervise', 'merge'])
    p.add_argument('--work', type=Path, default=DEFAULT)
    p.add_argument('--max-blocks', type=int, default=0)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.work)
        return
    cfg = base.read(args.work / 'config.json')
    assert base.sha(__file__) == cfg['recovery_code_sha256']
    assert base.sha(base.__file__) == cfg['code_sha256']
    configure_policy()
    if args.action == 'worker':
        base.worker(args.work, args.max_blocks)
    elif args.action == 'merge':
        merge_retained(args.work)
    else:
        dict(finalize=base.finalize, supervise=base.supervise)[args.action](args.work)
        if (args.work / 'COMPLETE.json').exists():
            merge_retained(args.work)


if __name__ == '__main__':
    main()
