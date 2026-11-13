"""Mask contradictory weak negatives using frozen current caption evidence.

No new positives, subtype guesses, text parsing, or model inference. Preparation
is offline; the training path still consumes only precomputed numeric labels.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

from p2_listen.contracts import read, sha, digest, atomic_json
from p2_clip_events.labels import LabelReader
from p2_clip_events.model import classes

POLICY = 'current-supplement-conflict-mask-v1'


def guard_row(row, evidence, duration, names, absence_weight=.1, statistics=None):
    result = copy.deepcopy(row)
    positive, unknown = set(row['positive']), set(row['unknown'])
    reasons = {}
    def weak(name):
        return name in names and name not in positive|unknown and row['allow_weak'] and absence_weight>0
    def valid(cue):
        a,b=float(cue['start']),float(cue['end'])
        if not math.isfinite(a) or not math.isfinite(b) or b<a:
            raise ValueError('Invalid supplement interval')
        return min(b,duration)>max(a,0.)
    # Mentioned baseline evidence is not an omitted class. Do not promote it
    # to a positive, but do not infer absence from the baseline disposition.
    for entry in (statistics or {}).get('evidence_ledger',[]):
        name=('visual:' if entry['raw']['modality']=='visual' else 'audio:')+entry['canonical']
        if entry['disposition']=='baseline' and weak(name):
            reasons[name]='mentioned_baseline_not_confirmed_absence'
    nod=evidence.get('nod_supplement',{}).get('events',[])
    if weak('visual:head nodding') and any(valid(cue) for cue in nod):
        reasons['visual:head nodding']='nod_supplement_vs_weak_negative'
    raised=[cue for cue in evidence.get('reaction_supplement',{}).get('cues',[])
            if cue['observation'] in ('brows raised','eyebrow raising','eyebrows raised')]
    if weak('visual:eyebrow raising') and any(valid(cue) for cue in raised):
        reasons['visual:eyebrow raising']='raised_brows_vs_weak_negative'
    # Brow lowering does not imply frown; raised brows do not imply surprise.
    audio=[name for name in names if name.startswith('audio:')]
    silent=evidence['audio_identity']['full_silence']
    if silent:
        for name in audio:
            if name in positive:reasons[name]='silent_target_vs_audio_positive'
    elif audio and all(weak(name) for name in audio):
        vocal=any(valid(cue) for cue in evidence.get('vocal_supplement',[]))
        other_positive=any(name.startswith('audio:') for name in positive)
        # Generic vocal evidence cannot choose among speech / nonverbal / laugh.
        # Similarly an open audio category outside this vocabulary must not be
        # turned into simultaneous absence of all configured audio categories.
        if vocal or other_positive:
            reason='vocal_supplement_vs_all_audio_weak_negative' if vocal else 'other_positive_audio_outside_vocabulary'
            reasons.update({name:reason for name in audio})
    if reasons:
        result['positive']=sorted(positive-set(reasons))
        result['unknown']=sorted(unknown|set(reasons))
        result['evidence_guard']=dict(policy=POLICY,masked_classes=sorted(reasons),reasons=reasons,
                                      caption_sha256=row['caption_sha256'])
    return result


_CONTEXT=None

def _worker_init(c):
    global _CONTEXT
    source=sqlite3.connect((Path(c['current_source'])/'sources.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    labels=sqlite3.connect((Path(c['clip_labels'])/'labels.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    _CONTEXT=(c,source,labels,classes(c['event_supervision']),read(Path(c['clip_labels'])/'manifest.json')['absence_weight'])


def _guard_range(bounds):
    c,source,labels,names,weight=_CONTEXT;lo,hi=bounds;updates=[];count=0
    sr=source.execute('SELECT ordinal,sid,value FROM samples WHERE ordinal>=? AND ordinal<? ORDER BY ordinal',(lo,hi))
    lr=labels.execute('SELECT ordinal,sid,value FROM labels WHERE ordinal>=? AND ordinal<? ORDER BY ordinal',(lo,hi))
    for (i,sid,sv),(j,lid,lv) in zip(sr,lr,strict=True):
        if i!=j or sid!=lid:raise ValueError('Guard source/label order mismatch')
        item=json.loads(sv);row=json.loads(lv)
        if row['sample_id']!=sid or row['source_digest']!=digest(item):raise ValueError('Guard source identity mismatch')
        raw=Path(item['statistics']['caption_file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['caption_sha256'] or row['caption_sha256']!=item['caption_sha256']:
            raise ValueError('Guard caption identity mismatch')
        cap=json.loads(raw);e=cap['input_evidence']
        if (cap['sample_id']!=sid or cap['caption']!=item['caption'] or cap['source']!=item['source']
                or e['audio_identity']!=item['listener_audio_identity']):raise ValueError('Guard caption/media mismatch')
        stat_ref=item['statistics']
        with Path(stat_ref['file']).open('rb') as stream:
            stream.seek(stat_ref['offset']);stat_raw=stream.readline()
        if hashlib.sha256(stat_raw).hexdigest()!=stat_ref['line_sha256']:
            raise ValueError('Guard statistics changed: '+sid)
        statistics=json.loads(stat_raw)
        if statistics['sample_id']!=sid or statistics['caption']!=item['caption']:
            raise ValueError('Guard statistics/caption mismatch')
        guarded=guard_row(row,e,item['source']['duration_s'],names,weight,statistics)
        if guarded!=row:
            # Check original detector files for the new visual masks. We retain
            # the caption's evidence as a conflict signal, never as ground truth.
            bindings={}
            for name,key in [('visual:head nodding','nod_supplement'),('visual:eyebrow raising','reaction_supplement')]:
                if guarded['evidence_guard']['reasons'].get(name) in ('nod_supplement_vs_weak_negative','raised_brows_vs_weak_negative'):
                    supplement=e[key];path=Path(supplement['detector_file']);expected=supplement['detector_sha256']
                    if sha(path)!=expected:raise ValueError('Guard detector changed: '+sid)
                    bindings[str(path)]=expected
            guarded['evidence_guard']['detector_bindings']=bindings
            updates.append((i,sid,guarded))
        count+=1
    if count!=hi-lo:raise ValueError('Guard incomplete range')
    return count,updates


def prepare_guarded(c,output,processes=8):
    if not c.get('current_cache_index') or processes<1:raise ValueError('Expected current-cache configuration and positive process count')
    labels=LabelReader(c['clip_labels']);temporary=None
    try:
        original=copy.deepcopy(labels.manifest)
        if original.get('evidence_guard'):raise ValueError('Guard the original label bank; do not stack policies')
        source_path=Path(c['current_source']);source=read(source_path/'manifest.json')
        if (sha(source_path/'sources.sqlite')!=source['database_sha256'] or labels.ids()!=source['training_ids']
                or original['cache_manifest_sha256']!=sha(source_path/'manifest.json')):raise ValueError('Guard source bank mismatch')
        output=Path(output).resolve()
        if output.exists():raise FileExistsError('Use a new guarded label directory')
        output.parent.mkdir(parents=True,exist_ok=True)
        temporary=Path(tempfile.mkdtemp(prefix=output.name+'.tmp-',dir=output.parent))
        target=sqlite3.connect(temporary/'labels.sqlite')
        try:labels.conn.backup(target)
        finally:target.close()
        checked=changed=0;reasons=Counter();masks=Counter()
        with sqlite3.connect(temporary/'labels.sqlite') as db,(temporary/'changes.jsonl').open('w') as log:
            with ProcessPoolExecutor(max_workers=processes,mp_context=multiprocessing.get_context('fork'),initializer=_worker_init,initargs=(c,)) as pool:
                bounds=((i,min(i+1024,original['samples'])) for i in range(0,original['samples'],1024))
                for count,updates in pool.map(_guard_range,bounds,chunksize=1):
                    checked+=count
                    for ordinal,sid,row in updates:
                        db.execute('UPDATE labels SET value=? WHERE ordinal=? AND sid=?',(json.dumps(row,separators=(',',':')),ordinal,sid))
                        guard=row['evidence_guard'];masks.update(guard['masked_classes']);reasons.update(guard['reasons'].values())
                        log.write(json.dumps(dict(sample_id=sid,**guard),ensure_ascii=False)+'\n');changed+=1
                    db.commit()
                    if checked%16384==0:print(json.dumps(dict(checked=checked,changed=changed)),flush=True)
            pos=Counter();weak_pos=Counter();weak_unknown=Counter();weak_rows=0
            for (value,) in db.execute('SELECT value FROM labels ORDER BY ordinal'):
                row=json.loads(value);pos.update(row['positive'])
                if set(row['positive'])&set(row['unknown']):raise ValueError('Guard positive/unknown overlap')
                if row['allow_weak']:
                    weak_rows+=1;weak_pos.update(row['positive']);weak_unknown.update(row['unknown'])
        if checked!=original['samples'] or sha(Path(c['clip_labels'])/'labels.sqlite')!=original['database_sha256']:
            raise ValueError('Guard incomplete or source changed')
        stats={};weight=original['absence_weight']
        for name in original['statistics']:
            negative=weak_rows-weak_pos[name]-weak_unknown[name];positive=pos[name]
            if negative<0:raise ValueError('Invalid guarded negative count')
            enabled=positive>0 and negative>0 and weight>0
            stats[name]=dict(positive=positive,weak_negative=negative,known_weight=positive+weight*negative,enabled=enabled,
                pos_weight=max(1.,min(10.,weight*negative/max(positive,1))),disabled_reason=None if enabled else 'no_positive_negative_contrast')
        manifest=dict(original,database_sha256=sha(temporary/'labels.sqlite'),statistics=stats,
            label_policy=original['label_policy']+'; '+POLICY,
            evidence_guard=dict(policy=POLICY,classes=list(classes(c['event_supervision'])),checked=checked,changed_samples=changed,
                masked_by_class=masks,reasons=reasons,changes_sha256=sha(temporary/'changes.jsonl'),
                base_clip_labels=str(Path(c['clip_labels']).resolve()),base_manifest_sha256=sha(Path(c['clip_labels'])/'manifest.json'),
                code_sha256=sha(__file__),new_positive_targets=0))
        atomic_json(temporary/'manifest.json',manifest)
        os.rename(temporary,output);temporary=None
        return dict(output=str(output),samples=checked,changed_samples=changed,masked_by_class=masks,
                    new_positive_targets=0,training_started=False,config_entry={'clip_labels':str(output)})
    finally:
        labels.conn.close()
        if temporary is not None:shutil.rmtree(temporary)
