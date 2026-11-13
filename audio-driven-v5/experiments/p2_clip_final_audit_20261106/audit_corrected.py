"""Full selected-source -> sparse -> numeric audit and evidence conflict census."""
from pathlib import Path
import sys, json, sqlite3, hashlib, multiprocessing, time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from itertools import islice
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_ltx_events.current import _freeze_row,checked_index
from p2_clip_events.labels import tensors
from p2_clip_events.model import classes
from p2_ltx_events.evidence_guard import guard_row
from listen_benchmark.native_caption_quality import generation_failures
C=read(Path(__file__).resolve().parent/'candidate_config.json')
OUT=Path(__file__).resolve().parent/'corrected'
OUT.mkdir(exist_ok=True)
NAMES=classes(C['event_supervision'])

def init():
    global SOURCE,LABELS,ARRAYS
    SOURCE=sqlite3.connect((Path(C['current_source'])/'sources.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    LABELS=sqlite3.connect((Path(C['clip_labels'])/'labels.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    ARRAYS={k:np.load(Path(C['clip_numeric_labels'])/(k+'.npy'),mmap_mode='r') for k in ('y','known','weight','source_digests')}
    torch.set_num_threads(1)

def audit(args):
    ordinal,raw=args;selected=json.loads(raw);sid=selected['sample_id']
    item,derived=_freeze_row(raw)
    saved=json.loads(SOURCE.execute('SELECT value FROM samples WHERE ordinal=? AND sid=?',(ordinal,sid)).fetchone()[0])
    clip=json.loads(LABELS.execute('SELECT value FROM labels WHERE ordinal=? AND sid=?',(ordinal,sid)).fetchone()[0])
    cap=read(selected['caption_file'])
    ref=item['statistics']
    with Path(ref['file']).open('rb') as stream:
        stream.seek(ref['offset']);stat=json.loads(stream.readline())
    derived=guard_row(derived,cap['input_evidence'],item['source']['duration_s'],NAMES,.1,stat)
    if 'evidence_guard' in derived:
        bindings={}
        for name,key in [('visual:head nodding','nod_supplement'),('visual:eyebrow raising','reaction_supplement')]:
            if derived['evidence_guard']['reasons'].get(name) in ('nod_supplement_vs_weak_negative','raised_brows_vs_weak_negative'):
                supplement=cap['input_evidence'][key]
                if sha(supplement['detector_file'])!=supplement['detector_sha256']:raise ValueError('Detector identity changed')
                bindings[supplement['detector_file']]=supplement['detector_sha256']
        derived['evidence_guard']['detector_bindings']=bindings
    if item!=saved or derived!=clip:raise ValueError('Source/sparse derivation mismatch: '+sid)
    if ARRAYS['source_digests'][ordinal].decode()!=digest(item):raise ValueError('Numeric source mismatch: '+sid)
    expected=tensors(clip,NAMES,.1)
    for j,name in enumerate(NAMES):
        for key in ('y','known','weight'):
            if not np.array_equal(expected[name][key].numpy(),ARRAYS[key][ordinal,j:j+1]):
                raise ValueError('Numeric mismatch: '+sid+'/'+name+'/'+key)
    cap=read(selected['caption_file']);e=cap['input_evidence'];duration=item['source']['duration_s']
    counts=Counter(samples=1,**{'split/'+item['split']:1})
    counts['allow_weak']+=bool(clip['allow_weak']);counts['conflicts_masked']+=len(clip['conflicts'])
    counts['restored']+=bool(item.get('orientation_restoration'))
    flags=generation_failures(item['caption'],cap['finish_reason'])
    counts['literal_assistant']+=('assistant' in item['caption'].lower())
    counts['guarded_samples']+=bool(clip.get('evidence_guard'))
    for reason in clip.get('evidence_guard',{}).get('reasons',{}).values():counts['guard/'+reason]+=1
    def state(name):
        j=NAMES.index(name)
        return 'masked' if not ARRAYS['known'][ordinal,j] else ('positive' if ARRAYS['y'][ordinal,j] else 'weak_negative')
    def valid(v):return min(float(v['end']),duration)>max(float(v['start']),0)
    nod=any(valid(v) for v in e.get('nod_supplement',{}).get('events',[]))
    if nod:
        status=state('visual:head nodding');counts['nod_evidence/'+status]+=1
        if status=='weak_negative':flags.append('nod_evidence_weak_negative')
    cues=e.get('reaction_supplement',{}).get('cues',[])
    observations=sorted({v['observation'] for v in cues if valid(v)})
    for obs in observations:counts['reaction_observation/'+obs]+=1
    # Physical eyebrow raising is comparable to eyebrow raising, not surprise.
    if any(obs in ('brows raised','eyebrow raising','eyebrows raised') for obs in observations):
        status=state('visual:eyebrow raising');counts['eyebrow_evidence/'+status]+=1
        if status=='weak_negative':flags.append('eyebrow_evidence_weak_negative')
    silent=e['audio_identity']['full_silence']
    if silent:
        counts['silent']+=1
        for name in NAMES:
            if name.startswith('audio:'):
                status=state(name);counts['silent/'+name+'/'+status]+=1
                if status=='positive':flags.append('silent_positive/'+name)
    vocal=[v for v in e.get('vocal_supplement',[]) if valid(v)]
    if vocal:
        counts['vocal_evidence']+=1
        states=[state(name) for name in NAMES if name.startswith('audio:')]
        if all(s=='weak_negative' for s in states):flags.append('vocal_evidence_all_audio_weak_negative')
        for obs in sorted({v['observation'] for v in vocal}):counts['vocal_observation/'+obs]+=1
    counts['finish_reason/'+str(cap.get('finish_reason'))]+=1
    if cap.get('finish_reason')=='length':flags.append('caption_length_finish')
    # Structural prompt leakage markers, not ordinary English use of assistant.
    if any(mark in item['caption'].lower() for mark in ('<|im_start|>','<|im_end|>','assistant\n','assistant:','<think>','</think>')):
        flags.append('caption_role_marker')
    counts.update({'flag/'+flag:1 for flag in flags})
    detail=dict(sample_id=sid,ordinal=ordinal,flags=flags,caption_file=selected['caption_file'],caption=item['caption'],
        states={n:state(n) for n in NAMES if n in ('visual:head nodding','visual:eyebrow raising') or n.startswith('audio:')},
        nod_events=e.get('nod_supplement',{}).get('events',[]),reaction_cues=cues,vocal_cues=vocal,silent=silent) if flags else None
    return dict(counts=counts,duration=duration,detail=detail)

if __name__=='__main__':
    start=time.monotonic();index=checked_index(C['current_cache_index'])
    q=read(Path(index['caches']['qwen_hidden']['work'])/'protocol.json');selection=Path(q['selection'])
    assert sha(selection)==index['selection_sha256']
    counts=Counter();hours=0;examples=defaultdict(list)
    with selection.open('rb') as stream,(OUT/'FLAGGED.jsonl').open('w') as flagged,ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('fork'),initializer=init) as pool:
        numbered=enumerate(stream)
        while batch:=list(islice(numbered,4096)):
            for result in pool.map(audit,batch,chunksize=16):
                counts.update(result['counts']);hours+=result['duration']/3600
                detail=result['detail']
                if detail:
                    flagged.write(json.dumps(detail,ensure_ascii=False)+'\n')
                    for flag in detail['flags']:
                        if len(examples[flag])<5:examples[flag].append(detail)
            atomic_json(OUT/'label_progress.json',dict(samples=counts['samples'],total=index['samples'],counts=counts,elapsed_s=time.monotonic()-start))
            print(json.dumps(dict(samples=counts['samples'],flags={k:v for k,v in counts.items() if k.startswith('flag/')})),flush=True)
    assert counts['samples']==index['samples']
    atomic_json(OUT/'LABEL_AUDIT.json',dict(structural_success=True,samples=counts['samples'],hours=hours,counts=counts,examples=examples,
        elapsed_s=time.monotonic()-start,selection_sha256=sha(selection),config_sha256=sha(Path(__file__).resolve().parent/'candidate_config.json'),
        source_manifest_sha256=sha(Path(C['current_source'])/'manifest.json'),clip_manifest_sha256=sha(Path(C['clip_labels'])/'manifest.json'),
        numeric_manifest_sha256=sha(Path(C['clip_numeric_labels'])/'manifest.json'),training_started=False))
    print(json.dumps(dict(done=True,samples=counts['samples'],flags={k:v for k,v in counts.items() if k.startswith('flag/')},hours=hours,elapsed_s=time.monotonic()-start)),flush=True)
