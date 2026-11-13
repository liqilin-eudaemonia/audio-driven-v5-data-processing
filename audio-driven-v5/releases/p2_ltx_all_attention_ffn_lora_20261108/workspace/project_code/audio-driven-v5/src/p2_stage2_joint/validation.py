"""Freeze dev-only trial validation after excluding both training sets and media overlaps."""
from bisect import bisect_right
from collections import defaultdict,Counter
from pathlib import Path
import hashlib,json,random
from p2_listen.contracts import read,sha,digest,atomic_json


def spans(row):
    if row.get('schema')=='j-upright-response-sft-reference-v1':
        audio=row['inputs']['speaker_audio'];first=row['inputs']['listener_first_frame']
        a,b=audio['start_s'],audio['end_s'];ids=[audio['participant_id'],first['participant_id']]
    else:
        s=row['source'];a,b=s['source_start_s'],s['source_end_s'];ids=[s['speaker_id'],s['listener_id']]
    if not 0<=a<b or len(set(ids))!=2:raise ValueError('Invalid paired media identity/time')
    return [(key,float(a),float(b))for key in ids]


class OverlapIndex:
    def __init__(self):self.pending=defaultdict(list);self.index={}
    def add(self,row):
        for key,a,b in spans(row):self.pending[key].append((a,b))
    def freeze(self):
        for key,items in self.pending.items():
            merged=[]
            for a,b in sorted(items):
                if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
                else:merged.append((a,b))
            self.index[key]=([a for a,b in merged],[b for a,b in merged])
        self.pending.clear()
    def overlaps(self,row,margin=.04):
        for key,a,b in spans(row):
            starts,ends=self.index.get(key,([],[]))
            i=bisect_right(starts,b+margin)-1
            if i>=0 and ends[i]>=a-margin:return True
        return False


def jsonl(path):
    with Path(path).open('rb')as f:
        for raw in f:
            if raw.strip():yield json.loads(raw)


def select(seal_path,selection_path,out,count=64,seed=20261108):
    """No model predictions or test rows enter selection; hashes bind the original sources."""
    if count!=64:raise ValueError('This trial freezes exactly64 independent dev samples')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'validation.json').exists():raise FileExistsError('Use an independent validation output')
    seal=read(seal_path);directory=Path(seal['directory'])
    for name in ('train.jsonl','dev.jsonl','index.sqlite'):
        if sha(directory/name)!=seal['files'][name]['sha256']:raise ValueError('Planning seal changed: '+name)
    index=OverlapIndex();training_ids=set();planning_ids=[];planning_captions={};n=0
    for row in jsonl(directory/'train.jsonl'):
        if row['split']!='train':raise ValueError('Wrong planning training split')
        sid=row['sample_id'];planning_ids.append(sid);training_ids.add(sid)
        planning_captions[sid]=row['target']['source_sha256'];index.add(row);n+=1
    if n!=seal['splits']['train']['count']or digest(sorted(planning_ids))!=seal['splits']['train']['ids_sha256']:
        raise ValueError('Planning training membership differs')
    stage1_count=0
    for row in jsonl(selection_path):training_ids.add(row['sample_id']);index.add(row);stage1_count+=1
    if stage1_count!=284218:raise ValueError('Require final500h selection')
    index.freeze();groups=defaultdict(list);excluded=Counter()
    for row in jsonl(directory/'dev.jsonl'):
        if row['split']!='dev':raise ValueError('No test or train candidates allowed')
        sid=row['sample_id']
        if sid in training_ids:excluded['training_id']+=1;continue
        if index.overlaps(row):excluded['training_media_overlap']+=1;continue
        caption=Path(row['target']['source_file'])
        if sha(caption)!=row['target']['source_sha256']:raise ValueError('Validation caption changed')
        evidence=read(caption)['input_evidence']
        visual=bool(evidence.get('nod_supplement',{}).get('events'))or bool(evidence.get('reaction_supplement',{}).get('cues'))
        voice=row['provenance']['processed_listener_waveform_nonzero']
        duration=row['duration_s'];stratum=(voice,visual,0 if duration<4 else 1 if duration<8 else 2)
        groups[stratum].append(row)
    rng=random.Random(seed)
    for key in groups:rng.shuffle(groups[key])
    chosen=[];validation_index=OverlapIndex();keys=sorted(groups)
    # Round-robin strata, also avoid overlapping validation clips counting twice.
    while len(chosen)<count and any(groups.values()):
        for key in keys:
            if not groups[key]or len(chosen)>=count:continue
            row=groups[key].pop()
            if any(a[0]==b[0]and min(a[2],b[2])>max(a[1],b[1])for old in chosen for a in spans(row)for b in spans(old)):
                excluded['validation_media_overlap']+=1;continue
            chosen.append(row)
    if len(chosen)!=count:raise ValueError('Not enough disjoint independent dev samples; never fall back to test/train')
    atomic_json(out/'planning_training_ids.json',sorted(planning_ids),immutable=True)
    atomic_json(out/'planning_caption_hashes.json',planning_captions,immutable=True)
    atomic_json(out/'validation.json',chosen,immutable=True)
    receipt=dict(success=True,count=count,seed=seed,stage1_samples=stage1_count,planning_training_samples=n,
        selection_sha256=sha(selection_path),planning_seal_sha256=sha(seal_path),planning_seal=str(Path(seal_path).resolve()),
        excluded=dict(excluded),no_training_ids=True,no_training_media_overlap=True,overlap_margin_s=.04,
        no_validation_media_overlap=True,source_split='dev',test_used=False,
        validation_sha256=sha(out/'validation.json'),planning_ids_sha256=sha(out/'planning_training_ids.json'),
        planning_caption_hashes_sha256=sha(out/'planning_caption_hashes.json'),
        strata='nonzero listener waveform x existing visual supplement present x duration<4/<8/>=8; no new event labels')
    atomic_json(out/'VALIDATION_SELECTED.json',receipt,immutable=True)
    return receipt
