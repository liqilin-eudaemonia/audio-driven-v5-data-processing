import hashlib, json, os
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
TRAIN = ROOT/'experiments/p2_full50h_duration384_20261028'
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def checkpoint(loss, step):
    return TRAIN/f'parallel_runs/guided_{loss}_s1/w24_from_000000/step_{step:06d}.pt'
def ready(path):
    p=Path(path);m=p.with_suffix('.complete.json')
    if not p.is_file() or not m.is_file():return False
    s=p.stat();return _checked(str(p),s.st_size,s.st_mtime_ns,sha(m))
@lru_cache(maxsize=64)
def _checked(path,size,mtime,marker_hash):
    return read(Path(path).with_suffix('.complete.json')).get('sha256')==sha(path)
def english_extension():
    # Same previously authorized English-only evaluation policy, explicitly traced
    # by native render(). Original annotation fields remain in source_record.
    import p2_guided_prompt_en.formatting as f
    f.TRANSLATIONS.update({'哈':'Ha','呵':'Heh'})
def guidance_item(obs,record):
    import copy
    from p2_guided_prompt_en.formatting import render
    from p2_plan_cache.protocol import render_clean
    from p2_listen.contracts import digest
    english_extension();english=render(record);translated=copy.deepcopy(record)
    events={e['event_id']:e for e in translated['events']}
    for entry in english['translations']:events[entry['event_id']][entry['field']]=entry['english']
    translated['evaluation_translation']=dict(original_record_sha256=digest(record),translations=english['translations'])
    guidance=render_clean(translated)
    assert guidance['prompt']==english['prompt']
    return dict(observation=obs,guidance=guidance,original_record=record,translations=english['translations'])
def environment(gpu):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True',
                P2_PLAN_GPU_MEMORY_GIB='64',CUBLAS_WORKSPACE_CONFIG=':4096:8')

def prepare():
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from p2_listen.contracts import digest,atomic_json
    from p2_plan_cache.protocol import code_identity
    from p2_guided_prompt_en.formatting import render
    from p2_guided_prompt_en.protocol import prepare as old_prepare
    english_extension()
    if not (HERE/'test48/COMPLETE.json').exists():raise RuntimeError('Full test48 annotations still pending')
    completion=read(HERE/'test48/COMPLETE.json')
    if completion['labels_sha256']!=sha(HERE/'test48/integrated_labels.jsonl'):raise ValueError('Label completion mismatch')
    all_test={r['sample_id']:r for r in map(json.loads,(HERE/'test48/integrated_labels.jsonl').read_text().splitlines())}
    original_test={r['sample_id']:r for r in map(json.loads,(HERE/'test24/integrated_labels.jsonl').read_text().splitlines())}
    assert len(all_test)==48 and len(original_test)==24 and all(all_test[s]==r for s,r in original_test.items())
    base=read(ROOT/'experiments/p2_full50h_hard_keep_20261028/plans/protocol.json')
    entries={};observations=[];sources={str(HERE/'eval_common.py'):sha(HERE/'eval_common.py')};translations=[]
    for split in ['train24','test24']:
        obs=read(HERE/split/'observations.json'); records={r['sample_id']:r for r in map(json.loads,(HERE/split/'integrated_labels.jsonl').read_text().splitlines())}
        assert set(records)=={o['sample_id'] for o in obs}
        observations+=obs
        for name in ['observations.json','integrated_labels.jsonl']:
            p=HERE/split/name;sources[str(p)]=sha(p)
        for o in obs:
            r=records[o['sample_id']];assert abs(r['duration_s']-o['requested_duration_s'])<1e-6
            item=guidance_item(o,r);g=item['guidance'];entries[o['sample_id']]=digest(item)
            atomic_json(HERE/'plans/samples'/(o['sample_id']+'.json'),item,immutable=True)
            translations.append(dict(sample_id=o['sample_id'],translations=item['translations']))
    protocol=dict(schema='p2-plan-cache-v1',config=base['config'],sources=sources,code=code_identity(),
        sample_ids=sorted(entries),entries=entries,max_text_tokens=8192,max_qwen_tokens=32768,
        plan_source='annotation_oracle',target_annotation_access=True,target_media_access=False,
        qwen_frozen=True,qwen_lora=False,truncation=False,annotation_text_policy='English evaluation rendering with original and translation trace')
    atomic_json(HERE/'plans/protocol.json',protocol,immutable=True)
    old_prepare(HERE/'test48/observations.json',HERE/'test48/integrated_labels.jsonl',HERE/'legacy_prompts',4096)
    atomic_json(HERE/'translations.json',translations,immutable=True)
    return build_jobs()

def build_jobs():
    from p2_listen.contracts import atomic_json
    old=read(ROOT/'experiments/p2_test24_fourway_eval_20261027/experiments.json')
    jobs=[]
    # User scope: final checkpoints only. Never enqueue step200.
    for loss,step in [('av',1249),('av_event',1249),('av_feature',1249)]:
        for split in ['test24','train24']:
            for o in read(HERE/split/'observations.json'):
                for seed in [137,251]:
                    sid=o['sample_id'];arm=f'guided_{loss}_step{step}'
                    jobs.append(dict(family='matrix',arm=arm,split=split,sample_id=sid,seed=seed,checkpoint=str(checkpoint(loss,step)),
                        output=str(HERE/'outputs'/arm/split/f'{sid}__seed{seed}')))
    for arm,e in old.items():
        for o in read(HERE/'test48/observations.json'):
            for seed in [137,251]:
                sid=o['sample_id'];jobs.append(dict(family='legacy',arm='legacy_'+arm,split='test48',sample_id=sid,seed=seed,
                    checkpoint=e['checkpoint'],run=e['run'],restoration=str(ROOT/'experiments/p2_test24_fourway_eval_20261027/source_restoration.json'),
                    output=str(HERE/'outputs'/('legacy_'+arm)/'test48'/f'{sid}__seed{seed}')))
    assert len(jobs)==672 and all('step200' not in j['arm'] for j in jobs)
    atomic_json(HERE/'jobs.json',jobs,immutable=True)
    return jobs
