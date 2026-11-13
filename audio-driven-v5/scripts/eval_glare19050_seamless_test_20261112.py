"""Checkpoint19050 full Seamless6624; reuse its completed200, all input caches."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
import eval_glare_seamless_test_20261112 as native
from p2_testsets_eval.protocol import read,rows,sha

WORK=ROOT/'experiments/glare19050_seamless_test6624_20261112'
QUICK=ROOT/'experiments/glare19050_vico43_responsenet586_seamless200_20261112'
FINAL=ROOT/'experiments/glare_final_seamless_test6624_20261112'
SELF=Path(__file__).resolve()
native.WORK=WORK;native.SELF=SELF
core=native.core;core.WORK=WORK;core.SELF=SELF
core.CHECKPOINT=ROOT/'experiments/glare_paper_settings_node11_gpu7_20261111/before_extension30epochs.pt'
core.MODEL='GLARE step19050';core.ARM='glare_step19050';core.claim=native.claim
save=core.save

def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():dst.symlink_to(src,target_is_directory=src.is_dir())

def prepare():
    WORK.mkdir(exist_ok=True)
    if (WORK/'PREPARED.json').exists():return
    assert read(QUICK/'COMPLETE.json')['passed'] and read(FINAL/'INPUT_COMPLETE.json')['clips']==6624
    assert read(FINAL/'PROSODY_COMPLETE.json')['clips']==6624
    parent=read(FINAL/'protocol.json');old=read(QUICK/'protocol.json');assert old['step']==19050 and sha(core.CHECKPOINT)==old['checkpoint_sha256']
    samples=[]
    for source in parent['samples']:
        s=copy.deepcopy(read(source['ready'])['sample']);s['ready']=str(WORK/'inputs'/s['id']/'READY.json')
        assert sha(s['reference'])==s['reference_sha256'] and sha(s['audio'])==s['audio_sha256']
        samples.append(s)
    assert len(samples)==len({s['id'] for s in samples})==6624
    p=copy.deepcopy(parent)
    p.update(schema='GLARE19050-Seamless-full6624-reuse200-GPU4-v1',created=time.time(),step=19050,model=core.MODEL,
        checkpoint=str(core.CHECKPOINT),checkpoint_sha256=old['checkpoint_sha256'],samples=samples,decode_batch=4,
        generation_workers=1,generation_min_free_mib=30*1024,expanded_generation_workers=4,expand_generation_after=str(FINAL/'GENERATION_COMPLETE.json'),
        score_workers_during_generation=2,score_workers_after_generation=8,
        reused_completed200=str(QUICK),reused_all_inputs_and_prosody=str(FINAL),
        acceleration='Reuse all6624 input/prosody caches,200 generated native/scoring videos and8 complete score blocks;1 generator now,4 after final32940 generation finishes;22GiB per generator;decode4',
        code={str(f):sha(f) for f in [SELF,ROOT/'scripts/eval_glare_seamless_test_20261112.py',ROOT/'scripts/eval_glare_final_20261112.py',ROOT/'scripts/glare_resident_inference.py',ROOT/'scripts/infer_glare_checkpoint.py']})
    save(WORK/'protocol.json',p);protocol_sha=sha(WORK/'protocol.json')
    link(FINAL/'prosody',WORK/'prosody')
    _,key=core.prosody_id()
    for s in samples:
        assert core.cache_path(s,key).exists()
        save(Path(s['ready']),dict(sample=s,reused=True,source_only=True,protocol_sha256=protocol_sha))
    for i in range(p['metric_shards']):
        ids=[s['id'] for s in read(FINAL/'blocks'/f'{i:03d}.json')];byid={s['id']:s for s in samples}
        save(WORK/'blocks'/f'{i:03d}.json',[byid[uid] for uid in ids])
    with (WORK/'ready_queue.jsonl').open('w') as f:
        for s in samples:f.write(json.dumps(dict(id=s['id'],ready=s['ready']))+'\n')
    reused=[]
    for s in samples[:200]:
        uid=s['id'];source=QUICK/'seamless/generated'/uid;r=read(source/'complete.json')
        assert r['step']==19050 and r['checkpoint_sha256']==p['checkpoint_sha256'] and r['reference_sha256']==s['reference_sha256'] and r['audio_sha256']==s['audio_sha256']
        for src,dst,h in [(source/'generated.mp4',WORK/'seamless/generated'/uid/'generated.mp4',r['native_sha256']),
                          (QUICK/'seamless/scoring'/core.ARM/(uid+'.listener.mp4'),WORK/'seamless/scoring'/core.ARM/(uid+'.listener.mp4'),r['score_sha256'])]:
            assert sha(src)==h;link(src,dst)
        receipt=dict(r,protocol_sha256=protocol_sha,reused_from=str(source/'complete.json'),original_receipt_sha256=sha(source/'complete.json'))
        save(WORK/'seamless/generated'/uid/'complete.json',receipt);reused.append(uid)
    assert set(reused)==set(p['quick200_ids'])
    for i in range(8):
        src=QUICK/'metrics'/f'{i:03d}';assert (src/'SUCCESS.json').exists();link(src,WORK/'metrics'/f'{i:03d}')
    save(WORK/'INPUT_COMPLETE.json',dict(clips=6624,reused=6624))
    save(WORK/'PROSODY_COMPLETE.json',dict(clips=6624,reused=True))
    save(WORK/'PREPARED.json',dict(passed=True,step=19050,total=6624,reused_generated200=reused,reused_score_blocks=8,protocol_sha256=protocol_sha))

def generate(slot):
    import torch
    g=core.gpu();torch.cuda.set_per_process_memory_fraction(22*1024/g['total_mib'])
    core.generate(slot,4)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','generate','score','merge']);p.add_argument('--slot',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='generate':generate(a.slot)
    elif a.action=='score':native.score(a.slot)
    elif a.action=='merge':native.merge()
    else:
        try:native.run()
        except BaseException:
            save(WORK/'FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
