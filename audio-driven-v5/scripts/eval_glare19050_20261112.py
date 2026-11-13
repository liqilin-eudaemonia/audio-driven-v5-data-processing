"""Checkpoint19050 comparison; reuse verified GLARE generation and scorers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
import eval_glare_final_20261112 as core
old_complete=core.all_generated
import eval_glare_seamless_test_20261112 as seamless
from p2_testsets_eval.protocol import read,sha,rows,METRIC_PYTHON

WORK=ROOT/'experiments/glare19050_vico43_responsenet586_seamless200_20261112'
PARENT=ROOT/'experiments/glare_final_vico43_responsenet586_20261112'
FULL=ROOT/'experiments/glare_final_seamless_test6624_20261112'
SELF=Path(__file__).resolve()
CHECKPOINT=ROOT/'experiments/glare_paper_settings_node11_gpu7_20261111/before_extension30epochs.pt'
core.WORK=WORK;core.SELF=SELF;core.CHECKPOINT=CHECKPOINT;core.ARM='glare_step19050';core.MODEL='GLARE step19050';core.all_generated=old_complete
seamless.WORK=WORK
save=core.save

def link(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.symlink_to(source,target_is_directory=source.is_dir())

def prepare():
    import numpy as np
    import torch
    WORK.mkdir(exist_ok=True)
    if (WORK/'protocol.json').exists():
        p=read(WORK/'protocol.json');assert sha(CHECKPOINT)==p['checkpoint_sha256'];return p
    parent=read(PARENT/'protocol.json');full=read(FULL/'protocol.json')
    assert read(PARENT/'COMPLETE.json')['passed']
    weights=torch.load(CHECKPOINT,map_location='cpu',weights_only=False,mmap=True);assert weights['step']==19050
    identity=weights['identity'];del weights
    samples=copy.deepcopy(parent['samples'])
    for s in full['samples'][:200]:samples.append(read(s['ready'])['sample'])
    assert len(samples)==len({s['key'] for s in samples})==829
    _,cacheid=core.prosody_id();reused=[]
    for s in samples:
        assert sha(s['audio'])==s['audio_sha256'] and sha(s['reference'])==s['reference_sha256']
        basename=core.cache_path(s,cacheid).name;source=(FULL if s['dataset']=='seamless' else PARENT)/'prosody'/basename
        assert source.exists(),source
        target=WORK/'prosody'/basename
        if target.exists():
            with np.load(target) as a,np.load(source) as b:assert np.array_equal(a['intensity'],b['intensity'])
        else:link(source,target)
        reused.append(dict(key=s['key'],source=str(source),source_sha256=sha(source)))
    p=copy.deepcopy(parent)
    p.update(schema='GLARE19050-same-ViCo43-ResponseNet586-Seamless200-GPU4-v1',created=time.time(),step=19050,checkpoint=str(CHECKPOINT),
        checkpoint_sha256=sha(CHECKPOINT),model=core.MODEL,samples=samples,total=829,metric_shards=16,
        metrics_decoded_gt_cache=full['metrics_decoded_gt_cache'],quick200_ids=full['quick200_ids'],
        parent_final_protocol_sha256=sha(PARENT/'protocol.json'),parent_seamless_protocol_sha256=sha(FULL/'protocol.json'),
        checkpoint_identity=identity,decode_batch=4,
        acceleration='all829 inputs/intensity caches reused; one22GiB-bounded resident generator, decode4; existing GT features/scorers; capacity based admission on shared GPU4',
        comparison='same seed137/nativeEuler nfe10/speakerCFG2/inputs/prosody/GT/score formulas as final32940; FP32 LIA decoder batch4 vs8 may have ordinary batching roundoff',
        code={str(f):sha(f) for f in [SELF,ROOT/'scripts/eval_glare_final_20261112.py',ROOT/'scripts/eval_glare_seamless_test_20261112.py',
             ROOT/'scripts/glare_resident_inference.py',ROOT/'scripts/infer_glare_checkpoint.py']})
    save(WORK/'protocol.json',p);save(WORK/'REUSED_CONDITIONS.json',dict(clips=829,records=reused))
    save(WORK/'PROSODY_COMPLETE.json',dict(clips=829,reused=True))
    v=WORK/'vico';v.mkdir(exist_ok=True);vp=read(PARENT/'vico/protocol.json')
    vp.update(model=core.MODEL,arms=[dict(name=core.ARM,step=19050,checkpoint=str(CHECKPOINT),original_sha256=p['checkpoint_sha256'])])
    save(v/'protocol.json',vp)
    for name in ['INPUT_PREFLIGHT.json','shards/00.json']:
        dest=v/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/'vico'/name,dest)
    ss=[s for s in samples if s['dataset']=='seamless']
    for i in range(8):save(WORK/'blocks'/f'{i:03d}.json',ss[i*25:(i+1)*25])
    save(WORK/'PRECHECK.json',dict(passed=True,step=19050,vico=43,responsenet=586,seamless=200,all_conditions_reused=True,checkpoint_sha256=p['checkpoint_sha256']))
    return p

def generate():
    import torch
    g=core.gpu();torch.cuda.set_per_process_memory_fraction(22*1024/g['total_mib'])
    core.generate(0,4)

def seamless_merge():
    import numpy as np
    from evaluation.video.paired_metrics.run import summarize,METRICS
    p=read(WORK/'protocol.json');actual=[];contracts=[];dist={k:[[],[]] for k in ['fid','fvd']};coverage={k:0 for k in dist}
    for i in range(8):
        f=WORK/'metrics'/f'{i:03d}';assert (f/'SUCCESS.json').exists();summary=read(f/'summary.json');contracts.append(read(f/'contract.json'));actual.extend(rows(f/'pairs.jsonl'))
        with np.load(f/'features.npz') as features:
            for k in dist:
                if features[k+'_gt'].size:dist[k][0].append(features[k+'_gt']);dist[k][1].append(features[k+'_generated'])
                coverage[k]+=summary['coverage'][k]['pairs']
    assert len(actual)==len({x['id'] for x in actual})==200 and {x['id'] for x in actual}==set(p['quick200_ids'])
    for c in contracts:assert c['code']==contracts[0]['code'] and c['assets']==contracts[0]['assets']
    result=summarize(actual,dist,set(METRICS),contracts[0]['schema'],contracts[0]['datasets'],200,{})
    for k,n in coverage.items():result['coverage'][k]['pairs']=n
    result.update(model=core.MODEL,checkpoint_sha256=p['checkpoint_sha256'],dataset='Seamless fixed200',audio_metrics=p['audio_metrics'],aggregation='global200 features, not mean shard scores')
    save(WORK/'seamless/RESULTS.json',result)

def update_comparison():
    try:
        from render_glare_checkpoint_comparison import render
        render()
    except Exception:
        save(WORK/'DOC_UPDATE_FAILURE.json',dict(traceback=traceback.format_exc(),at=time.time()))

def finish():
    p=read(WORK/'protocol.json')
    if not (WORK/'MEDIA_AUDIT_COMPLETE.json').exists():core.media_audit()
    proof=read(WORK/'MEDIA_AUDIT_COMPLETE.json');assert proof['passed'] and proof['clips']==829 and proof['protocol_sha256']==sha(WORK/'protocol.json')
    result=dict(model=core.MODEL,step=19050,checkpoint=str(CHECKPOINT),checkpoint_sha256=p['checkpoint_sha256'],
       vico_image=read(WORK/'vico/IMAGE_RESULTS.json'),vico_motion=read(WORK/'vico/bfm/RESULTS.json'),vico_DIM=read(WORK/'vico/DIM_RESULTS.json'),
       responsenet=read(WORK/'responsenet/RESULTS.json'),seamless200=read(WORK/'seamless/RESULTS.json'))
    assert result['vico_image']['clips']==result['vico_DIM']['clips']==43 and result['responsenet']['requested_pairs']==586 and result['seamless200']['requested_pairs']==200
    save(WORK/'RESULTS.json',result);save(WORK/'COMPLETE.json',dict(passed=True,clips=829,at=time.time(),results_sha256=sha(WORK/'RESULTS.json')))
    save(WORK/'status.json',dict(stage='complete',clips=829,at=time.time()))

class ExistingGenerator:
    def __init__(self,pid,samples):self.pid=pid;self.samples=samples;self.returncode=None
    def poll(self):
        if self.returncode is not None:return self.returncode
        path=Path(f'/proc/{self.pid}/stat')
        if path.exists() and path.read_text().split(') ',1)[1].split()[0]!='Z':return None
        self.returncode=0 if all((WORK/s['dataset']/'generated'/s['id']/'complete.json').exists() for s in self.samples) else 1
        return self.returncode
    def terminate(self):
        if self.poll() is None:os.kill(self.pid,signal.SIGTERM)

def score_ready(action,index,p):
    if action=='response-worker':
        selected=[s for s in p['samples'] if s['dataset']=='responsenet'];n=math.ceil(len(selected)/16);selected=selected[index*n:(index+1)*n]
    else:selected=read(WORK/'blocks'/f'{index:03d}.json')
    return all((WORK/s['dataset']/'generated'/s['id']/'complete.json').exists() for s in selected)

def run():
    core.gpu();p=read(WORK/'protocol.json');started=time.time()
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(WORK/'LAUNCH.json',dict(pid=os.getpid(),gpu=4,started=started,policy='one resident generator; at most2 additional feature/scoring workers; preserve ongoing final6624'))
        active={};done=set();pending_motion=[(kind,i) for kind in ['bfm','emoca'] for i in range(4)];scoring=[]
        started_gen=False;started_image=False;prepared_motion=False;motion_scored=False;started_audit=False;scoring_initialized=False
        previous=WORK/'processes/generate.json'
        if previous.exists():
            old=read(previous);cmd=Path(f'/proc/{old["pid"]}/cmdline')
            if cmd.exists() and str(SELF).encode() in cmd.read_bytes() and b'generate' in cmd.read_bytes():
                proc=ExistingGenerator(old['pid'],p['samples'])
                if proc.poll() is None:active['generate']=proc;started_gen=True
        try:
            while True:
                for tag,proc in list(active.items()):
                    code=proc.poll()
                    if code is None:continue
                    del active[tag]
                    if code:raise RuntimeError(f'{tag} failed: logs/{tag}.log')
                    done.add(tag)
                counts={d:len(list((WORK/d/'generated').glob('*/complete.json'))) for d in ['vico','responsenet','seamless']}
                g=core.gpu()
                if not started_gen and sum(counts.values())==829:started_gen=True
                if not started_gen and g['free_mib']>=30*1024:
                    active['generate']=core.launch('generate','generate',core.GENPY,'gen');started_gen=True
                if counts['vico']==43 and not started_image and g['free_mib']>=24*1024:
                    save(WORK/'vico/GENERATION_COMPLETE.json',dict(outputs=43));active['image']=core.launch('image','image',core.IMAGEPY);started_image=True
                if 'image' in done and not prepared_motion:
                    if not (WORK/'vico/METRICS_PROTOCOL.json').exists():core.prepare_motion()
                    else:core.motion_template().prepare(WORK/'vico')
                    pending_motion=[(kind,i) for kind,i in pending_motion if not
                        (WORK/'vico/bfm'/f'EXTRACTION_COMPLETE_{i:02d}.json' if kind=='bfm' else
                         WORK/'vico/emoca'/f'shard_{i:02d}/EXTRACTION_COMPLETE.json').exists()]
                    prepared_motion=True
                motion_active=[t for t in active if t.startswith('feature_')]
                limit=1 if 'generate' in active else 2
                if prepared_motion and pending_motion and len(motion_active)<limit and g['free_mib']>=24*1024:
                    kind,i=pending_motion.pop(0);tag=f'feature_{kind}_{i:02d}'
                    active[tag]=core.launch('motion-worker',tag,core.BFMPY if kind=='bfm' else core.MOREPY,extra=['--kind',kind,'--slot',str(i)])
                features_done=prepared_motion and not pending_motion and not any(t.startswith('feature_') for t in active)
                if features_done and not motion_scored:
                    if not (WORK/'vico/SCORING_ENVIRONMENT.json').exists():active['motion_score']=core.launch('motion-score','motion_score',core.MOREPY)
                    else:done.add('motion_score')
                    motion_scored=True
                if sum(counts.values())==829 and not (WORK/'GENERATION_COMPLETE.json').exists():save(WORK/'GENERATION_COMPLETE.json',dict(outputs=829))
                if not started_audit and (WORK/'MEDIA_AUDIT_COMPLETE.json').exists():started_audit=True
                if sum(counts.values())==829 and not started_audit:
                    active['media_audit']=core.launch('media-audit','media_audit',core.MOREPY);started_audit=True
                if 'motion_score' in done and not scoring_initialized:
                    scoring=[('response-worker',i) for i in range(16) if not (WORK/'responsenet/metric_shards'/f'{i:02d}/SUCCESS.json').exists()]
                    scoring += [('score',i) for i in range(8) if not (WORK/'metrics'/f'{i:03d}/SUCCESS.json').exists()]
                    scoring_initialized=True
                scored_active=[t for t in active if t.startswith('score_')]
                if scoring and len(scored_active)<int(os.environ.get('GLARE19050_SCORE_WORKERS','8')) and g['free_mib']>=22*1024:
                    candidate=next((job for job in scoring if score_ready(*job,p)),None)
                    if candidate is not None:
                        scoring.remove(candidate);action,i=candidate;tag=f'score_{action}_{i:02d}'
                        active[tag]=core.launch(action,tag,METRIC_PYTHON,'response',['--slot',str(i)])
                if motion_scored and 'motion_score' in done and not scoring and scoring_initialized and not active and sum(counts.values())==829:
                    for action in ['response-merge','seamless-merge']:
                        proc=core.launch(action,action,METRIC_PYTHON,'response');assert proc.wait()==0,action
                    finish();return
                save(WORK/'status.json',dict(stage='running',step=19050,counts=counts,total=829,active=list(active),pending_feature_shards=len(pending_motion),pending_score_shards=len(scoring),gpu=g,elapsed_s=time.time()-started,at=time.time()))
                time.sleep(5)
        except BaseException:
            for proc in active.values():
                if proc.poll() is None:proc.terminate()
            save(WORK/'FAILURE.json',dict(traceback=traceback.format_exc(),at=time.time()));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',nargs='?',default='run');parser.add_argument('--work',type=Path);parser.add_argument('--slot',type=int,default=0);parser.add_argument('--kind');parser.add_argument('--child');parser.add_argument('--index',type=int)
    a=parser.parse_args()
    if a.child:core.image_metric(a.child,a.index)
    elif a.action=='prepare':prepare()
    elif a.action=='generate':generate()
    elif a.action=='image':core.image_metric()
    elif a.action=='motion-worker':core.motion_worker(a.kind,a.slot)
    elif a.action=='motion-score':core.motion_score()
    elif a.action=='response-worker':core.response_worker(a.slot)
    elif a.action=='response-merge':core.response_merge();update_comparison()
    elif a.action=='score':seamless.score(a.slot)
    elif a.action=='seamless-merge':seamless_merge();update_comparison()
    elif a.action=='media-audit':core.media_audit()
    elif a.action=='finish':finish()
    elif a.action=='run':run()
    else:raise ValueError(a.action)
