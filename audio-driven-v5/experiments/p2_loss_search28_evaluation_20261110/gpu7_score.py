"""Sequential scoring on node11 GPU7, alongside the independent 28-GPU training."""
from pathlib import Path
import argparse
import copy
import fcntl
import inspect
import os
import socket
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
from types import SimpleNamespace

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse'),str(WORK)]
from p2_listen.contracts import read,sha,atomic_json as save
from p2_testsets_eval.protocol import PYTHON,METRIC_PYTHON
import eval_runtime as E

HOST='172.29.189.119'
UUID='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'


def configure(work):
    p=E.prepare(work)
    assignment=dict(host=HOST,node='node11',physical_gpu=7,uuid=UUID)
    if 'scoring_device' in p:
        if p['scoring_device']!=assignment or p['workers']!=4:
            raise ValueError('Scoring assignment changed')
        return p
    if (work/'GENERATION_COMPLETE.json').exists() or (work/'SCORING_COMPLETE.json').exists():
        raise ValueError('Cannot repartition completed evaluation')
    p.update(workers=4,scoring_device=assignment,
             authorized_scope='node11 GPU7 only; can run alongside 28-GPU training')
    cluster=read(work/'cluster.json')
    node=copy.deepcopy(next(n for n in cluster['nodes']if n['host']==HOST));node['gpus']=[7]
    cluster['nodes']=[node];save(work/'cluster.json',cluster)
    from p2_testsets_eval.protocol import partition
    for rank,items in enumerate(partition(read(work/'shared_items.json'),4)):
        save(work/'shards'/f'{rank:02d}.json',items)
    save(work/'protocol.json',p)
    return p


def validate_device(p):
    if p.get('scoring_device')!=dict(host=HOST,node='node11',physical_gpu=7,uuid=UUID):
        raise ValueError('Unapproved scoring assignment')
    if HOST not in subprocess.check_output(['hostname','-I'],text=True).split():
        raise ValueError('Scoring must run on node11')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='7':
        raise ValueError('Scoring must expose only physical GPU7')
    actual=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if actual!=UUID or p['execution_code']!=E.code_identity():
        raise ValueError('GPU UUID or execution code changed')


def worker(work,stage,slot=0):
    p=read(work/'protocol.json');validate_device(p)
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    import torch
    torch.set_num_threads(4);torch.cuda.set_device(0)
    # Bound our allocator while preserving the existing GPU7 services.
    torch.cuda.set_per_process_memory_fraction(.53,0)
    from p2_listen.config import extractor_config
    from p2_testsets_eval import runtime as rt
    import p2_ltx_events.inference as previous
    d=read(work/'DATA_CONFIG.json');cfg=extractor_config(d)
    E.ARM=p['arms'][0]['name'];items=read(work/'shared_items.json')
    folder=work/'locks';folder.mkdir(exist_ok=True)
    with (folder/f'gpu7_{stage}_{slot}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(work/'devices'/f'{stage}_gpu7_{slot}.json',dict(pid=os.getpid(),host=socket.gethostname(),
             physical_gpu=7,uuid=UUID,stage=stage,at=time.time()))
        old_progress=rt.progress
        def progress(w,s,r,**kw):return old_progress(w,s,4*slot,**kw)
        with patch.object(rt,'data_config',lambda:(d,cfg)),patch.object(rt,'progress',progress),\
             patch.object(previous,'load_model',lambda path,device='cuda:0':E.load_model(path,work,device)):
            if stage=='render':E.render(work,items,0,p)
            elif stage=='decode':rt.decode(work,items,4*slot,p,p['arms'])
            else:raise ValueError(stage)
        rt.progress(work,stage,0,status='complete',samples=len(items))


def remote(work,stage):
    from listen_cluster import transport
    from stage_driver import command
    cl=read(work/'cluster.json');node=next(n for n in cl['nodes']if n['host']==HOST)
    env=['env','CUDA_VISIBLE_DEVICES=7','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=1',
         'TOKENIZERS_PARALLELISM=false','PYTORCH_ALLOC_CONF=expandable_segments:True',
         'PYTHONPATH=/private/liqi_workspace/.cache/paired_video_metrics/python']
    if stage in ('render','decode'):
        cmd=[str(PYTHON),'-u',str(Path(__file__)),'worker','--work',str(work),'--stage',stage]
    elif stage=='events':
        cmd=[str(PYTHON),'-u',str(WORK/'run_events.py'),'--work',str(work)]
    elif stage=='metric':
        cmd=[str(METRIC_PYTHON),'-u',str(ROOT/'scripts/p2_pipelined_testsets_eval.py'),
             'metrics','--work',str(work),'--rank','0','--arms',read(work/'protocol.json')['arms'][0]['name']]
    else:raise ValueError(stage)
    guard=ROOT.parent/'audio-driven-v4-p3-autoselect-20261013/scripts/pipeline3_cluster.py'
    argv=transport(cl,node,[str(PYTHON),str(guard),'--guard',*env,*cmd])
    from p2_loss_search28.config import WORK as TRAIN
    command(argv,work/f'gpu7_{stage}.log',[work/'STOP',WORK/'STOP',TRAIN/'STOP'],stdin_guard=True)


def progressive_queue(work):
    """Reuse the measured GPU7 scheduler; adapt only population and model entry."""
    import p2_all_lora_followups_gpu7_20261109 as native
    from binder import ShardBinder
    p=read(work/'protocol.json')
    source=inspect.getsource(native.queue)
    if source.count('200')!=5:
        raise ValueError('Native scheduler changed; review population adaptation')
    source=source.replace('200',"p['total_samples']")
    def check(work):
        current=read(work/'protocol.json')
        if current['execution_code']!=E.code_identity():raise ValueError('Evaluation code changed')
        if read(work/'cluster.json')['nodes'][0]['host']!=HOST:raise ValueError('Wrong scheduling host')
    adapter=SimpleNamespace(ARM=p['arms'][0]['name'],STEP=p['arms'][0]['step'],
                            prepare=configure,final=E.final)
    with patch.object(native,'SELF',Path(__file__)),patch.object(native,'E',adapter):
        namespace=dict(native.__dict__,check_runtime=check,Binder=ShardBinder)
        exec(compile(source,str(Path(__file__))+'::native_gpu7_queue','exec'),namespace)
        namespace['queue'](work)


def score_one(work):
    from stage_driver import verify_scoring
    if (work/'SCORING_COMPLETE.json').exists():return verify_scoring(work)
    p=configure(work)
    if not (work/'FINAL_AUDIT.json').exists():
        progressive_queue(work)
    if not (work/'EVENTS_COMPLETE.json').exists():remote(work,'events')
    from reference_events import freeze
    freeze(work)
    event_done=read(work/'EVENTS_COMPLETE.json')
    if event_done['summary_sha256']!=sha(work/'events/summary.json'):
        raise ValueError('Detector summary changed')
    arm=p['arms'][0]['name'];folder='metrics/'+arm+'/seamless/'
    save(work/'SCORING_COMPLETE.json',dict(success=True,at=time.time(),
         checkpoint_sha256=read(work/'export/RECEIPT.json')['original_sha256'],
         evidence={f:sha(work/f)for f in ['FINAL_AUDIT.json','RESULTS.json','REFERENCE_EVENTS_ACCEPTED.json',
                   'event_pairs_fixed_reference.jsonl','fixed_reference_event_summary.json',
                   folder+'pairs.jsonl',folder+'audio_pairs.jsonl']},
         partial_event_coverage=event_done['status']!='complete',detector_human_calibration_pending=True))
    return verify_scoring(work)


def evaluate(target):
    from stage_driver import stage_candidates,checkpoint_receipts_ready
    from p2_loss_search28.config import WORK as TRAIN
    names=stage_candidates(target)
    with (WORK/'stage_driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
        save(WORK/'STAGE_CONTROLLER.json',dict(pid=os.getpid(),start_ticks=ticks,target=target,
             device='node11GPU7',at=time.time()))
        for name in names:
            while not checkpoint_receipts_ready(target,[name]):
                if (WORK/'STOP').exists() or (TRAIN/'STOP').exists():return
                save(WORK/f'STAGE_{target}_STATUS.json',dict(stage='waiting_for_training_checkpoints',
                     target=target,candidate=name,device='node11GPU7',at=time.time()))
                time.sleep(30)
            work=WORK/'evaluations'/f'{name}_{target}'
            if any(p.exists()for p in [WORK/'STOP',TRAIN/'STOP',work/'STOP']):return
            request=dict(checkpoint=str(TRAIN/'runs'/name/f'step_{target:06d}.pt'),samples=64 if target==500 else 200)
            if read(work/'REQUEST.json')!=request:raise ValueError('Evaluation request changed')
            save(WORK/f'STAGE_{target}_STATUS.json',dict(stage='generation_and_scoring',target=target,
                 candidate=name,device='node11GPU7',at=time.time()))
            score_one(work)
        save(WORK/f'STAGE_{target}_STATUS.json',dict(stage='generation_review_ready',target=target,
             candidates=names,device='node11GPU7',at=time.time(),promoted=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','worker'])
    parser.add_argument('--target',type=int,default=500);parser.add_argument('--work',type=Path)
    parser.add_argument('--stage',choices=['render','decode'])
    parser.add_argument('--rank',type=int,choices=[3],default=3)
    parser.add_argument('--slot',type=int,choices=range(4),default=0)
    parser.add_argument('--step',type=int)
    args=parser.parse_args()
    try:
        if args.action=='worker':worker(args.work.resolve(),args.stage,args.slot)
        else:evaluate(args.target)
    except Exception as exc:
        dest=args.work if args.work else WORK
        save(dest/'GPU7_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()))
        raise
