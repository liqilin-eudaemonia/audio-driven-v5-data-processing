"""Resume the sealed Seamless200 evaluation on node11 GPUs4-7.

Reuse the established four-GPU scheduler and the frozen followup sampler/model
loader. Only physical placement changes; protocol and completed outputs stay put.
"""
import argparse
import fcntl
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha
import p2_all_lora_followups_eval_20261109 as E
import p2_all_lora5000_seamless_eval4_20261109 as Q

E.configure(7500)
WORK=E.WORK
CHAIN=ROOT/'experiments/p2_ltx_all_lora_followups_gpu7_20261109'
RUNTIME='RUNTIME_GPU4_20261110.json'


def check_runtime(work):
    assert work.resolve()==WORK.resolve()
    r=read(work/RUNTIME)
    assert r['node']=='node11' and r['physical_gpus']==[4,5,6,7]
    assert r['script']==str(SELF) and r['script_sha256']==sha(SELF)
    assert r['scheduler_sha256']==sha(Q.__file__)
    assert r['protocol_sha256']==sha(work/'protocol.json')
    assert all(sha(p)==digest for p,digest in read(work/'protocol.json')['execution_code'].items())


def prepare(work):
    check_runtime(work)
    return E.prepare(work)


def final(work):
    E.final(work)
    a=read(work/'FINAL_AUDIT.json')
    a.update(world=4,physical_gpus=[4,5,6,7],runtime_overlay=str(work/RUNTIME))
    save(work/'FINAL_AUDIT.json',a)


def worker(work,stage,rank,slot=0):
    check_runtime(work)
    assert 0<=rank<4 and 0<=slot<4
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(rank+4)
    p=read(work/'protocol.json');node=read(work/'cluster.json')['nodes'][0]
    assert node['host'] in os.popen('hostname -I').read().split()
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    import torch
    torch.set_num_threads(4);torch.cuda.set_device(0)
    from p2_listen.config import extractor_config
    from p2_testsets_eval import runtime as rt
    from p2_testsets_eval.shared_items import all_items
    d=read(work/'DATA_CONFIG.json');cfg=extractor_config(d)
    f=work/'locks'/f'{stage}_{rank:02d}_{slot}.lock';f.parent.mkdir(exist_ok=True)
    with f.open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(work/'devices'/f'{stage}_{rank:02d}_{slot}.json',dict(pid=os.getpid(),at=time.time(),
             host=socket.gethostname(),rank=rank,physical_gpu=rank+4,stage=stage))
        old_progress=rt.progress
        def progress(w,s,r,**kw):return old_progress(w,s,rank+4*slot,**kw)
        with patch.object(rt,'data_config',lambda:(d,cfg)),patch.object(rt,'progress',progress):
            if stage=='render':
                import p2_ltx6000_latest_sft_dual200_20261107 as native
                import p2_ltx_events.inference as previous
                def loading(path,device='cuda:0'):
                    result=E.load_model(work,path,device)
                    save(work/'model_acceptance'/f'{rank:02d}.json',dict(success=True,at=time.time(),rank=rank,
                         physical_gpu=rank+4,strict_all_lora_load=True,targets=len(result[0].text_targets)))
                    return result
                with patch.object(native,'ARM',E.ARM),patch.object(previous,'load_model',loading):
                    # Rank00 negative was verified bitwise equal across original ranks.
                    E.render(work,all_items(work,[],p),0,p)
            else:rt.decode(work,all_items(work,[],p),rank,p,p['arms'])
        progress(work,stage,0,status='complete',slot=slot)


def queue(work):
    check_runtime(work)
    with (CHAIN/'queue.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(CHAIN/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),script=str(SELF),physical_gpus=[4,5,6,7]))
        save(CHAIN/'status.json',dict(stage='evaluation',step=7500,at=time.time(),physical_gpus=[4,5,6,7],
             driver=str(SELF),live_progress=str(work/'status.json')))
        # The proven scheduler releases each completed render GPU into decode
        # and scoring. Its shared sample claims skip all committed prior outputs.
        with patch.object(Q,'SELF',SELF),patch.object(Q,'E',SimpleNamespace(ARM=E.ARM,prepare=prepare,final=final)):
            Q.queue(work)
        if read(work/'status.json')['stage']=='complete':
            save(CHAIN/'COMPLETE_7500.json',dict(success=True,at=time.time(),step=7500,outputs=200,work=str(work)))
            assert all(read(CHAIN/f'COMPLETE_{s}.json')['success'] for s in (5500,7000,7500))
            save(CHAIN/'status.json',dict(stage='complete',at=time.time(),steps=[5500,7000,7500],outputs=600,physical_gpus=[4,5,6,7]))
            save(CHAIN/'HANDOFF_COMPLETE.json',dict(success=True,at=time.time(),steps=[5500,7000,7500],outputs=600))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['queue','worker','check'])
    parser.add_argument('--work',type=Path,default=WORK);parser.add_argument('--stage',choices=['render','decode'])
    parser.add_argument('--rank',type=int);parser.add_argument('--slot',type=int,default=0);a=parser.parse_args()
    try:
        if a.action=='worker':worker(a.work.resolve(),a.stage,a.rank,a.slot)
        elif a.action=='check':prepare(a.work.resolve());print('sealed conditions and four-GPU runtime accepted')
        else:queue(a.work.resolve())
    except Exception as exc:
        save(a.work/'GPU4_FAILURE_20261110.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()))
        raise
