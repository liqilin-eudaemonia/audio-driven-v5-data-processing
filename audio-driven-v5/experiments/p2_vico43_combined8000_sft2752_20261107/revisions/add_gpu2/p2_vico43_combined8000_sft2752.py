"""ViCo43 controlled planner replacement: combined8000 + latest SFT2752.

Historical generation and checkpoint-bound source stay unchanged. The only
render adapter accepts the audited SFT condition identity instead of base H.
"""
import argparse
import copy
import fcntl
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import contextmanager
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha
import p2_vico256_combined_eval as legacy
import p2_combined_whisper_eval as combined

WORK=ROOT/'experiments/p2_vico43_combined8000_sft2752_20261107'
BASE=ROOT/'experiments/p2_vico256_combined8000_20261106'
SFT=ROOT/'experiments/p2_planning_sft_upright_3b_32gpu_resume_20261106/training'
ADAPTER=SFT/'final_adapter'
ARM='combined8000_sft2752'
NATIVE_CONTEXT=combined.combined_runtime
PY=legacy.PY
PIN=ROOT/'experiments/p2_j_whisper_event_feature_train31_20261105/code_snapshot/dependencies/audio-driven-v5/src/p2_whisper_event_feature/config.py'


@contextmanager
def pinned_context():
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_whisper_event_feature import config as c,training as t
    expected=read(BASE/'protocol.json')['checkpoint_identity']['source']
    live=ROOT/'src/p2_whisper_event_feature/config.py'
    assert sha(PIN)==expected[str(live)]
    def source():
        for path,h in expected.items():assert sha(PIN if Path(path)==live else path)==h,path
        return dict(expected)
    if not getattr(c,'_vico2752_pinned',False):
        exec(compile(PIN.read_bytes(),str(live),'exec'),c.__dict__)
        c.source_identity=t.source_identity=source;t.validate_baseline=c.validate_baseline
        c._vico2752_pinned=True
    with NATIVE_CONTEXT():yield


def prepare(work):
    if (work/'protocol.json').exists():
        p=read(work/'protocol.json')
        for f,h in p['execution_code'].items():assert sha(f)==h,f
        return p
    work.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(2)
    with pinned_context():
        from p2_testsets_eval.runtime import data_config
        from p2_whisper_event_feature.config import source_identity
        from p2_j_sft_hidden import adapter_identity
        d,cfg=data_config();parent=read(BASE/'protocol.json')
        assert parent['checkpoint_identity']['source']==source_identity()
        assert read(SFT/'COMPLETE.json')['success'] and read(SFT/'COMPLETE.json')['global_step']==2752
        assert read(SFT.parent/'COMPLETE.json')['success']
        adapter=adapter_identity(ADAPTER)
        assert Path(read(ADAPTER/'adapter_config.json')['base_model_name_or_path']).resolve()==Path(cfg.omni_path).resolve()
        items=read(BASE/'shards/00.json');assert len(items)==len({x['id'] for x in items})==43
        bins=[[],[],[]];loads=[0,0,0]
        for item in sorted(items,key=lambda x:-x['geometry']['native_frames']):
            i=min(range(3),key=lambda j:loads[j]);bins[i].append(item);loads[i]+=item['geometry']['native_frames']
        for i,rows in enumerate(bins):save(work/'render_shards'/f'{i:02d}.json',rows)
        sources={}
        for x in items:
            src=BASE/'conditions'/x['key'];dst=work/'conditions'/x['key'];dst.mkdir(parents=True,exist_ok=True)
            for name in ['first_frame.png','input_audit.json']:
                combined.link(src/name,dst/name)
            # Real source-only waveform verification, not just a metadata copy.
            _,audit=legacy.source_wave(x,dst)
            sources[x['id']]=dict(image_sha256=sha(dst/'first_frame.png'),**audit)
        combined.link(BASE/'whisper',work/'whisper')
        combined.link(BASE/'negative/00',work/'negative/00')
        for name in ['INPUT_PREFLIGHT.json','shards/00.json']:
            (work/name).parent.mkdir(parents=True,exist_ok=True)
            (work/name).write_bytes((BASE/name).read_bytes())
        session=work/'adapter_session'
        save(session/'protocol.json',dict(schema='latest-sft-inference-session-v1',adapter=adapter,data_config=d,
            training_complete=str(SFT/'COMPLETE.json'),training_complete_sha256=sha(SFT/'COMPLETE.json'),global_step=2752))
        p=copy.deepcopy(parent);p.update(schema='vico43-combined8000-latest-sft2752-v1',created_at=time.time(),
            planner='SFT_Qwen2.5-Omni-3B_upright2752',planner_adapter=str(ADAPTER),planner_adapter_identity=adapter,
            sft_hidden_cache=str(session),planner_batch=4,text_batch=8,
            training_qwen_hidden_source='base_pre_sft',inference_qwen_hidden_source='SFT2752',
            comparison='Fixed combined8000 weights/input/Whisper/sampling; replace Qwen planner and its H/Gemma conditions only',
            authorized_scope='node11 GPUs7/0/1, one worker per GPU with >=88000MiB free; no eviction',worker_cli=str(SELF),
            render_gpus=[7,0,1],render_shards=3,
            source_base_protocol=str(BASE/'protocol.json'),source_base_protocol_sha256=sha(BASE/'protocol.json'))
        p['arms'][0].update(name=ARM,planner=p['planner'])
        files=[SELF,Path(legacy.__file__),Path(combined.__file__),ROOT/'src/p2_testsets_eval/runtime.py',
            ROOT/'src/p2_testsets_eval/whisper_runtime.py',ROOT/'src/p2_j_sft_hidden.py',
            ROOT/'src/p2_ltx6000_sft_eval_20261107.py',ROOT/'src/p2_plan_cache/encoding.py',
            ROOT/'src/p2_planning_sft/data.py',ROOT/'scripts/p2_vico_official_metrics.py',PIN]
        p['execution_code']={str(f):sha(f) for f in files}
        save(work/'protocol.json',p);save(work/'SOURCE_INPUTS.json',sources)
        save(work/'CPU_ACCEPTANCE.json',dict(passed=True,inputs=43,source_only=True,step=8000,planner_step=2752,
            same_checkpoint=parent['arms'][0]['checkpoint'],same_sampling=p['generation'],same_Whisper=True,
            fresh_Qwen_H_and_Gemma=True,adapter=adapter))
        save(work/'PINNED_EXECUTION.json',dict(executed_config=str(PIN),executed_sha256=sha(PIN),
            workspace_sha256=sha(ROOT/'src/p2_whisper_event_feature/config.py'),checkpoint_source_exact=True,
            workspace_preserved=True,render_shard_frames=loads))
        save(work/'status.json',dict(stage='prepared',total=43,at=time.time()))
        return p


@contextmanager
def sft_runtime():
    with pinned_context():
        from p2_ltx6000_sft_eval_20261107 import SFTInferenceSession
        from p2_testsets_eval import whisper_runtime as wr
        original=wr.render
        source=inspect.getsource(original)
        needle="plan['adapter'] is None"
        assert source.count(needle)==1
        source=source.replace(needle,"plan['adapter'] == p['planner_adapter_identity']")
        needle="step=arm['step'],seed=p['seed'],geometry=g,planning_sha256="
        assert source.count(needle)==1
        source=source.replace(needle,"step=arm['step'],seed=p['seed'],geometry=g,planning_adapter=p['planner_adapter_identity'],planning_sha256=")
        namespace=dict(wr.__dict__);exec(compile(source,str(SELF)+':SFT_identity_adapter','exec'),namespace)
        with patch.object(wr,'render',namespace['render']),patch('p2_j_sft_hidden.Reader',SFTInferenceSession):
            yield


def worker(work,stage,shard=0):
    legacy.ARM=ARM
    namespace=dict(legacy.__dict__)
    if stage=='render':
        def scoped_read(path):
            if Path(path)==work/'shards/00.json':return read(work/'render_shards'/f'{shard:02d}.json')
            return read(path)
        namespace['read']=scoped_read
        export=inspect.getsource(legacy.render_export)
        # Disjoint fixed shards: preserve all sampling/export math, validate only
        # this worker's outputs and never announce global completion early.
        export=export.replace('if done==2:','if False:')
        old="if len(list((work/'conversion').glob('*.verified.json')))!=43:raise ValueError('Incomplete scored media')"
        assert export.count(old)==1
        export=export.replace(old,"if any(not (work/'conversion'/f'{x[\"id\"]}.verified.json').exists() for x in items):raise ValueError('Incomplete shard media')")
        export=export.replace("work/'GENERATION_COMPLETE.json'",f"work/'GENERATION_SHARD_{shard:02d}.json'")
        export=export.replace('dict(outputs=43,at=time.time()', 'dict(outputs=len(items),at=time.time()')
        exec(compile(export,str(SELF)+':disjoint_export','exec'),namespace)
    source=inspect.getsource(legacy.worker)
    old="if os.environ.get('CUDA_VISIBLE_DEVICES')!='7' or torch.cuda.device_count()!=1:raise ValueError('Expected only physical GPU7')"
    assert source.count(old)==1
    source=source.replace(old,"if int(os.environ.get('CUDA_VISIBLE_DEVICES','-1')) not in p['render_gpus'] or torch.cuda.device_count()!=1:raise ValueError('Unexpected physical GPU')")
    exec(compile(source,str(SELF)+':worker_scope','exec'),namespace)
    with (work/f'{stage}_{shard:02d}.lock').open('a') as lock,patch.object(combined,'combined_runtime',sft_runtime):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['worker'](work,stage)


def metric(work,child=None,index=None):
    import p2_vico_official_metrics as m
    m.ARM=ARM;m.__file__=str(SELF)
    if child:return m.child(work,child,index)
    m.run(work)
    r=read(work/'RESULTS.json');r.update(model='Combined8000 + Qwen SFT2752',planner_step=2752,
        planner_adapter_identity=read(work/'protocol.json')['planner_adapter_identity'])
    save(work/'RESULTS.json',r)


def run(work):
    p=prepare(work)
    with (work/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(work/'RUNTIME.json',dict(pid=os.getpid(),host=os.uname().nodename,at=time.time()))
        for stage in ['conditions','render']:
            marker=work/(stage.upper()+'_COMPLETE.json')
            if marker.exists():continue
            save(work/'status.json',dict(stage=stage,total=43,at=time.time()))
            if stage=='conditions':
                with (work/(stage+'.log')).open('a') as log:
                    subprocess.run([str(PY),str(SELF),'worker','--work',str(work),'--stage',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            else:
                pending=list(range(3));active={};handles=[]
                try:
                    while pending or active:
                        for gpu,(proc,sh) in list(active.items()):
                            code=proc.poll()
                            if code is not None:
                                if code:raise RuntimeError(f'Render shard{sh} failed: {code}')
                                del active[gpu]
                        for gpu in p['render_gpus']:
                            if not pending or gpu in active:continue
                            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                            if free<88000:continue
                            sh=pending.pop(0);log=(work/f'render_{sh:02d}.log').open('a');handles.append(log)
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
                            proc=subprocess.Popen([str(PY),str(SELF),'worker','--work',str(work),'--stage','render','--shard',str(sh)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                            active[gpu]=(proc,sh)
                        save(work/'RENDER_RUNTIME.json',dict(pending=pending,active={str(g):dict(pid=q.pid,shard=s) for g,(q,s) in active.items()},at=time.time()))
                        if active or pending:time.sleep(5)
                    assert len(list((work/'conversion').glob('*.verified.json')))==43
                    save(work/'GENERATION_COMPLETE.json',dict(outputs=43,at=time.time()))
                finally:
                    for proc,sh in active.values():
                        if proc.poll() is None:proc.terminate();proc.wait()
                    for h in handles:h.close()
            save(marker,dict(at=time.time()))
        save(work/'status.json',dict(stage='official_metrics',total=43,at=time.time()))
        with (work/'metrics.log').open('a') as log:
            subprocess.run([str(legacy.MPY),str(SELF),'metrics','--work',str(work)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        save(work/'GENERATION_AND_IMAGE_COMPLETE.json',dict(outputs=43,at=time.time(),results_sha256=sha(work/'RESULTS.json')))
        save(work/'status.json',dict(stage='image_complete_motion_and_review_pending',total=43,at=time.time()))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',nargs='?',default='metrics',choices=['prepare','worker','run','metrics'])
    a.add_argument('--work',type=Path,default=WORK);a.add_argument('--stage');a.add_argument('--child');a.add_argument('--index',type=int);a.add_argument('--shard',type=int,default=0)
    args=a.parse_args();work=args.work.resolve()
    if args.child:metric(work,args.child,args.index)
    elif args.action=='prepare':prepare(work)
    elif args.action=='worker':worker(work,args.stage,args.shard)
    elif args.action=='run':run(work)
    else:metric(work)
