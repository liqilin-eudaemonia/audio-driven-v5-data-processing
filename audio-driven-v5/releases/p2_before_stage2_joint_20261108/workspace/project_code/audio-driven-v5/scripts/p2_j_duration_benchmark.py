"""32-rank ABBA test; retain consumed prefix and resume the selected policy."""
import argparse
import copy
import csv
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,atomic_json,sha
from p2_j_duration_execution import make_pairs,install_backend,Schedule
from p2_j_s1_queue import commands
from listen_cluster import launch


def equal_tensors(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_tensors(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal_tensors(x,y) for x,y in zip(a,b))
    return a==b


def migrate(saved,path,c,source):
    from p2_listen.cache import atomic_torch
    from p2_matrix16.checkpoint import checked
    before=copy.deepcopy(saved['identity']['experiment']);after=copy.deepcopy(c)
    before.pop('output');after.pop('output');after.pop('duration_sampling')
    if before!=after or saved['identity']['world']!=32:
        raise ValueError('Migration may only add sampling and change output')
    if any(source.get(k)!=v for k,v in saved['identity']['source'].items()):
        raise ValueError('Original training implementation changed')
    identity=copy.deepcopy(saved['identity']);identity.update(experiment=c,source=source)
    value=dict(saved,identity=identity);target=Path(c['output'])/path.name
    atomic_torch(target,value);atomic_json(target.with_suffix('.complete.json'),dict(sha256=sha(target),step=saved['step']))
    verify=checked(target)
    for key in ('model','optimizer','rank_rng'):
        if not equal_tensors(saved[key],verify[key]):raise ValueError('Migration changed '+key)
    atomic_json(target.parent/'run.json',identity,immutable=True)
    atomic_json(target.parent/'migration.json',dict(source=str(path),source_sha256=sha(path),step=saved['step'],
        model_optimizer_rng_bitwise_unchanged=True,change='duration schedule added; consumed prefix unchanged'),immutable=True)
    return target


def run(config,cluster,work,new_output):
    import torch
    torch.set_num_threads(4)
    from p2_matrix16.checkpoint import checked
    from p2_clip_events import backend as original_backend
    base=read(config);old=Path(base['output']);work.mkdir(parents=True,exist_ok=True)
    stopped=False;original_path=None;selected_started=False
    def status(stage,**kw):
        value=dict(stage=stage,time=time.time(),**kw);atomic_json(work/'state.json',value);print(json.dumps(value),flush=True)
    def execute(path,checkpoint,stop,tag,aligned=True):
        cc=copy.deepcopy(cluster);cc['master_port']=29770+len(list((work/'attempts').glob('*')))
        cmds=commands(cc,path,resume=checkpoint,stop_after=stop)
        if aligned:
            for cmd in cmds:cmd[cmd.index(str(ROOT/'scripts/p2_clip_events.py'))]=str(ROOT/'scripts/p2_j_duration_train.py')
        launch(cc,cmds,work/'attempts'/tag)
    try:
        if read(old/'queue/state.json')['state']!='training':raise RuntimeError('Original run is not in the training phase')
        native_source=original_backend.source_identity()
        if read(old/'run.json')['source']!=native_source:raise ValueError('Original training source changed')
        before=read(old/'status.json')['step'];status('saving_original',step=before)
        if (old/'STOP').exists():raise RuntimeError('Existing stop request; do not override')
        (old/'STOP').write_text('Temporary pause for authorized duration-alignment benchmark.\n');stopped=True
        began=time.time()
        while read(old/'queue/state.json')['state']!='stopped':
            if time.time()-began>300:raise RuntimeError('Original training did not stop at checkpoint boundary')
            time.sleep(2)
        start=read(old/'status.json')['step'];original_path=old/f'step_{start:06d}.pt';saved=checked(original_path)
        if saved['step']!=start or start<before:raise ValueError('Missing graceful checkpoint')
        atomic_json(work/'ORIGINAL_SAVED.json',dict(step=start,checkpoint=str(original_path),sha256=sha(original_path)),immutable=True)
        backend=install_backend();source=backend.source_identity()
        sample_list=Path(base['j_cache'])/'sample_list.csv'
        with sample_list.open() as stream:rows=[dict(requested_duration_s=float(r['duration_s'])) for r in csv.DictReader(stream)]
        specs={}
        for policy in ('random','duration'):
            pairs=make_pairs(rows,base['training']['steps'],32,base['training']['seed'],start,policy)
            path=work/(policy+'.schedule.json')
            atomic_json(path,dict(sample_list_sha256=sha(sample_list),pairs=pairs),immutable=True)
            specs[policy]=dict(policy=policy,window=512,start_step=start,manifest=str(path),sha256=sha(path),sample_list_sha256=sha(sample_list))
        def prepare(name,policy,output):
            c=copy.deepcopy(base);c.update(output=str(output),duration_sampling=specs[policy])
            path=work/(name+'.json');atomic_json(path,c,immutable=True);Schedule(c)
            pre=read(old/'preflight.json');pre['source']=source
            pre['reused_full_preflight']=dict(path=str(old/'preflight.json'),sha256=sha(old/'preflight.json'),reason='Only sampling order/output changed')
            atomic_json(output/'preflight.json',pre,immutable=True)
            return path,migrate(saved,original_path,c,source)
        probes={}
        for name,policy in [('a1','random'),('b1','duration'),('b2_resume','duration'),('a2','random')]:
            probes[name]=prepare(name,policy,work/name)
        for name,(path,checkpoint) in probes.items():
            status('benchmark',run=name,first_step=start+1,last_step=start+16)
            if name=='b2_resume':
                execute(path,checkpoint,start+8,name+'_first8')
                execute(path,work/name/f'step_{start+8:06d}.pt',start+16,name+'_resume8')
            else:execute(path,checkpoint,start+16,name)
        left=checked(work/'b1'/f'step_{start+16:06d}.pt');right=checked(work/'b2_resume'/f'step_{start+16:06d}.pt')
        equivalence={k:equal_tensors(left[k],right[k]) for k in ('model','optimizer','rank_rng')};del left,right
        timing={}
        for name in probes:
            metrics=[json.loads(line) for line in (work/name/'metrics.jsonl').read_text().splitlines()]
            if [r['step'] for r in metrics]!=list(range(start+1,start+17)):raise ValueError('Probe coverage differs')
            timing[name]=dict(mean_seconds=statistics.mean(r['seconds'] for r in metrics),total_seconds=sum(r['seconds'] for r in metrics),rank0_peak_gib=max(r['peak_reserved_gib'] for r in metrics))
        a=statistics.mean(timing[n]['mean_seconds'] for n in ('a1','a2'));b=statistics.mean(timing[n]['mean_seconds'] for n in ('b1','b2_resume'))
        result=dict(world=32,window=512,samples_per_probe=512,start_step=start,timing=timing,
            random_mean_s=a,aligned_mean_s=b,time_reduction=1-b/a,strict_resume_bitwise_equal=equivalence,
            adopt=1-b/a>=.05 and all(equivalence.values()),formal_probe_steps_counted=0,consumed_prefix_preserved=True)
        atomic_json(work/'RESULT.json',result,immutable=True)
        if result['adopt']:
            path,checkpoint=prepare('selected_duration','duration',new_output)
            atomic_json(old/'DURATION_REPLACEMENT.json',dict(output=str(new_output),config=str(path),benchmark=str(work/'RESULT.json'),preserved_steps=start),immutable=True)
            del saved
            status('aligned_training',output=str(new_output),resume_step=start,result=result);selected_started=True
            execute(path,checkpoint,None,'formal_duration')
            final=read(new_output/'status.json')
        else:
            (old/'STOP').unlink();stopped=False;del saved
            status('original_training',resume_step=start,result=result);selected_started=True
            execute(config,original_path,None,'formal_original',aligned=False);final=read(old/'status.json')
        status('complete' if final['complete'] else 'stopped',step=final['step'])
    except Exception as exc:
        atomic_json(work/'ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()))
        if stopped and original_path is not None and not selected_started:
            (old/'STOP').unlink(missing_ok=True)
            status('benchmark_failed_original_resuming',error=repr(exc))
            execute(config,original_path,None,'fallback_original',aligned=False)
        else:status('failed',error=repr(exc))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--cluster',required=True,type=Path);parser.add_argument('--work',required=True,type=Path)
    parser.add_argument('--new-output',required=True,type=Path);args=parser.parse_args()
    run(args.config.resolve(),read(args.cluster),args.work.resolve(),args.new_output.resolve())
