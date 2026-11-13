"""One-shot checkpoint-boundary switch; preserve optimizer/RNG and stop only owned jobs."""
import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,atomic_json
spec=importlib.util.spec_from_file_location('planning_cluster',ROOT/'scripts/p2_planning_cluster.py')
cluster_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cluster_module)


def remote(node,cluster,code):
    cmd=shlex.join([cluster['python'],'-c',code])
    out=subprocess.run(cluster_module.ssh(node,cluster)+[cmd],text=True,capture_output=True,timeout=40)
    if out.returncode:raise RuntimeError(node['name']+': '+out.stderr[-2000:])
    return out.stdout.strip()


def checkpoint_files(checkpoint,world):
    names=['adapter_config.json','adapter_model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json','training_args.bin']+[f'rng_state_{i}.pth' for i in range(world)]
    if not all((checkpoint/n).is_file() and (checkpoint/n).stat().st_size>0 for n in names):return None
    try:read(checkpoint/'trainer_state.json')
    except (ValueError,OSError):return None
    return {n:dict(bytes=(checkpoint/n).stat().st_size,mtime_ns=(checkpoint/n).stat().st_mtime_ns) for n in names}


def main():
    p=argparse.ArgumentParser();p.add_argument('--old-config',required=True);p.add_argument('--new-config',required=True);p.add_argument('--cluster',required=True);p.add_argument('--step',type=int,default=500);p.add_argument('--dry-run',action='store_true');p.add_argument('--capture-only',action='store_true');a=p.parse_args()
    old_config=read(a.old_config);new_config=read(a.new_config);cluster=read(a.cluster);parent=Path(old_config['output']);new_root=Path(new_config['output'])
    work=parent/'performance';report=work/'input_equivalence50.json';state_path=work/'migration_state.json'
    old_identity=read(parent/'run.json');old_sha=sha(parent/'run.json')
    if old_identity['config']!=old_config:raise ValueError('Old live configuration changed')
    if new_root==parent or (new_root/'run.json').exists():raise ValueError('New output is not fresh')
    for name,h in old_identity['code'].items():
        if sha(work/'original_source'/name)!=h:raise ValueError('Original implementation snapshot changed')
    report_data=read(report)
    if not report_data['success'] or report_data['count']<50:raise ValueError('Equivalence checks incomplete')
    if report_data['pipeline']!=new_config['data']['input_pipeline']:raise ValueError('Pipeline differs from real verification')
    for name,h in report_data['code'].items():
        if sha(ROOT/'src/p2_planning_sft'/name)!=h:raise ValueError('Input code changed after tests')
    launcher=read(parent/'launcher.json')
    if launcher['host']!=os.uname().nodename:raise ValueError('Run this switch on the original launcher host')
    logdirs=sorted((parent/'launch_logs').iterdir());launch_dir=logdirs[-1]
    owned={n['name']:read(launch_dir/(n['name']+'.pid.json')) for n in cluster['nodes']}
    checkpoint=parent/f'checkpoint-{a.step}'
    def status(stage,**extra):
        value=dict(stage=stage,time=time.time(),parent=str(parent),output=str(new_root),checkpoint=str(checkpoint),**extra)
        atomic_json(state_path,value);print(json.dumps(value),flush=True)
    if a.dry_run:
        print(json.dumps(dict(validated=True,checkpoint=str(checkpoint),gpu_action=False,report=sha(report))));return
    status('waiting_for_checkpoint',target_step=a.step)
    frozen_pid=None;stopped=False
    try:
        # Catch rank0 immediately after it serializes trainer state. Other ranks
        # can finish their own RNG writes while rank0 waits at this boundary.
        while True:
            if (checkpoint/'trainer_state.json').is_file():
                try:state=read(checkpoint/'trainer_state.json')
                except (ValueError,OSError):time.sleep(.1);continue
                if state['global_step']!=a.step:raise ValueError('Wrong checkpoint step')
                leader=cluster['nodes'][0];agent=owned[leader['name']]['pid']
                code=f'''import os,signal
from pathlib import Path
parent={agent}
children=Path(f'/proc/{{parent}}/task/{{parent}}/children').read_text().split()
found=[]
for value in children:
 p=Path('/proc')/value
 try:
  env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\\0') if b'=' in x)
  cmd=(p/'cmdline').read_bytes()
  if env.get(b'RANK')==b'0' and b'p2_planning_sft.py' in cmd:
   os.kill(int(value),signal.SIGSTOP);found.append(int(value))
 except FileNotFoundError:pass
assert len(found)==1,found
print(found[0])'''
                frozen_pid=int(remote(leader,cluster,code));break
            if (parent/'COMPLETE.json').exists():raise RuntimeError('Parent completed without requested checkpoint')
            os.kill(launcher['pid'],0);time.sleep(.2)
        status('checkpoint_boundary_frozen',rank0_pid=frozen_pid)
        deadline=time.time()+180;last=None;stable=0
        while time.time()<deadline:
            current=checkpoint_files(checkpoint,old_identity['world_size'])
            if current is not None and current==last:stable+=1
            else:stable=0
            if stable>=3:break
            last=current;time.sleep(.3)
        else:raise RuntimeError('Checkpoint did not finish')
        records={}
        for name,st in current.items():
            records[name]=dict(st,sha256=sha(checkpoint/name))
            now=(checkpoint/name).stat()
            if (now.st_size,now.st_mtime_ns)!=(st['bytes'],st['mtime_ns']):raise RuntimeError('Checkpoint still changing')
        receipt=work/f'checkpoint-{a.step}-receipt.json'
        atomic_json(receipt,dict(checkpoint=str(checkpoint),parent_run_sha256=old_sha,files=records),immutable=True)
        new_config['performance_resume']=dict(checkpoint=str(checkpoint),parent_run_sha256=old_sha,step=a.step,
            checkpoint_receipt=str(receipt),equivalence_report=str(report),equivalence_report_sha256=sha(report))
        atomic_json(a.new_config,new_config)
        # Validate full new identity before stopping the old run.
        from p2_planning_sft.reference_data import datasets
        from p2_planning_sft.performance_resume import validate_performance_resume
        from p2_planning_sft.data import SYSTEM,USER
        from p2_listen.contracts import digest
        train,dev,full=datasets(new_config)
        identity=dict(config=new_config,smoke=False,train_manifest=train.manifest_sha256,dev_manifest=dev.manifest_sha256,
            train_ids=digest(train.ids),train_count=len(train),dev_count=len(dev),system=SYSTEM,user=USER,
            code={f.name:sha(f) for f in (ROOT/'src/p2_planning_sft').glob('*.py')},world_size=32,
            reference_data=dict(train=train.identity,diagnostic_dev=dev.identity,full_dev=full.identity),full_dev_count=len(full))
        validation=validate_performance_resume(new_config,identity,new_root)
        status('checkpoint_verified',migration=validation)
        def stop(node):
            owner=owned[node['name']]['supervisor']
            # Identify the wrapper by both script and original config, not PID alone.
            code=f'''import os,signal,time
from pathlib import Path
pid={owner};p=Path('/proc')/str(pid)
if p.exists():
 cmd=(p/'cmdline').read_bytes()
 assert b'p2_planning_cluster.py' in cmd and {str(Path(a.old_config).resolve()).encode()!r} in cmd
 os.kill(pid,signal.SIGTERM)
'''
            if node['name']==cluster['nodes'][0]['name']:
                code+=f'os.kill({frozen_pid},signal.SIGCONT)\n'
            return remote(node,cluster,code)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(stop,cluster['nodes']))
        stopped=True;frozen_pid=None
        # The launcher exits after remote wrappers terminate. Do not start a
        # second process group on a still-running rank or occupied rendezvous.
        deadline=time.time()+100
        while time.time()<deadline:
            p=Path('/proc')/str(launcher['pid'])
            if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
            time.sleep(1)
        else:raise RuntimeError('Original launcher has not exited')
        if a.capture_only:
            status('checkpoint_captured_parent_stopped',checkpoint_preserved=True)
            return
        new_root.mkdir(parents=True,exist_ok=True)
        command=[cluster['python'],'-u',str(ROOT/'scripts/p2_planning_cluster.py'),'--config',str(Path(a.new_config).resolve()),'--cluster',str(Path(a.cluster).resolve()),'--launch']
        with (new_root/'launcher.log').open('ab') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(new_root/'launcher.json',dict(pid=process.pid,host=os.uname().nodename,started_unix=time.time(),command=command))
        status('accelerated_run_launched',launcher_pid=process.pid,checkpoint_preserved=True)
    except BaseException as error:
        if frozen_pid is not None:
            remote(cluster['nodes'][0],cluster,f'import os,signal; os.kill({frozen_pid},signal.SIGCONT)')
        status('failed',error=repr(error),parent_stopped=stopped)
        raise

if __name__=='__main__':main()
