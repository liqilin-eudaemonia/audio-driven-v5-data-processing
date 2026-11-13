"""Wait for idle32, benchmark three exact executions, verify18 real updates, stop."""
import argparse,concurrent.futures,fcntl,json,math,os,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json,sha
from listen_cluster import transport,launch
from p2_j_whisper_event_feature import commands
DEFAULT=ROOT/'experiments/p2_j_whisper_event_feature_20261105/test32'


def resources(cluster):
    def one(node):
        code="""import json,os,subprocess
from pathlib import Path
p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20,check=True)
rows=[]
for line in p.stdout.splitlines():
 g,used,total,util,temp=line.split(',');rows.append(dict(gpu=int(g),used_mib=float(used),total_mib=float(total),util=float(util),temperature=float(temp)))
print(json.dumps(dict(gpus=rows,cpu_stat=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]],load=list(os.getloadavg()),cpus=os.cpu_count())))
"""
        try:
            result=subprocess.run(transport(cluster,node,[cluster['python'],'-c',code]),capture_output=True,text=True,timeout=40,check=True)
            value=json.loads(result.stdout);value['node']=node['name'];value['host']=node['host']
            value['idle']=len(value['gpus'])==8 and all(x['used_mib']<2048 and x['util']<10 for x in value['gpus'])
            return value
        except Exception as exc:return dict(node=node['name'],idle=False,error=repr(exc))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,cluster['nodes']))
    return dict(at=time.time(),all_idle=len(rows)==4 and all(r['idle'] for r in rows),nodes=rows)


def run(a):
    from p2_whisper_event_feature.config import load,source_identity,validate_baseline
    from p2_whisper_event_feature.data import data_identity
    c=load(a.config);cluster=read(a.cluster);out=Path(c['output']);a.work.mkdir(parents=True,exist_ok=True)
    def state(stage,**kw):
        value=dict(stage=stage,at=time.time(),**kw);atomic_json(a.work/'state.json',value);print(json.dumps(value),flush=True)
    if (out/'run.json').exists():raise RuntimeError('Test driver requires a fresh combined run; never overwrite an existing checkpoint')
    if (a.work/'COMPLETE.json').exists():raise RuntimeError('32-card test already finished')
    tests=(out/'cpu_tests_optimization.log').read_text()
    if '73 passed' not in tests:raise RuntimeError('Run combined CPU/prefetch regressions first')
    if not read(out/'CPU_PREFETCH_CHECK.json')['success']:raise RuntimeError('Real CPU prefetch verification incomplete')
    pre=read(out/'preflight.json')
    if pre['source']!=source_identity() or pre['data_identity']!=data_identity(c):raise RuntimeError('Exact full preflight required')
    atomic_json(a.work/'controller.json',dict(pid=os.getpid(),script=str(Path(__file__).resolve()),config=str(a.config)))
    quiet=0;began=time.time()
    while quiet<2:
        if (a.work/'STOP').exists() or (out/'STOP').exists():state('stopped_before_test');return
        snapshot=resources(cluster);atomic_json(a.work/'resources.json',snapshot)
        quiet=quiet+1 if snapshot['all_idle'] else 0
        if quiet<2:
            state('waiting_for_existing_cache_workers',idle_gpus=sum(x['used_mib']<2048 and x['util']<10 for n in snapshot['nodes'] for x in n.get('gpus',[])),quiet_polls=quiet)
            if time.time()-began>7200:raise TimeoutError('32 GPUs did not become idle within2h')
            time.sleep(20)
    def execute(cmds,logs,stage,progress_path):
        cancel=threading.Event();done=threading.Event();started=time.time()
        def monitor():
            while not done.wait(20):
                try:
                    snap=resources(cluster);atomic_json(a.work/'resources.json',snap)
                    with (a.work/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,**snap))+'\n')
                    age=time.time()-(progress_path.stat().st_mtime if progress_path.exists() else started)
                    if (a.work/'STOP').exists() or (out/'STOP').exists() or age>1200:cancel.set();return
                except Exception as exc:atomic_json(a.work/'monitor_error.json',dict(error=repr(exc),at=time.time()))
        worker=threading.Thread(target=monitor,daemon=True);worker.start()
        try:launch(cluster,cmds,logs,cancel_event=cancel)
        finally:done.set();worker.join(timeout=45)
    benchmark=a.work/'benchmark';bench_commands=commands(cluster,a.config)
    for command in bench_commands:
        command[command.index(str(ROOT/'scripts/p2_j_whisper_event_feature.py'))]=str(ROOT/'scripts/p2_whisper_event_feature_benchmark.py')
        command.extend(['--work',str(benchmark),'--steps','16','--start-step','3'])
    atomic_json(a.work/'benchmark_commands.json',bench_commands)
    state('benchmarking32',matched_samples=512,candidates=3)
    execute(bench_commands,a.work/'benchmark_logs','benchmark',benchmark/'progress.json')
    result=read(benchmark/'RESULT.json');baseline=result['results'][0];best=result['selected']
    if not result['success'] or result['formal_updates']!=0:raise RuntimeError('Invalid matched benchmark result')
    # A tiny timing fluctuation is insufficient to change the execution defaults.
    selected=best if best['mean_seconds']<=baseline['mean_seconds']*.97 else baseline
    c['combined_execution']=dict(prefetch_batches=selected['prefetch_batches'],gradient_bucket_mb=selected['bucket_mb'])
    validate_baseline(c,read(c['baseline_config']));atomic_json(a.config,c)
    atomic_json(a.work/'EXECUTION_SELECTED.json',dict(selected=selected,baseline=baseline,
                speedup=baseline['mean_seconds']/selected['mean_seconds'],minimum_improvement=.03,config_sha256=sha(a.config)))
    state('real18_updates32',selected=selected['name'],mean_benchmark_seconds=selected['mean_seconds'])
    # The benchmark released all its workers before this stage. Same configured
    # initialization, full real Adam updates and strict complete checkpoints.
    execute(commands(cluster,a.config,stop_after=18),a.work/'training18_logs','training18',out/'status.json')
    rows=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
    if [r['step'] for r in rows]!=list(range(1,19)):raise RuntimeError('Real18 updates incomplete')
    for row in rows:
        if not all(math.isfinite(row[k]) for k in ['av','event','feature_video','feature_audio','weighted_total']):raise RuntimeError('Nonfinite joint objective')
        expected=row['av']+.05*row['event']+.025*(row['feature_video']+row['feature_audio'])
        if abs(expected-row['weighted_total'])>1e-8:raise RuntimeError('Incorrect joint objective normalization')
    for name in ['connector','events','features','text_lora','whisper']:
        if not any(math.isfinite(r['grad_norm'][name]) and r['grad_norm'][name]>0 for r in rows):raise RuntimeError('Missing real gradient: '+name)
    from p2_whisper_event_feature.checkpoint import checked
    saved=checked(out/'step_000018.pt')
    if saved['step']!=18 or saved['identity']['world']!=32 or saved['identity']['experiment']!=c:raise RuntimeError('Real checkpoint identity mismatch')
    actual=[r['seconds'] for r in rows[2:]]
    report=dict(success=True,world=32,verified_real_updates=18,checkpoint=str(out/'step_000018.pt'),
       checkpoint_sha256=sha(out/'step_000018.pt'),five_optimizer_groups_receive_gradients=True,
       objective_correct=True,selected=selected['name'],real_mean_seconds_grouped=sum(actual)/len(actual),
       real_step_seconds_grouped=actual,longest_sample=read(out/'LONGEST_SAMPLE_CHECK.json'),
       estimated_remaining_hours=(8927-18)*(sum(actual)/len(actual))/3600,
       formal_full_training_started=False,training_stopped_after_test=True,at=time.time())
    atomic_json(a.work/'COMPLETE.json',report);state('complete_test_stopped',**{k:report[k] for k in ['selected','real_mean_seconds_grouped','estimated_remaining_hours']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/p2_j_qwen_whisper_event_feature_20261105.json')
    p.add_argument('--cluster',type=Path,default=ROOT/'configs/p2_j_qwen_whisper_event_feature_32gpu_20261105.json')
    p.add_argument('--work',type=Path,default=DEFAULT);a=p.parse_args()
    a.config=a.config.resolve();a.work=a.work.resolve();a.work.mkdir(parents=True,exist_ok=True)
    with (a.work/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:
            atomic_json(a.work/'FAILED.json',dict(at=time.time(),error=repr(exc)));raise
