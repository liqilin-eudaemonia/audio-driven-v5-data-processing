"""Own only this run's GPU children; persist CPU/GPU/IO telemetry and restart alerts."""
import argparse, json, os, socket, subprocess, sys, time, signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json

def cpu():
    return [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]


class Adopted:
    def __init__(self,pid,statefile):self.pid=pid;self.statefile=statefile
    def poll(self):
        try:
            if Path(f'/proc/{self.pid}/stat').read_text().split(')')[-1].split()[0]!='Z':return None
        except FileNotFoundError:pass
        try:return 0 if json.loads(self.statefile.read_text()).get('status')=='finished_available_shards' else 1
        except (OSError,ValueError):return 1
    @property
    def returncode(self):return self.poll()


def verify_owned(pid,run):
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode()
    if 'run_listen_full_raw.py' not in cmd or str(run) not in cmd or os.getpgid(pid)!=pid:
        raise RuntimeError('Refusing to adopt/stop an unrelated process')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--node',required=True)
    p.add_argument('--only-gpus',default='');p.add_argument('--branch-override',choices=['visual','audio','asr'])
    p.add_argument('--adopt-existing',action='store_true');a=p.parse_args()
    run=Path(a.run).resolve();cfg=json.loads((run/'config.json').read_text());jobs=cfg['nodes'][a.node]['jobs']
    suffix=''
    if a.only_gpus:
        selected={int(x) for x in a.only_gpus.split(',')};jobs=[j for j in jobs if j['gpu'] in selected]
        suffix='-recovery-'+a.only_gpus.replace(',','-')
    if a.branch_override:
        if not a.only_gpus:raise ValueError('Branch handoff requires explicit GPU selection')
        jobs=[dict(j,branch=a.branch_override,batch=64 if a.branch_override=='visual' else 128,
                   prepare_workers=16 if a.branch_override=='visual' else 8) for j in jobs]
        suffix+='-'+a.branch_override
    node=run/'nodes'/a.node;node.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(node/('supervisor'+suffix+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Never preempt another job, including after a restart.
    used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True,timeout=20)
    occupied={int(line.split(',')[0]):float(line.split(',')[1]) for line in used.splitlines()}
    children={};restarts={};ended=set();current_branch={}
    if a.adopt_existing:
        for job in jobs:
            gpu=job['gpu'];record=node/f'g{gpu}.json'
            if not record.exists():continue
            info=json.loads(record.read_text());child=Adopted(info['pid'],run/'workers'/f'{a.node}-g{gpu}.json')
            if child.poll() is None:
                verify_owned(child.pid,run);children[gpu]=child;current_branch[gpu]=info['branch']
    if any(occupied[j['gpu']]>256 and j['gpu'] not in children for j in jobs):raise RuntimeError('GPU not idle; refusing to preempt: '+used)
    atomic_json(node/('supervisor'+suffix+'.json'),dict(pid=os.getpid(),host=socket.gethostname(),started_at=time.time()))
    last=cpu();previous=time.time()
    with (node/('telemetry'+suffix+'.jsonl')).open('a',buffering=1) as telemetry:
        while not (run/'STOP').exists():
            assignments=json.loads((run/'allocations.json').read_text()) if (run/'allocations.json').exists() else {}
            for original_job in jobs:
                job=dict(original_job)
                desired=assignments.get(f"{a.node}-g{job['gpu']}",job['branch'])
                job.update(branch=desired,batch=64 if desired=='visual' else 128,
                           prepare_workers=16 if desired=='visual' else 8)
                gpu=job['gpu'];worker=f'{a.node}-g{gpu}';child=children.get(gpu)
                if gpu in current_branch and current_branch[gpu]!=desired:
                    if child is not None and child.poll() is None:
                        verify_owned(child.pid,run);os.killpg(child.pid,signal.SIGTERM)
                        # Wait for the previous process and CUDA context to exit.
                        continue
                    ended.discard(gpu);restarts[gpu]=0;children.pop(gpu,None);child=None
                    current_branch[gpu]=desired
                if gpu in ended or child is not None and child.poll() is None:continue
                attempts=restarts.get(gpu,0)
                if child is not None:
                    if child.returncode==0:
                        ended.add(gpu);continue
                    attempts+=1;restarts[gpu]=attempts
                    if attempts>2:
                        ended.add(gpu);atomic_json(node/f'alert-g{gpu}.json',dict(worker=worker,reason='repeated worker failures',exitcode=child.returncode));continue
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
                    HF_ENABLE_PARALLEL_LOADING='true',HF_PARALLEL_LOADING_WORKERS='4',VLLM_WORKER_MULTIPROC_METHOD='spawn')
                python=cfg['python']['asr' if job['branch']=='asr' else 'vllm']
                if job['branch']!='asr':env['LD_LIBRARY_PATH']='/private/software/conda/envs/vllm/lib:'+env.get('LD_LIBRARY_PATH','')
                cmd=[python,str(ROOT/'scripts/run_listen_full_raw.py'),'--run',str(run),'--branch',job['branch'],
                     '--worker',worker,'--rank',str(job['rank']),'--batch',str(job['batch']),
                     '--prepare-workers',str(job['prepare_workers']),'--memory','.90']
                logfile=node/f'g{gpu}{suffix}-{job["branch"]}-{int(time.time())}-attempt{attempts}.log'
                with logfile.open('a') as log:
                    child=subprocess.Popen(cmd,env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children[gpu]=child
                current_branch[gpu]=desired
                atomic_json(node/f'g{gpu}.json',dict(worker=worker,pid=child.pid,command=cmd,branch=job['branch'],attempt=attempts,log=str(logfile)))
            now=time.time();curr=cpu();diff=[x-y for x,y in zip(curr,last)];last=curr
            row=dict(utc=now,host=socket.gethostname(),node=a.node,cpu_busy_percent=100*(sum(diff)-diff[3]-diff[4])/max(1,sum(diff)),
                cpu_iowait_percent=100*diff[4]/max(1,sum(diff)),load_average=os.getloadavg(),interval_s=now-previous,
                meminfo={l.split(':')[0]:l.split(':')[1].strip() for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:','Cached:'))},
                children={str(g):dict(pid=c.pid,returncode=c.poll()) for g,c in children.items()})
            previous=now
            try:
                query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw','--format=csv,noheader,nounits'],text=True,timeout=15)
                row['gpus']=[dict(zip(['gpu','memory_used_mib','memory_total_mib','gpu_util','memory_util','power_w'],[float(x.strip()) for x in line.split(',')])) for line in query.splitlines()]
            except Exception as exc:row['telemetry_error']=repr(exc)
            telemetry.write(json.dumps(row)+'\n');atomic_json(node/('latest'+suffix+'.json'),row)
            if len(ended)==len(jobs):break
            time.sleep(10)
    atomic_json(node/('finished'+suffix+'.json'),dict(utc=time.time(),stopped=(run/'STOP').exists()))

if __name__=='__main__':main()
