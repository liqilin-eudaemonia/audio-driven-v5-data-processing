"""Restart one authorized raw node supervisor, adopting only this run's workers."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_fusion import read,atomic,lock

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(')')[-1].split()[0]!='Z'
    except FileNotFoundError:return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--node',required=True);a=p.parse_args();run=a.run.resolve();cfg=read(run/'config.json');folder=run/'nodes'/a.node;folder.mkdir(parents=True,exist_ok=True)
    jobs=cfg['nodes'][a.node]['jobs'];child=None;record=read(folder/'guardian.json');pid=record.get('supervisor_pid');started=time.time()
    with lock(folder/'guardian.lock') as got:
        if not got:raise RuntimeError('Raw node guardian already active')
        while not (run/'STOP').exists():
            finished=all(read(run/f'workers/{a.node}-g{j["gpu"]}.json').get('status')=='finished_available_shards' and not alive(read(folder/f'g{j["gpu"]}.json').get('pid',-1)) for j in jobs)
            if finished:atomic(folder/'guardian.json',dict(pid=os.getpid(),heartbeat=time.time(),status='all_workers_finished'));return
            if pid and alive(pid):
                cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode()
                if str(run) not in cmd or 'supervise_listen_full_raw.py' not in cmd or os.getpgid(pid)!=pid:raise RuntimeError('Recorded supervisor is not owned')
                last=read(folder/'latest.json').get('utc',started)
                if time.time()-max(last,started)>900:os.killpg(pid,signal.SIGTERM)
            else:
                if child is not None:child.wait()
                cmd=[cfg['python']['vllm'],str(run/'code/scripts/supervise_listen_full_raw.py'),'--run',str(run),'--node',a.node,'--adopt-existing']
                with (folder/f'supervisor-guardian-{time.time_ns()}.log').open('a') as log:child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                pid=child.pid;started=time.time()
            atomic(folder/'guardian.json',dict(pid=os.getpid(),supervisor_pid=pid,heartbeat=time.time(),gpus=[j['gpu'] for j in jobs],status='running'))
            time.sleep(10)

if __name__=='__main__':main()
