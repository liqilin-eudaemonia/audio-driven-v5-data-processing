"""Low-overhead resource and completion telemetry for worker scaling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]/'data_corpora/seamless_listen_v7_full'
OUT=ROOT/'scheduling_v2/telemetry'


def cpu_sample():
    allowed=set(os.sched_getaffinity(0));total=idle=iowait=0
    for line in Path('/proc/stat').read_text().splitlines():
        key,*values=line.split()
        if key.startswith('cpu') and key[3:].isdigit() and int(key[3:]) in allowed:
            v=list(map(int,values));total+=sum(v[:8]);idle+=v[3]+v[4];iowait+=v[4]
    return total,idle,iowait


def sample(node):
    a=cpu_sample();time.sleep(2);b=cpu_sample();elapsed=b[0]-a[0]
    completed=errors=claimed=0;active=0;wait_s=scan_s=0.
    for path in (ROOT/'progress').glob('rank_*.json'):
        r=json.loads(path.read_text())
        if (r['rank']%24)//8!=node or r.get('scheduling')!='dynamic_v2':continue
        completed+=r['counts'].get('complete_pairs',0);errors+=r['counts'].get('error_pairs',0)
        claimed+=r['counts'].get('claimed_pairs',0)
        active+=not r['stage'].startswith('finished')
        wait_s+=r.get('timing_seconds',{}).get('waiting_for_preparation',0)
        scan_s+=r.get('timing_seconds',{}).get('audio_screening',0)
    result=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    gpus=[dict(zip(['gpu','utilization_percent','memory_mib'],map(int,line.split(','))))
          for line in result.stdout.strip().splitlines()]
    return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),unix=time.time(),node=node,
        allowed_cpus=len(os.sched_getaffinity(0)),cpu_busy_percent=100*(1-(b[1]-a[1])/elapsed),
        cpu_iowait_percent=100*(b[2]-a[2])/elapsed,gpus=gpus,active_model_workers=active,
        cpu_preparation_slots=active*4,completed_pairs=completed,error_pairs=errors,claimed_pairs=claimed,
        cumulative_wait_seconds=wait_s,cumulative_scan_seconds=scan_s)


def main(node,duration):
    OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    while time.monotonic()-start<duration:
        begin=time.monotonic();value=sample(node)
        with (OUT/f'node_{node}.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        target=OUT/f'node_{node}.json';tmp=target.with_suffix('.pending')
        tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(target)
        if (ROOT/'screening_summary.json').exists() and json.loads((ROOT/'screening_summary.json').read_text()).get('workers_finished'):break
        time.sleep(max(0,15-(time.monotonic()-begin)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',type=int,required=True);p.add_argument('--duration',type=int,default=3600)
    a=p.parse_args();main(a.node,a.duration)
