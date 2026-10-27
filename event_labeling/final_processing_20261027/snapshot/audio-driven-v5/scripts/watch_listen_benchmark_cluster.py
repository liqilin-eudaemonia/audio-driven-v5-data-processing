#!/usr/bin/env python3
"""Record actual three-node utilization and refresh version-separated results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--ssh-key',required=True)
    a=p.parse_args();root=Path(a.run);plan=json.loads((root/'full50_plan.json').read_text())
    previous={}
    command="""import json,subprocess
from pathlib import Path
print(json.dumps(dict(gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw','--format=csv,noheader,nounits'],text=True),cpu=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:])),loadavg=Path('/proc/loadavg').read_text().strip())))
"""
    def probe(item):
        name,node=item
        try:
            result=subprocess.run(['ssh','-i',a.ssh_key,'-o','BatchMode=yes','-o','ConnectTimeout=5','root@'+node['host'],
                                   '/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python -'],
                                  input=command,text=True,capture_output=True,timeout=15,check=True)
            data=json.loads(result.stdout);cpu=data.pop('cpu');current=(sum(cpu[:8]),cpu[3]+cpu[4]);last=previous.get(name)
            data['cpu_percent']=None if last is None else 100*(1-(current[1]-last[1])/max(1,current[0]-last[0]));previous[name]=current
            data['gpus']=[dict(zip(('index','memory_used_mib','memory_total_mib','gpu_percent','memory_percent','power_w'),map(float,line.split(',')))) for line in data['gpus'].strip().splitlines()]
            return name,data
        except Exception as exc:return name,dict(error=str(exc))
    with ThreadPoolExecutor(max_workers=3) as pool, (root/'cluster_resources.jsonl').open('a',buffering=1) as log:
        while not (root/'STOP_MONITOR').exists():
            record=dict(utc=datetime.now(timezone.utc).isoformat(),nodes=dict(pool.map(probe,plan['nodes'].items())))
            log.write(json.dumps(record)+'\n');tmp=root/'cluster_resources_latest.tmp';tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(root/'cluster_resources_latest.json')
            with (root/'aggregate.log').open('a') as output:
                subprocess.run([sys.executable,str(Path(__file__).with_name('aggregate_listen_benchmark_cluster.py')),'--run',str(root)],stdout=output,stderr=subprocess.STDOUT,timeout=60)
            time.sleep(15)


if __name__=='__main__':main()
