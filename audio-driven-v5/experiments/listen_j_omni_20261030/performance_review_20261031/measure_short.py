import sys,json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;label=sys.argv[1];duration=sys.argv[2]
p=subprocess.run([sys.executable,str(W/'nod_cpu_share/measure.py'),label,duration],capture_output=True,text=True)
(D/(label+'.log')).write_text(p.stdout+p.stderr)
if p.returncode:print(p.stderr[-2000:]);sys.exit(p.returncode)
reports=[json.loads((W/'nod_cpu_share'/label/(n+'.json')).read_text()) for n in ['node17','node8','node4','node12']]
print(json.dumps(dict(label=label,total_clips_per_second=sum(x['clips_per_second'] for x in reports),mean_gpu_util=sum(x['mean_gpu_util'] for x in reports)/4,nodes=[{k:x.get(k) for k in ['node','clips_per_second','mean_gpu_util','cpu_cores','nod_clips_per_second']} for x in reports])))
